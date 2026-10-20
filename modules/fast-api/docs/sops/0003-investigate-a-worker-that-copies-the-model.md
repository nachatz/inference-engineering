# 0003 · Investigate a worker that copies the model

| owner | last run | applies to |
| --- | --- | --- |
| @nachatz | 2026-10-18 | gunicorn with preload on Linux |

## When

- `tests/integration/test_gunicorn.py::TestPreload::test_workers_share_the_model` fails.
- [SOP 0001](0001-measure-worker-memory.md) shows a worker with more than 30% of the model as
  private memory.
- A pod's memory grows with its worker count, or with time.

## Before you start

- A server running with the failing configuration, and its master pid: `pgrep -o gunicorn`.

## Steps

1. Measure, so you know which workers and how much: [SOP 0001](0001-measure-worker-memory.md).

2. Work down the list; each line is a way the copy-on-write sharing has broken before.

   | check | how | fix |
   | --- | --- | --- |
   | preload is on | `curl /debug/flags`: `preload` is true; the master's PSS includes the model | set `GUNICORN_PRELOAD=1` |
   | the model loads at import, not in the lifespan | the model is loaded in [`app/core/preload.py`](../../app/core/preload.py) only | move it; [ADR 0005](../adrs/0005-per-worker-state-after-the-fork.md) |
   | warm-up runs before the fork | `MODEL_WARMUP_RUNS` ≥ 1 | set it |
   | `gc.freeze()` runs after the app is built | `curl /debug/runtime`: `gc_frozen_objects` above 0 | restore it in [`app/main.py`](../../app/main.py) |
   | one thread per session | `curl /debug/flags`: `intra_op_threads` is 1; `/debug/runtime`: few `native_threads` | set it; [ADR 0003](../adrs/0003-one-core-per-worker.md) |
   | no modules imported after the fork | step 4 | import them in [`gunicorn.conf.py`](../../gunicorn.conf.py) |
   | bytecode compiled at build time | `UV_COMPILE_BYTECODE=1` in the image | rebuild; [PMR 0002](../pmrs/0002-cold-start-workers-copied-server-modules.md) |

3. Find the mappings that hold a worker's private memory.

   ```bash
   awk '/^[0-9a-f]+-/ {name = ($6 == "" ? "[anon]" : $6)}
        /^Private_Dirty/ {dirty[name] += $2}
        END {for (n in dirty) print dirty[n] " kB", n}' /proc/<worker pid>/smaps | sort -rn | head
   ```

   A large `[anon]` or `[heap]` entry close to the model size means the weights were loaded or
   copied in the worker.

4. List the modules a worker imported after the fork, the way [PMR
   0002](../pmrs/0002-cold-start-workers-copied-server-modules.md) did:
   serve a copy of the app with a route returning `sorted(set(sys.modules) - set(master_modules))`.

## Done when

- [SOP 0001](0001-measure-worker-memory.md) passes, and `just cov` passes on Linux, slow tests
  included.

## If it doesn't work

| symptom | likely cause | next step |
| --- | --- | --- |
| private memory grows slowly, not at start | something writes to shared objects as it serves | recycle workers with `GUNICORN_MAX_REQUESTS`; recycled workers fork from the master and share again |
| only the first worker after a deploy is large | cold start | [PMR 0002](../pmrs/0002-cold-start-workers-copied-server-modules.md) |
