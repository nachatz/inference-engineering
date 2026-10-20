# 0001 · Measure worker memory

| owner | last run | applies to |
| --- | --- | --- |
| @nachatz | 2026-10-18 | a running gunicorn on Linux; not GPU memory |

## When

- Before sizing a pod ([SOP 0002](0002-size-a-cpu-pod.md)).
- After changing the model, anything that runs at import, or anything in the lifespan.
- When a pod's memory grows with its worker count.

## Before you start

- Linux: the numbers come from `/proc/<pid>/smaps_rollup`.
- `pgrep` installed (`procps`).
- An exported model: `just model`.

Know the three numbers:

| number | counts | use it for |
| --- | --- | --- |
| RSS | every page a process maps, shared or not | nothing here: it counts the shared model once per worker |
| PSS | each shared page split between the processes sharing it | the real memory of the pod: add it up |
| private | pages no other process maps (`Private_Clean + Private_Dirty`) | what each extra worker costs |

`top`, `ps` and `kubectl top` report RSS-like numbers. Do not size from them.

## Steps

1. Start the server with the worker count you will run.

   ```bash
   GUNICORN_WORKERS=4 just serve
   ```

2. Send traffic, so every worker has run inference and touched what it will touch.

   ```bash
   LOAD_TEST_SECONDS=10 just benchmarks load-test
   ```

3. Read the memory of the master and every worker.

   ```bash
   just benchmarks mem            # the oldest gunicorn; MEM_PID=<master pid> to pick one
   ```

   Measured on the benchmark machine ([ADR
   0002](../adrs/0002-preload-the-model-and-fork-workers.md)), 4 workers:

   ```
   gunicorn master 685 and 4 workers
        pid  role       RSS MB    PSS MB  private MB
        685  master      457.5     119.3        32.4
        699  worker      432.0      97.1        13.8
        ...
      total             2186.7     508.5        88.0
   PSS total 0.51 GB, RSS total 2.19 GB
   ```

   In a container without `pgrep`, ask a worker about itself instead; each call lands on one worker.

   ```bash
   curl localhost:8000/debug/memory       # this worker's RSS, PSS and private bytes
   curl localhost:8000/debug/allocator    # the malloc actually loaded: jemalloc in the image
   ```

4. Compare each worker's private memory with the model file.

   ```bash
   ls -l model/model.onnx
   ```

## Done when

- Every worker's private memory is under 30% of the model file (the threshold the slow test
  enforces).
- Total PSS is about one model, plus the master, plus workers × private.

## If it doesn't work

| symptom | likely cause | next step |
| --- | --- | --- |
| private per worker ≈ the model size | the workers each loaded a copy | [SOP 0003](0003-investigate-a-worker-that-copies-the-model.md) |
| `LookupError: no gunicorn process is running` | the server isn't up, or runs under another name | start it, or set `MEM_PID` |
| `PermissionError` on `/proc/<pid>` | another user owns the process | run as that user |
| PSS grows slowly over hours | workers copy shared pages as they write to them | [SOP 0003](0003-investigate-a-worker-that-copies-the-model.md), consider `GUNICORN_MAX_REQUESTS` |
