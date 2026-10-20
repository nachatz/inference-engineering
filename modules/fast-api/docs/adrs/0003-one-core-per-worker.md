# 0003 · One core, one thread and one inference per worker

| status | date | owner | supersedes |
| --- | --- | --- | --- |
| accepted | 2026-10-18 | @nachatz | none |

## Context

ONNX Runtime defaults to an intra-op thread pool as wide as the machine.
With N workers that is N pools of N threads on N cores: N² runnable threads.
`fork()` copies only the calling thread, so a pool started before the fork is broken in every
worker.
`session.run` blocks; on the event loop it would stop the worker answering health checks.

## Decision

| rule | value | where |
| --- | --- | --- |
| workers | one per core: `GUNICORN_WORKERS`, else the cgroup CPU quota, else the CPUs the process may run on, with a startup warning | [`app/common/cpu.py`](../../app/common/cpu.py), [`gunicorn.conf.py`](../../gunicorn.conf.py) |
| threads per session | `MODEL_INTRA_OP_THREADS=1`, `inter_op_num_threads=1` | [`app/inference/runtime/onnx.py`](../../app/inference/runtime/onnx.py) |
| other native thread pools | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1, set before NumPy loads them in the master | [`Dockerfile`](../../Dockerfile), [`justfiles/app.just`](../../justfiles/app.just) |
| inferences at a time per worker | one, behind a bounded queue: `InferenceSlot.acquire()` | [`app/inference/capacity/slot.py`](../../app/inference/capacity/slot.py) |
| where inference runs | `anyio.to_thread.run_sync`, off the event loop, inside the slot | [`app/inference/handlers/predict.py`](../../app/inference/handlers/predict.py) |

## Alternatives

| option | why not |
| --- | --- |
| one worker, a wide thread pool | one event loop and one GIL for all requests; the pool's threads don't survive a fork |
| `os.cpu_count()` workers | counts the host's cores, not the container's quota; a 16-CPU pod on a 64-core node would run 64 workers |
| inference on the event loop | health checks wait behind inference, so probes fail under load |

## Consequences

- No thread pool exists when gunicorn forks.
- The event loop stays free, which has a side effect on connection balancing: see [ADR
  0004](0004-one-listening-socket-per-worker.md).
- Requests beyond one per worker queue for the slot; latency under load is queueing, visible as
  `predict_request_seconds` minus `predict_inference_seconds`.
- A pod with a CPU request but no CPU limit has no cgroup quota, so the worker count falls back to
  every CPU of the node: the failure `os.cpu_count()` was rejected for. gunicorn logs a warning at
  startup (`no CPU limit and no GUNICORN_WORKERS`); set a CPU limit or `GUNICORN_WORKERS`.
- JSON parsing still runs on the event loop. 64 rows of 1024 floats is about 1.4 MB to parse per
  request, and admission bounds how many parse at once, but a binary payload would remove that cost
  altogether: the subject of a later episode.
