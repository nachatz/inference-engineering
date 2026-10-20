# 0002 · Preload the model and fork the workers

| status | date | owner | supersedes |
| --- | --- | --- | --- |
| accepted | 2026-10-18 | @nachatz | none |

> ▶ Companion to the "One model, many workers" video in the Inference Engineering series on
> [YouTube (@nikachatz)](https://www.youtube.com/@nikachatz).

## Context

A CPU server runs one gunicorn worker per core, each a uvicorn event loop doing one inference at a
time.
By default each worker imports the app after the fork, so a model loaded at import is loaded once
per worker.
For the 353 MB model that is 1.4 GB for 4 workers and 5.6 GB for the 16 a production pod runs.

## Decision

Set `preload_app = True`: gunicorn imports the app once, in the master, then forks.
`fork()` shares the parent's pages copy-on-write, and inference only reads the weights, so every
worker shares one copy.

The minimal version, as shown in the video
([`examples/01-preload-prefork/`](../../examples/01-preload-prefork/)):

```python
import gc, numpy as np, onnxruntime as ort
from fastapi import FastAPI

def load_model(path="model.onnx"):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # one core per worker
    session = ort.InferenceSession(path, opts)
    session.run(None, {"x": np.zeros((1, 1024), np.float32)})   # warm-up
    return session

session = load_model()   # with preload_app: runs once, in the master
gc.freeze()              # GC won't touch (and copy) the shared pages
app = FastAPI()

@app.post("/predict")
async def predict(features: list[float]):
    x = np.asarray([features], np.float32)
    return {"label": int(session.run(None, {"x": x})[0].argmax())}
```

```python
bind = "0.0.0.0:8000"
workers = 4   # start near one per CPU core
worker_class = "uvicorn_worker.UvicornWorker"   # FastAPI in each worker
preload_app = True   # import app.py (and the model) once, then fork
```

The production version:

| rule | where |
| --- | --- |
| load and warm up the model at import, in the master | [`app/core/preload.py`](../../app/core/preload.py) |
| `gc.disable()` while the shared heap is built, `gc.freeze()` before the fork, `gc.enable()` in each worker | [`app/main.py`](../../app/main.py), [`app/apps/lifespan.py`](../../app/apps/lifespan.py) |
| never load the model in the lifespan, which runs in every worker | [`app/apps/lifespan.py`](../../app/apps/lifespan.py) |
| `preload_app` on by default, `GUNICORN_PRELOAD=0` to compare | [`gunicorn.conf.py`](../../gunicorn.conf.py) |
| import uvicorn's and anyio's lazy modules in the master | [`gunicorn.conf.py`](../../gunicorn.conf.py), [PMR 0002](../pmrs/0002-cold-start-workers-copied-server-modules.md) |

Why each rule matters:

- **Import time, not lifespan.** The lifespan runs after the fork, so a model loaded there is a copy
  per worker.
- **`gc.disable()`, then `gc.freeze()`, then `gc.enable()`.** CPython's documented pattern for
  prefork servers. With the GC off while the heap is built, no collection frees objects and leaves
  holes that later allocations write into; frozen objects are never scanned, and a scan writes to
  every object it visits. Each worker turns the GC back on in its lifespan.
  The weights live in ONNX Runtime's C++ heap, which the GC never touches; this protects the Python
  heap around them.
  It doesn't stop reference counting: touching a shared Python object still writes its count, and
  that copies its page. The ~14 MB private per worker below is that cost, and it stays small.
- **Warm-up in the master.** The first `session.run` allocates buffers; doing it before the fork
  shares them too.
- **No worker recycling.** `GUNICORN_MAX_REQUESTS` is 0. A replacement would be forked from the
  master and share the model, but a worker that exits gracefully after the fork aborts in a native
  destructor (`std::system_error`), so recycling isn't something to rely on yet.

## Alternatives

| option | why not |
| --- | --- |
| one model per worker (no preload) | memory scales with workers: measured ~406 MB private per worker below |
| one process, many threads | the GIL and one event loop; a crash takes down every request |
| a model server process, workers call it over IPC | an extra hop and serialization for every request; the right shape on GPUs, not CPUs |
| `mmap` the weights from disk | ONNX Runtime copies the weights into its own heap when it loads the model: +356 MB private for the 353 MB model, with the weights embedded or in an external data file, with prepacking on or off. Measured on Windows; check on Linux before quoting it there |

## Consequences

Measured by `just bench` ([`bench/preload/benchmark.py`](../../bench/preload/benchmark.py)), raw
runs in
[`bench/results/cpu-preload.json`](../../bench/results/cpu-preload.json):

- Machine: Docker Desktop's Linux VM (WSL2) on Windows 11, AMD Ryzen 7 9800X3D (8 cores, 96 MB L3).
- 8 vCPUs and 16.3 GB RAM for the VM, Python 3.13.16, ONNX Runtime 1.31.0, the 353 MB model.
- Each row is the median of 3 runs: a 10 s headless locust run, 16 users, no wait, a new connection
  per request.
- The load generator runs on the same 8 vCPUs. Locust rounds latency percentiles to two significant
  figures.

| workers | preload | req/s | p50 ms | p99 ms | PSS MB | RSS MB | private/worker MB | ready s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | no | 142.9 | 110 | 130 | 458 | 499 | 408.9 | 0.60 |
| 1 | yes | 145.4 | 110 | 120 | 457 | 893 | 15.6 | 0.61 |
| 2 | no | 161.7 | 100 | 180 | 870 | 954 | 406.2 | 0.66 |
| 2 | yes | 236.2 | 62 | 150 | 473 | 1326 | 14.8 | 0.61 |
| 4 | no | 168.0 | 85 | 250 | 1684 | 1861 | 405.7 | 0.86 |
| 4 | yes | 302.8 | 42 | 140 | 502 | 2190 | 14.4 | 0.61 |
| 8 | no | 166.9 | 80 | 250 | 3304 | 3674 | 405.0 | 0.98 |
| 8 | yes | 291.5 | 46 | 150 | 555 | 3914 | 13.7 | 0.61 |

As shown in the video, on a 2-vCPU Linux sandbox (ONNX Runtime 1.29, the same model, 4 workers):

| | without preload | with preload |
| --- | ---: | ---: |
| PSS, all processes | 1.64 GB | 0.48 GB |
| private memory per worker | 397 MB | ~10 MB |
| throughput | 76 req/s | 94 req/s |
| ready in | 3.0 s | 1.6 s |

What follows:

- **Memory.** A worker costs ~14 MB instead of ~406 MB; a pod holds one model copy. Size pods with
  [SOP 0002](../sops/0002-size-a-cpu-pod.md).
- **RSS is the wrong number.** It counts the shared model once per process, so it goes *up* with
  preload. Measure with [SOP 0001](../sops/0001-measure-worker-memory.md).
- **Throughput.** Batch-1 inference reads all 353 MB of weights, so it is bound by memory bandwidth.
  Separate copies compete for it; preloaded workers read the same pages and share the L3 cache.
  The gain depends on the CPU: 303 vs 168 req/s here, 94 vs 76 in the video's sandbox.
- **Sharing can silently break.** A worker that writes to the shared pages copies them.
  [`tests/integration/test_gunicorn.py`](../../tests/integration/test_gunicorn.py) fails above 30%
  of the model private per worker.
  Diagnose with [SOP 0003](../sops/0003-investigate-a-worker-that-copies-the-model.md).
- **CPU only.** GPU memory is not shared by `fork()`, and a forked child cannot use a CUDA context
  its parent created.
  On GPUs one process owns the model and batches requests; NVIDIA MPS shares compute between
  processes, not weights.
