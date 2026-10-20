# Inference Engineering

The companion code for the **Inference Engineering** series on YouTube
([@nikachatz](https://www.youtube.com/@nikachatz)): how to serve models in production, one technique per video, each
with runnable code, a benchmark you can reproduce, and a write-up.

The code lives in [`modules/fast-api/`](modules/fast-api/): a production-grade CPU model server (FastAPI, gunicorn and
ONNX Runtime) that each video builds on. Next to it, [`examples/`](modules/fast-api/examples/) holds the minimal code
from each video, exactly as shown on screen.

&#160;

______________________________________________________________________

## Topics

| # | video | minimal code |
| ---: | --- | --- |
| 01 | One model, many workers: preload and prefork on CPU | [examples/01-preload-prefork](modules/fast-api/examples/01-preload-prefork/) |

&#160;

______________________________________________________________________

## Results

01 · preload and prefork, 4 gunicorn workers serving a 353 MB model. Measured with `just bench` on Docker Desktop's Linux
VM (8 vCPUs, AMD Ryzen 7 9800X3D, ONNX Runtime 1.31), medians of 3 runs. The full table, for 1 to 8 workers, is in
[`bench/results/cpu-preload.md`](modules/fast-api/bench/results/cpu-preload.md).

| 4 workers | without preload | with preload |
| --- | ---: | ---: |
| PSS, all processes | 1684 MB | 502 MB |
| RSS, all processes (double-counts shared pages) | 1861 MB | 2190 MB |
| private memory per worker | 405.7 MB | 14.4 MB |
| throughput | 168.0 req/s | 302.8 req/s |
| p99 latency (locust, 2 significant figures) | 250 ms | 140 ms |

&#160;

______________________________________________________________________

## Quick start

```bash
cd modules/fast-api
mise install               # optional: the pinned Python, uv, just and yamlfmt
uv sync
just model verify          # export model/model.onnx with PyTorch and check it against PyTorch
just serve                 # gunicorn with preload_app, one worker per core (Linux/macOS)
just load-test
just memory
just bench                 # with vs without preload, 1-8 workers → bench/results/cpu-preload.{json,md}
just cov                   # the whole suite, 100% coverage
```

gunicorn, the benchmarks and the slow tests need Linux (`fork()`, `/proc`). On Windows run them in Docker or WSL.

&#160;

______________________________________________________________________

## Production checklist

| concern | where |
| --- | --- |
| Export the PyTorch model to ONNX once, at build time; the server never imports PyTorch | [`model/export.py`](modules/fast-api/model/export.py), [`Dockerfile`](modules/fast-api/Dockerfile) |
| Check the export matches PyTorch | [`model/verify.py`](modules/fast-api/model/verify.py) |
| One model copy per pod: `preload_app`, then fork | [`gunicorn.conf.py`](modules/fast-api/gunicorn.conf.py) |
| Load and warm up the model at import, in the master; per-worker state only in the lifespan | [`app/core/preload.py`](modules/fast-api/app/core/preload.py), [`app/apps/lifespan.py`](modules/fast-api/app/apps/lifespan.py) |
| `gc.freeze()` once the app is built | [`app/main.py`](modules/fast-api/app/main.py) |
| One thread per ONNX Runtime session | [`app/inference/runtime/onnx.py`](modules/fast-api/app/inference/runtime/onnx.py) |
| One inference at a time per worker, off the event loop | [`app/inference/handlers/predict.py`](modules/fast-api/app/inference/handlers/predict.py) |
| Spread connections across workers (`reuse_port`) | [`gunicorn.conf.py`](modules/fast-api/gunicorn.conf.py) |
| One worker per core, from the container's CPU quota | [`app/common/cpu.py`](modules/fast-api/app/common/cpu.py) |
| Validate inputs: batch size, feature count | [`app/inference/adapters/classifier.py`](modules/fast-api/app/inference/adapters/classifier.py), [`app/inference/handlers/predict.py`](modules/fast-api/app/inference/handlers/predict.py) |
| One error shape, from registered exception handlers | [`app/exceptions/handlers.py`](modules/fast-api/app/exceptions/handlers.py), [`app/apps/inference.py`](modules/fast-api/app/apps/inference.py) |
| Liveness, readiness and startup probes | [`app/routers/health.py`](modules/fast-api/app/routers/health.py) |
| Prometheus metrics across workers (multiprocess mode) | [`app/observability/prometheus.py`](modules/fast-api/app/observability/prometheus.py), [`gunicorn.conf.py`](modules/fast-api/gunicorn.conf.py) |
| OpenTelemetry tracing, started per worker after the fork; correlation IDs | [`app/observability/opentelemetry.py`](modules/fast-api/app/observability/opentelemetry.py), [`app/apps/inference.py`](modules/fast-api/app/apps/inference.py) |
| Timeouts, keepalive above the load balancer's, graceful shutdown | [`gunicorn.conf.py`](modules/fast-api/gunicorn.conf.py) |
| Small, non-root runtime image without PyTorch, on jemalloc | [`Dockerfile`](modules/fast-api/Dockerfile) |
| Debug endpoints: flags, allocator, memory, threads, model | [`app/routers/debug.py`](modules/fast-api/app/routers/debug.py), [`app/debug/handlers/debug.py`](modules/fast-api/app/debug/handlers/debug.py) |
| Measure memory as PSS, not RSS | [`bench/memory/`](modules/fast-api/bench/memory/) |
| Benchmark with vs without preload | [`bench/preload/`](modules/fast-api/bench/preload/) |
| A test that fails if the workers stop sharing the model | [`tests/integration/test_gunicorn.py`](modules/fast-api/tests/integration/test_gunicorn.py) |
