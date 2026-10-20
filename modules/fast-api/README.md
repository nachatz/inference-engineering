# fast-api

A production-grade CPU model server: a PyTorch model exported to ONNX at build time, served with ONNX Runtime by
[FastAPI](https://fastapi.tiangolo.com/) in [gunicorn](https://gunicorn.org/)-managed [uvicorn](https://www.uvicorn.org/)
workers that share one copy of the model.

&#160;

______________________________________________________________________

## Quick start

```bash
uv sync                    # serving deps, plus PyTorch (CPU) for the export and the dev tools
just model verify          # export model/model.onnx (353 MB) and check it against PyTorch
just serve                 # gunicorn, one worker per core, preload on (Linux/macOS)
just load-test                     # headless locust: LOAD_TEST_USERS, LOAD_TEST_SECONDS, ... in bench/config.py
just memory                        # RSS vs PSS vs private of the running gunicorn, per process
```

```bash
curl localhost:8000/readyz
curl localhost:8000/debug/allocator   # also /debug/flags, /debug/memory, /debug/runtime, /debug/model
curl -X POST localhost:8000/v1/predict -H 'content-type: application/json' \
  -d "{\"inputs\": [$(python3 -c 'print([0.0] * 1024)')]}"
```

Every command is a [just](https://just.systems/) recipe, see the [justfile](justfile). gunicorn, `bench/` and the slow tests need
Linux (`fork()`, `/proc`); on Windows run them in Docker or WSL. `just dev` runs a single uvicorn process with
auto-reload on any OS.

Settings are environment variables, one prefix per concern, see [`app/core/config.py`](app/core/config.py): `GUNICORN_*`
(`GUNICORN_WORKERS`, `GUNICORN_PRELOAD=0`, `GUNICORN_MAX_REQUESTS`, ...), `MODEL_*` (`MODEL_PATH`, `MODEL_NAME`,
`MODEL_VERSION`, `MODEL_SHA256`, `MODEL_ADAPTER`, `MODEL_MAX_BATCH`, `MODEL_MAX_QUEUE`, `MODEL_MAX_QUEUE_WAIT_S`,
`MODEL_INFERENCE_STUCK_S`, `MODEL_MAX_BODY_BYTES`, `MODEL_MAX_ADMITTED`, `MODEL_MAX_BODY_READ_S`, ...), `OTEL_*` (`OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
`OTEL_SAMPLE_RATIO`), `SENTRY_*` (`SENTRY_ENABLED`, `SENTRY_DSN`, `SENTRY_RELEASE`), plus `APP_TITLE`, `ENVIRONMENT`, `LOG_LEVEL` and `LOG_FORMAT` (`json` or `text`).
`/debug/*` and the OpenAPI docs are always on. The build and benchmark tools
take none of their own flags either: `EXPORT_*` in [`model/config.py`](model/config.py) and `LOAD_TEST_*`, `MEM_*`,
`BENCH_*` in [`bench/config.py`](bench/config.py). The export writes to `MODEL_PATH`, the file the server loads. Tested with Python 3.13. Run
[mise](https://mise.jdx.dev/) `install` to get the pinned Python, uv, just and yamlfmt.

&#160;

______________________________________________________________________

## Layout

```
app/
├── main.py              the ASGI entry point gunicorn loads: preload(), create_app(), gc.freeze()
├── apps/                the FastAPI apps
│   ├── base.py          BaseApp: the shared routers, exception handlers, /metrics, request IDs, tracing
│   ├── inference.py     InferenceApp(BaseApp): /v1/predict, its handlers and middleware · create_app(served)
│   └── lifespan.py      per-worker state only: the inference slot and its watchdog, the tracer provider
├── core/                what every part uses, and what runs before the fork
│   ├── preload.py       runs in the gunicorn master: logging, then load and warm up the model
│   ├── config.py        settings, one BaseSettings per env prefix
│   ├── dependencies.py  the app's settings, for Depends
│   ├── logger.py        one JSON (or text) format for the app's, gunicorn's and uvicorn's loggers
│   └── outputs.py       WorkerReady · ModelLoaded · InferenceStuck: what the app logs
├── inference/           the model and the predict feature
│   ├── runtime/         the model: what is loaded before the fork and shared by every worker
│   │   ├── tensor.py    TensorSpec: an input or output; array() converts and checks a request's values
│   │   ├── onnx.py      OnnxModel: the session, its specs, run (no NaN or infinity out), warm-up
│   │   ├── served.py    ServedModel: load, checksum, check against the adapter and warm up
│   │   └── info.py      ModelInfo: which artifact this is
│   ├── adapters/        what a service writes: request and response models, to_feed, from_outputs, check
│   │                    each registers itself by name (MODEL_ADAPTER): classifier · scorer
│   ├── capacity/        what a worker takes on, and how it refuses the rest
│   │   ├── admission.py AdmissionMiddleware: at most MODEL_MAX_ADMITTED requests, body due in time
│   │   └── slot.py      InferenceSlot: one inference behind a bounded queue · the stuck watchdog
│   ├── handlers/        PredictHandler: check, queue, inference off the event loop, the inference span
│   ├── dependencies.py  the served model (app state) and this worker's slot (lifespan state), for Depends
│   └── exceptions.py    NotReady · Overloaded · InputMismatch · NonFiniteOutput · IncompatibleModel · ...
├── health/              HealthHandler (readiness) · models
├── debug/               DebugHandler: flags, allocator, memory, threads, model, read from /proc · models
├── exceptions/          ExceptionHandler[E] and one handler per exception · ErrorResponse
├── observability/       opentelemetry.py (tracing, per worker) · sentry.py (errors, started in preload) · metrics.py · prometheus.py
├── routers/             health.py (/healthz, /readyz) · debug.py (/debug/*) · v1/predict.py; /metrics is prometheus_client's app
└── common/              cpu.py (workers from the CPU quota) · memory.py · enums/
gunicorn.conf.py         workers, preload_app, reuse_port, timeouts, logging, Prometheus multiprocess dir
cmd/                     every entry point, one script each: export_model · verify_model · load_test · memory · preload_benchmark
model/                   build time only (PyTorch): model.py · export.py · verify.py · config.py
bench/                   one package per tool; their entry points are in cmd/
├── common/              proc.py (psutil: children, CPU time, listening sockets) · server.py (gunicorn as a subprocess)
├── load/                locustfile.py · runner.py: one headless locust run → LoadResult
├── memory/              pool.py: a gunicorn master and its workers → PoolMemory
├── preload/             benchmark.py · machine.py · models.py · report.py → bench/results/cpu-preload.{json,md}
├── config.py            LOAD_TEST_*, MEM_*, BENCH_*
└── results/             what `just bench` last measured
examples/                the code from the videos, verbatim
Dockerfile               export stage (PyTorch) → runtime image (ONNX Runtime only, non-root, jemalloc)
tests/
├── factories/           polyfactory factories for the request models
├── fixtures/            shared fixtures (tiny ONNX model, clients, in-memory spans), re-exported by conftest.py
├── unit/                mirrors app/, model/ and bench/
└── integration/         the API, tracing, concurrency, and test_gunicorn.py: real servers (slow, Linux)
typings/                 type stubs for onnxruntime
```

&#160;

______________________________________________________________________

## Development

```bash
just fmt        # format and lint with ruff, mdformat and yamlfmt
just validate   # format check + mypy --strict + pytest (slow tests included on Linux) with 100% coverage
just --list --list-submodules   # every recipe, including the per-part ones
```

Each part of the codebase has its own justfile in [`justfiles/`](justfiles/) (`app`, `onnx` for `model/`,
`benchmarks` for `bench/`, `examples`, `repo`, `image`), and the root `justfile` delegates to them. Run one part on its own with
e.g. `just app test` or `just onnx mypy`. `pytest -m "not slow"` skips the tests that start gunicorn.

```bash
docker build -t fast-api .
docker run --rm -p 8000:8000 fast-api gunicorn -c gunicorn.conf.py   # no CMD: the deploy files set it
just image::smoke image::scan           # what CI runs: run it locked down, then scan it
```

&#160;

______________________________________________________________________

## Design decisions

| decision | why | where |
| --- | --- | --- |
| Serve ONNX Runtime; PyTorch only at build time | no torch in the image; the server reads inputs from the file | [`model/export.py`](model/export.py), [`app/inference/runtime/onnx.py`](app/inference/runtime/onnx.py) |
| Load the model in the gunicorn master, then fork | one shared copy per pod instead of one per worker | [`app/core/preload.py`](app/core/preload.py), [`gunicorn.conf.py`](gunicorn.conf.py) |
| One core, one thread, one inference per worker | no thread pools to break at fork; health checks never wait on inference | [`app/inference/capacity/slot.py`](app/inference/capacity/slot.py) |
| One listening socket per worker (`reuse_port`) | with a shared socket, the worker with a free event loop takes every connection | [`gunicorn.conf.py`](gunicorn.conf.py) |
| Per-worker state only in the lifespan | threads don't survive fork; the model never goes there | [`app/apps/lifespan.py`](app/apps/lifespan.py) |
| Admission before the body, bounded queue, deadlines for the body and the queue, body limit, stuck-worker exit | overload becomes fast 503s and 408s, not timeouts or memory growth: a worker holds at most `MODEL_MAX_ADMITTED` requests (TestOverload measures it) | [`app/inference/capacity/admission.py`](app/inference/capacity/admission.py), [`app/inference/capacity/slot.py`](app/inference/capacity/slot.py), [`app/apps/inference.py`](app/apps/inference.py) |
| One JSON log line per event, with request and trace IDs | the access log, a traceback and a span share an ID | [`app/core/logger.py`](app/core/logger.py) |
| A service writes an adapter, checked at startup | the runtime stays generic across CPU ONNX models; a model the adapter can't serve stops gunicorn before it forks | [`app/inference/adapters/`](app/inference/adapters/) |
| No NaN or infinity in, none out | a NaN logit still has an argmax: it would be a 200 with a label | [`app/inference/runtime/tensor.py`](app/inference/runtime/tensor.py), [`app/inference/runtime/onnx.py`](app/inference/runtime/onnx.py) |
