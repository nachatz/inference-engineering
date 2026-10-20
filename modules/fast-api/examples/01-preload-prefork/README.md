# 01 · Preload and prefork: the minimal version

The code from the video, as shown on screen: an ONNX model loaded once in the gunicorn master and shared by four
forked workers. It's kept short on purpose and left exactly as in the video, so it skips the import sorting and
formatting the rest of the repo uses.

```bash
just model                                   # from modules/fast-api: writes model/model.onnx
cp model/model.onnx examples/01-preload-prefork/
(cd examples/01-preload-prefork && uv run gunicorn -c gunicorn.conf.py app:app) &
LOAD_TEST_PATH=/predict LOAD_TEST_RAW=true just load-test
just memory
```

The production version of the same idea is [`app/`](../../app/) with [`gunicorn.conf.py`](../../gunicorn.conf.py).
It adds input validation, one inference at a time per worker without blocking the event loop, health and readiness
checks, exception handlers, Prometheus metrics, OpenTelemetry tracing, and a worker count that follows the container's
CPU quota. The numbers are in [`bench/results/cpu-preload.md`](../../bench/results/cpu-preload.md).
