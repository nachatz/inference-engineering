# 0005 · Create per-worker state after the fork

| status | date | owner | supersedes |
| --- | --- | --- | --- |
| accepted | 2026-10-18 | @nachatz | none |

## Context

With preload ([ADR 0002](0002-preload-the-model-and-fork-workers.md)) everything created at import
is created once, in the master.
Some state must not be: threads do not survive `fork()`, and some objects are meant to be one per
worker.

## Decision

The FastAPI lifespan ([`app/apps/lifespan.py`](../../app/apps/lifespan.py)) runs in every worker after the
fork.
It is an `AsyncExitStack` of single-purpose managers, and holds only per-worker state.

| state | why per worker | where |
| --- | --- | --- |
| the inference slot and its stuck-inference watchdog | one inference at a time *per worker* | [`app/inference/capacity/slot.py`](../../app/inference/capacity/slot.py) |
| the garbage collector, enabled again | the master built the shared heap with it off | [`app/main.py`](../../app/main.py) |
| the OpenTelemetry tracer provider and exporter | the batch exporter runs a thread | [`app/observability/opentelemetry.py`](../../app/observability/opentelemetry.py) |
| Prometheus metric files | each worker writes its own; `/metrics` merges them | [`app/observability/prometheus.py`](../../app/observability/prometheus.py) |

Tracing exports over OTLP/HTTP rather than gRPC, whose runtime has its own fork pitfalls.
The app is instrumented when it is built (in the master); that only wraps it in middleware.

## Alternatives

| option | why not |
| --- | --- |
| create everything at import | the exporter thread exists only in the master; workers would export nothing |
| gunicorn's `post_fork` hook | outside the app; the lifespan also runs under `just dev` and in tests |

## Consequences

- The model never goes in the lifespan; [SOP
  0003](../sops/0003-investigate-a-worker-that-copies-the-model.md) checks for it.
- `/readyz` returns 503 until a worker's lifespan has run, so a worker that has not started gets no
  traffic.
- Prometheus multiprocess mode needs `PROMETHEUS_MULTIPROC_DIR` in gunicorn's environment; the
  Dockerfile and `just serve` set it.
