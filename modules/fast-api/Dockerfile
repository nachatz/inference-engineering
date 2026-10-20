# syntax=docker/dockerfile:1

# Base images are pinned by digest, so a rebuild of the same commit gets the same bytes; Renovate bumps tag and digest.
FROM ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 AS uv

FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /srv
COPY pyproject.toml uv.lock ./

FROM base AS export
RUN uv sync --frozen --no-default-groups --group export

COPY app/__init__.py app/
COPY app/core/__init__.py app/core/config.py app/core/logger.py app/core/
COPY app/common/ app/common/
COPY model/ model/
COPY cmd/export_model.py cmd/verify_model.py cmd/
RUN PYTHONPATH=. .venv/bin/python cmd/export_model.py && PYTHONPATH=. .venv/bin/python cmd/verify_model.py

FROM base AS deps
RUN uv sync --frozen --no-default-groups

FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS runtime
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libjemalloc2 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2" /usr/local/lib/libjemalloc.so.2 \
    && pip uninstall --yes pip \
    && useradd --system --uid 10001 --no-create-home app

ENV PATH=/srv/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/srv/model/model.onnx \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 \
    MALLOC_CONF=background_thread:true,metadata_thp:auto,dirty_decay_ms:90000,muzzy_decay_ms:90000

WORKDIR /srv
COPY --from=deps /srv/.venv .venv
COPY --from=export /srv/model/model.onnx model/model.onnx
COPY gunicorn.conf.py .
COPY app/ app/

USER app
EXPOSE 8000

# No CMD: a deployment runs `gunicorn -c gunicorn.conf.py`. As the container's command, gunicorn is PID 1, so
# SIGTERM reaches it directly and it drains the workers.