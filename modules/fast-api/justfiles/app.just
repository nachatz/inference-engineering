# The FastAPI app in app/, its gunicorn config, and their tests.

set working-directory := ".."

paths := "app gunicorn.conf.py typings tests/conftest.py tests/factories tests/fixtures tests/utils tests/unit/app tests/integration"

# One process with auto-reload, logging text for a person rather than JSON for a pipeline.
dev:
    LOG_FORMAT="${LOG_FORMAT:-text}" uv run uvicorn app.main:app --reload

# prometheus_client reads PROMETHEUS_MULTIPROC_DIR when it's imported, and OpenMP, OpenBLAS and MKL size their thread
# pools when NumPy loads them, in the master: all set before gunicorn starts. `@` keeps just from echoing the command,
# so everything on stdout is a JSON log line.
serve:
    @PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/fast-api-prometheus}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" exec uv run gunicorn -c gunicorn.conf.py

fmt:
    uv run ruff format {{ paths }}
    uv run ruff check --fix {{ paths }}

check:
    uv run ruff format --check {{ paths }}
    uv run ruff check {{ paths }}

mypy:
    uv run mypy {{ paths }}

test:
    uv run pytest tests/unit/app tests/integration
