# The entry points in cmd/: thin scripts over app/, model/ and bench/. They aren't a package, since the standard
# library has a `cmd` module; PYTHONPATH puts the module root on the path so they can import the rest.

set working-directory := ".."

paths := "cmd"

# Export the PyTorch model to MODEL_PATH. Needs the `export` group (torch).
export-model:
    PYTHONPATH=. uv run python cmd/export_model.py

# Check the ONNX file at MODEL_PATH matches PyTorch; exits 1 above EXPORT_TOLERANCE.
verify-model:
    PYTHONPATH=. uv run python cmd/verify_model.py

# Headless locust against a running server. LOAD_TEST_*: see bench/config.py.
load-test:
    PYTHONPATH=. uv run python cmd/load_test.py

# RSS, PSS and private memory of the running gunicorn (MEM_PID to pick one). Linux only.
memory:
    PYTHONPATH=. uv run python cmd/memory.py

# With vs without preload, 1 to 8 workers: writes bench/results/cpu-preload.{json,md}. BENCH_*: see bench/config.py.
preload-benchmark:
    PYTHONPATH=. uv run python cmd/preload_benchmark.py

fmt:
    uv run ruff format {{ paths }}
    uv run ruff check --fix {{ paths }}

check:
    uv run ruff format --check {{ paths }}
    uv run ruff check {{ paths }}

mypy:
    PYTHONPATH=. uv run mypy {{ paths }}
