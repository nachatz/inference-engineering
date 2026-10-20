# Benchmarks and memory tools in bench/, and their tests. Linux only (they read /proc).

set working-directory := ".."

paths := "bench tests/unit/bench"

# One headless locust run of bench/load/locustfile.py, writing its stats to `csv`. Called by bench/load/runner.py: each
# `$` parameter is exported as the LoadTestSettings field it names, which is how the locustfile reads them.
locust csv $LOAD_TEST_HOST $LOAD_TEST_PATH $LOAD_TEST_USERS $LOAD_TEST_SECONDS $LOAD_TEST_ROWS $LOAD_TEST_FEATURES $LOAD_TEST_RAW $LOAD_TEST_KEEP_ALIVE:
    uv run locust --locustfile bench/load/locustfile.py --headless --only-summary --loglevel WARNING \
        --exit-code-on-error 0 --csv "{{ csv }}" --host "$LOAD_TEST_HOST" \
        --users "$LOAD_TEST_USERS" --spawn-rate "$LOAD_TEST_USERS" --run-time "${LOAD_TEST_SECONDS}s"

fmt:
    uv run ruff format {{ paths }}
    uv run ruff check --fix {{ paths }}

check:
    uv run ruff format --check {{ paths }}
    uv run ruff check {{ paths }}

mypy:
    uv run mypy {{ paths }}

test:
    uv run pytest tests/unit/bench
