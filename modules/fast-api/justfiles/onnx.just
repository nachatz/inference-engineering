# The PyTorch model in model/, its ONNX export, and their tests. Build time only: needs the `export` group (torch).

set working-directory := ".."

paths := "model tests/unit/model"

fmt:
    uv run ruff format {{ paths }}
    uv run ruff check --fix {{ paths }}

check:
    uv run ruff format --check {{ paths }}
    uv run ruff check {{ paths }}

mypy:
    uv run mypy {{ paths }}

test:
    uv run pytest tests/unit/model
