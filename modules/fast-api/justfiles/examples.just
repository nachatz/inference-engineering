# The code from the videos in examples/, kept verbatim: linted, but never formatted or type checked.

set working-directory := ".."

fmt:
    uv run ruff check --fix examples

check:
    uv run ruff check examples
