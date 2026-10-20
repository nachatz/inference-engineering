# Module-wide markdown and yaml (README, configs).

set working-directory := ".."

fmt:
    uv run mdformat .
    yamlfmt

check:
    uv run mdformat --check .
    yamlfmt -lint
