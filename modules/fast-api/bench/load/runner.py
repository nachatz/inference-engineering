import csv
import subprocess
import tempfile
from pathlib import Path

from bench.config import LoadTestSettings
from bench.load.outputs import LoadResult


def run(config: LoadTestSettings) -> LoadResult:
    """One `just benchmarks locust` run with `config`, in its own process: locust monkey-patches the one it runs in."""
    root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as out:
        recipe = ["just", "benchmarks", "locust", f"{out}/locust"]
        fields = [
            config.host,
            config.path,
            config.users,
            config.seconds,
            config.rows,
            config.features,
            config.raw,
            config.keep_alive,
        ]

        subprocess.run([*recipe, *map(str, fields)], cwd=root, check=True, capture_output=True)

        with Path(f"{out}/locust_stats.csv").open() as stats:
            (total,) = (row for row in csv.DictReader(stats) if row["Name"] == "Aggregated")

    return LoadResult(
        requests=int(total["Request Count"]),
        errors=int(total["Failure Count"]),
        rps=float(total["Requests/s"]),
        p50_ms=float(total["50%"]),
        p99_ms=float(total["99%"]),
    )
