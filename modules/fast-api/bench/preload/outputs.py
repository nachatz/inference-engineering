from pathlib import Path

from app.common.outputs import Output
from bench.preload.models import Metrics


class Run(Output):
    workers: int
    preload: bool
    repeat: int
    metrics: Metrics

    def __repr__(self) -> str:
        m = self.metrics
        preload = "on" if self.preload else "off"

        return (
            f"{self.workers} workers, preload {preload}, run {self.repeat + 1}: "
            f"{m.rps:.1f} req/s, p99 {m.p99_ms:.0f} ms, PSS {m.pss_total_mb:.0f} MB, "
            f"{m.private_per_worker_mb:.1f} MB private per worker"
        )


class ResultsWritten(Output):
    json_path: Path
    markdown_path: Path

    def __repr__(self) -> str:
        return f"wrote {self.json_path} and {self.markdown_path}"
