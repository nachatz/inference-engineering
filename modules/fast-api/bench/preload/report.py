from pydantic import BaseModel

from bench.config import PreloadBenchmarkSettings
from bench.preload.models import Machine
from bench.preload.models import Summary
from bench.preload.outputs import Run


class Results(BaseModel):
    """Everything one `just bench` measured: written as bench/results/cpu-preload.json, rendered as the .md table."""

    machine: Machine
    config: PreloadBenchmarkSettings
    summaries: list[Summary]
    runs: list[Run]

    def markdown(self) -> str:
        m, c = self.machine, self.config
        connections = "one kept-alive connection per user" if c.keep_alive else "a new connection per request"

        lines = [
            "# CPU preload benchmark",
            "",
            (
                f"Written by `just bench`. Medians of {c.repeats} runs per row. Each run is a {c.seconds} s headless "
                f"locust run: {c.users} users with no wait time, {c.rows} row(s) per request, "
                f"{connections}. "
                "The load generator runs on the same machine."
            ),
            "",
            f"- Machine: {m.platform}, {m.cpu}, {m.cpus} CPUs, {m.memory_gb} GB RAM",
            f"- Python {m.python}, ONNX Runtime {m.onnxruntime}",
            f"- Model: `{m.model}`, {m.model_mb} MB",
            "",
            (
                "| workers | preload | req/s | p50 ms | p99 ms | errors | PSS total MB | RSS total MB "
                "| private per worker MB | first response s | all workers ready s |"
            ),
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]

        for s in self.summaries:
            r = s.median
            preload = "yes" if s.preload else "no"
            lines.append(
                f"| {s.workers} | {preload} | {r.rps:.1f} | {r.p50_ms:.1f} | {r.p99_ms:.1f} "
                f"| {r.errors} | {r.pss_total_mb:.0f} | {r.rss_total_mb:.0f} | {r.private_per_worker_mb:.1f} "
                f"| {r.first_response_s:.2f} | {r.all_workers_ready_s:.2f} |"
            )

        return "\n".join(lines) + "\n"
