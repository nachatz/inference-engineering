from pathlib import Path

from bench.preload.models import Metrics
from bench.preload.outputs import ResultsWritten
from bench.preload.outputs import Run


class TestRun:
    def test_reports_the_headline_numbers(self, metrics: Metrics) -> None:
        run = Run(workers=4, preload=True, repeat=0, metrics=metrics)

        assert repr(run) == (
            "4 workers, preload on, run 1: 302.8 req/s, p99 140 ms, PSS 502 MB, 14.4 MB private per worker"
        )


class TestResultsWritten:
    def test_names_both_files(self) -> None:
        written = ResultsWritten(json_path=Path("a.json"), markdown_path=Path("a.md"))

        assert repr(written) == "wrote a.json and a.md"
