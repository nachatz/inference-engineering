from bench.config import PreloadBenchmarkSettings
from bench.preload.models import Machine
from bench.preload.models import Metrics
from bench.preload.models import Summary
from bench.preload.report import Results


class TestResults:
    def test_renders_one_row_per_configuration(self, metrics: Metrics) -> None:
        machine = Machine(
            platform="Linux",
            cpu="AMD Ryzen 7 9800X3D",
            cpus=8,
            memory_gb=16.3,
            python="3.13.16",
            onnxruntime="1.31.0",
            model="model/model.onnx",
            model_mb=352.6,
        )
        summaries = [
            Summary(workers=4, preload=False, repeats=3, median=metrics),
            Summary(workers=4, preload=True, repeats=3, median=metrics),
        ]

        results = Results(machine=machine, config=PreloadBenchmarkSettings(), summaries=summaries, runs=[])
        lines = results.markdown().splitlines()

        assert lines[0] == "# CPU preload benchmark"
        assert "- Machine: Linux, AMD Ryzen 7 9800X3D, 8 CPUs, 16.3 GB RAM" in lines
        assert [line.split(" | ")[1] for line in lines[-2:]] == ["no", "yes"]
