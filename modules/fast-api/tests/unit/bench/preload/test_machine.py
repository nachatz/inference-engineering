from pathlib import Path

import psutil

from bench.common.units import GB
from bench.preload.machine import machine


class TestMachine:
    def test_reads_the_cpu_model_and_the_memory(self, tmp_path: Path) -> None:
        (tmp_path / "cpuinfo").write_text("processor\t: 0\nmodel name\t: AMD Ryzen 7 9800X3D 8-Core Processor\n")
        model = tmp_path / "model.onnx"
        model.write_bytes(b"\0" * 2_000_000)

        described = machine(model, cpuinfo=tmp_path / "cpuinfo")

        assert described.cpu == "AMD Ryzen 7 9800X3D 8-Core Processor"
        assert described.memory_gb == round(psutil.virtual_memory().total / GB, 1)
        assert described.model_mb == 2.0
