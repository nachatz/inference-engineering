import os
import platform
from pathlib import Path

import onnxruntime as ort
import psutil

from bench.common.units import GB
from bench.common.units import MB
from bench.preload.models import Machine


def machine(model: Path, cpuinfo: Path = Path("/proc/cpuinfo")) -> Machine:
    """The machine the benchmark ran on, so its numbers are never quoted without one. Linux only: the CPU's model
    name comes from /proc/cpuinfo, since neither the standard library nor psutil reports it."""
    lines = cpuinfo.read_text().splitlines()
    cpu = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")), "unknown")

    return Machine(
        platform=platform.platform(),
        cpu=cpu,
        cpus=os.cpu_count() or 1,
        memory_gb=round(psutil.virtual_memory().total / GB, 1),
        python=platform.python_version(),
        onnxruntime=ort.__version__,
        model=str(model),
        model_mb=round(model.stat().st_size / MB, 1),
    )
