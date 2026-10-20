import math
import os
from pathlib import Path


def cgroup_cpus(cpu_max: Path = Path("/sys/fs/cgroup/cpu.max")) -> int | None:
    """The container's CPU quota from cgroup v2 (`"1600000 100000"` is 16 CPUs), or None when there is none."""
    try:
        quota, period = cpu_max.read_text().split()
    except (OSError, ValueError):
        return None

    if quota == "max":
        return None

    return max(1, math.floor(int(quota) / int(period)))


def available_cpus() -> int:
    """The CPUs this process may use: the container's cgroup quota, else the CPUs it may run on."""
    return cgroup_cpus() or os.process_cpu_count() or 1
