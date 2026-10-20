from app.common.memory import ProcessMemory
from bench.common.proc import children
from bench.memory.outputs import PoolMemory


def measure(master: int) -> PoolMemory:
    workers = [ProcessMemory.of(pid) for pid in children(master)]
    return PoolMemory(master=ProcessMemory.of(master), workers=workers)
