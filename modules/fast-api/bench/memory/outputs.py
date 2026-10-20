from app.common.memory import ProcessMemory
from app.common.outputs import Output
from bench.common.units import GB
from bench.common.units import MB


class PoolMemory(Output):
    """A gunicorn master and its workers.

    RSS counts every page a process maps, so a page shared by 4 workers is counted 4 times when you add them up.
    PSS splits each shared page between the processes that share it, so adding up PSS gives the real memory used.
    """

    master: ProcessMemory
    workers: list[ProcessMemory]

    @property
    def processes(self) -> list[ProcessMemory]:
        return [self.master, *self.workers]

    @property
    def rss(self) -> int:
        return sum(p.rss for p in self.processes)

    @property
    def pss(self) -> int:
        return sum(p.pss for p in self.processes)

    @property
    def private(self) -> int:
        return sum(p.private for p in self.processes)

    def __repr__(self) -> str:
        def row(pid: object, role: str, memory: ProcessMemory | PoolMemory) -> str:
            return f"{pid:>8}  {role:<7}{memory.rss / MB:>10.1f}{memory.pss / MB:>10.1f}{memory.private / MB:>12.1f}"

        return "\n".join(
            [
                f"gunicorn master {self.master.pid} and {len(self.workers)} workers",
                f"{'pid':>8}  {'role':<7}{'RSS MB':>10}{'PSS MB':>10}{'private MB':>12}",
                row(self.master.pid, "master", self.master),
                *(row(worker.pid, "worker", worker) for worker in self.workers),
                row("total", "", self),
                f"PSS total {self.pss / GB:.2f} GB, RSS total {self.rss / GB:.2f} GB",
            ]
        )
