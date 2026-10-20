from app.common.memory import ProcessMemory
from bench.memory.outputs import PoolMemory


class TestPoolMemory:
    def test_totals_every_process_in_decimal_units(self) -> None:
        master = ProcessMemory(pid=1, rss=400_000_000, pss=100_000_000, private=1_000_000)
        worker = ProcessMemory(pid=2, rss=400_000_000, pss=100_000_000, private=10_000_000)

        pool = PoolMemory(master=master, workers=[worker, worker])

        assert (pool.rss, pool.pss, pool.private) == (1_200_000_000, 300_000_000, 21_000_000)
        assert repr(pool).splitlines()[0] == "gunicorn master 1 and 2 workers"
        assert repr(pool).splitlines()[-1] == "PSS total 0.30 GB, RSS total 1.20 GB"
