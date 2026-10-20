import os
from pathlib import Path

import pytest

from app.common import cpu


class TestCgroupCpus:
    @pytest.mark.parametrize(
        ("cpu_max", "expected"),
        [("1600000 100000\n", 16), ("150000 100000\n", 1), ("50000 100000\n", 1), ("max 100000\n", None)],
    )
    def test_reads_the_quota(self, tmp_path: Path, cpu_max: str, expected: int | None) -> None:
        (tmp_path / "cpu.max").write_text(cpu_max)
        assert cpu.cgroup_cpus(tmp_path / "cpu.max") == expected

    def test_none_without_a_cgroup(self, tmp_path: Path) -> None:
        assert cpu.cgroup_cpus(tmp_path / "missing") is None

    def test_none_when_unreadable(self, tmp_path: Path) -> None:
        (tmp_path / "cpu.max").write_text("garbage")
        assert cpu.cgroup_cpus(tmp_path / "cpu.max") is None


class TestAvailableCpus:
    def test_prefers_the_cgroup_quota(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpu, "cgroup_cpus", lambda: 3)
        assert cpu.available_cpus() == 3

    def test_then_the_cpus_this_process_may_run_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpu, "cgroup_cpus", lambda: None)
        monkeypatch.setattr(os, "process_cpu_count", lambda: 2)
        assert cpu.available_cpus() == 2

    def test_at_least_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cpu, "cgroup_cpus", lambda: None)
        monkeypatch.setattr(os, "process_cpu_count", lambda: None)
        assert cpu.available_cpus() == 1
