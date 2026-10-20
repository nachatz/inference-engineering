import os
import sys
from collections import namedtuple

import psutil
import pytest
from pytest_mock import MockerFixture

from app.common.memory import ProcessMemory

# The fields of psutil's Linux pfullmem that ProcessMemory reads, with the ones it ignores.
pfullmem = namedtuple("pfullmem", ["rss", "vms", "shared", "pss", "uss", "swap"])


class TestProcessMemory:
    def test_takes_rss_pss_and_uss_from_psutil(self, mocker: MockerFixture) -> None:
        process = mocker.patch.object(psutil, psutil.Process.__name__)
        process.return_value.memory_full_info.return_value = pfullmem(412, 900, 300, 120, 10, 0)

        assert ProcessMemory.of(7) == ProcessMemory(pid=7, rss=412, pss=120, private=10)
        process.assert_called_once_with(7)

    @pytest.mark.skipif(sys.platform != "linux", reason="psutil reports PSS on Linux only")
    def test_reads_this_process(self) -> None:
        memory = ProcessMemory.of(os.getpid())

        assert 0 < memory.private <= memory.pss <= memory.rss
