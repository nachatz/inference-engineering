import os
import socket
import subprocess
import sys
from collections import namedtuple

import psutil
import pytest
from pytest_mock import MockerFixture

from bench.common.proc import children
from bench.common.proc import cpu_seconds
from bench.common.proc import listening_sockets
from bench.common.proc import oldest_gunicorn

Found = namedtuple("Found", ["pid", "info"])


class TestChildren:
    def test_lists_a_processes_children(self) -> None:
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"]) as child:
            try:
                assert child.pid in children(os.getpid())
            finally:
                child.kill()


class TestOldestGunicorn:
    def test_picks_the_first_gunicorn_started(self, mocker: MockerFixture) -> None:
        found = [
            Found(30, {"name": "gunicorn", "create_time": 200.0}),
            Found(10, {"name": "gunicorn", "create_time": 100.0}),
            Found(20, {"name": "python", "create_time": 50.0}),
        ]
        mocker.patch.object(psutil, "process_iter", return_value=found)

        assert oldest_gunicorn() == 10

    def test_says_how_to_start_one(self, mocker: MockerFixture) -> None:
        mocker.patch.object(psutil, "process_iter", return_value=[])

        with pytest.raises(LookupError, match="just serve"):
            oldest_gunicorn()


class TestCpuSeconds:
    def test_adds_user_and_system_time(self) -> None:
        before = cpu_seconds(os.getpid())
        sum(i * i for i in range(2_000_000))

        assert cpu_seconds(os.getpid()) > before


class TestListeningSockets:
    def test_counts_the_sockets_listening_on_one_port(self) -> None:
        with socket.socket() as listening:
            listening.bind(("127.0.0.1", 0))
            listening.listen()
            port = listening.getsockname()[1]

            assert listening_sockets(port) == 1

        assert listening_sockets(port) == 0
