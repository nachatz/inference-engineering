import gc
import os
from collections import namedtuple

import psutil
import pytest
from pytest_mock import MockerFixture
from pytest_mock import MockType

from app.common.enums.allocator import Allocator
from app.common.memory import ProcessMemory
from app.core.config import AllocatorSettings
from app.core.config import settings
from app.debug.exceptions import ProcUnavailable
from app.debug.handlers.debug import DebugHandler
from app.debug.models.debug import TensorResponse
from app.inference.runtime.served import ServedModel

pmmap = namedtuple("pmmap", ["path", "rss"])


@pytest.fixture
def process(mocker: MockerFixture) -> MockType:
    """This process as psutil sees it on Linux with jemalloc preloaded."""
    process: MockType = mocker.create_autospec(psutil.Process, instance=True)
    process.pid = os.getpid()
    process.num_threads.return_value = 3
    process.memory_maps.return_value = [
        pmmap("/usr/local/bin/python3.13", 4096),
        pmmap("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2", 8192),
    ]
    return process


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psutil, "LINUX", True)


@pytest.fixture
def not_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(psutil, "LINUX", False)


class TestFlags:
    def test_reports_the_settings_in_effect(self, served: ServedModel, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings.gunicorn, "preload", False)

        flags = DebugHandler(served, settings).flags()

        assert not flags.preload
        assert (flags.intra_op_threads, flags.max_queue) == (settings.model.intra_op_threads, settings.model.max_queue)


class TestAllocator:
    @pytest.mark.usefixtures("linux")
    def test_finds_jemalloc_in_the_process_maps(
        self, served: ServedModel, process: MockType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        preloaded = AllocatorSettings(
            ld_preload="/usr/local/lib/libjemalloc.so.2", malloc_conf="background_thread:true"
        )
        monkeypatch.setattr(settings, "allocator", preloaded)

        allocator = DebugHandler(served, settings, process).allocator()

        assert allocator.allocator is Allocator.JEMALLOC
        assert allocator.malloc_conf == "background_thread:true"

    @pytest.mark.usefixtures("linux")
    def test_falls_back_to_glibc(self, served: ServedModel, process: MockType) -> None:
        process.memory_maps.return_value = [pmmap("/usr/lib/x86_64-linux-gnu/libc.so.6", 4096)]

        assert DebugHandler(served, settings, process).allocator().allocator is Allocator.GLIBC

    @pytest.mark.usefixtures("not_linux")
    def test_unknown_off_linux(self, served: ServedModel, process: MockType) -> None:
        assert DebugHandler(served, settings, process).allocator().allocator is Allocator.UNKNOWN


class TestMemory:
    @pytest.mark.usefixtures("linux")
    def test_reads_this_worker(self, served: ServedModel, process: MockType, mocker: MockerFixture) -> None:
        worker = ProcessMemory(pid=os.getpid(), rss=400, pss=100, private=10)
        of = mocker.patch.object(ProcessMemory, ProcessMemory.of.__name__, return_value=worker)

        assert DebugHandler(served, settings, process).memory() == worker
        of.assert_called_once_with(os.getpid())

    @pytest.mark.usefixtures("not_linux")
    def test_needs_linux(self, served: ServedModel, process: MockType) -> None:
        with pytest.raises(ProcUnavailable):
            DebugHandler(served, settings, process).memory()


class TestRuntime:
    def test_counts_threads_and_frozen_objects(self, served: ServedModel, process: MockType) -> None:
        runtime = DebugHandler(served, settings, process).runtime()

        assert (runtime.pid, runtime.native_threads) == (os.getpid(), 3)
        assert runtime.gc_frozen_objects == gc.get_freeze_count()
        assert runtime.versions.onnxruntime

    def test_reads_this_process_by_default(self, served: ServedModel) -> None:
        assert DebugHandler(served, settings).runtime().native_threads >= 1


class TestModel:
    def test_describes_the_served_model(self, served: ServedModel) -> None:
        model = DebugHandler(served, settings).model()

        assert (model.path, model.info) == (served.path, served.info)
        assert model.inputs == [TensorResponse(name="x", dtype="float32", shape=["batch", 8])]
        assert model.outputs == [TensorResponse(name="logits", dtype="float32", shape=["batch", 3])]
        assert model.adapter == "ClassifierAdapter"
