import pytest
from pydantic import ValidationError

from app.common.outputs import Output
from app.core.outputs import NoCpuLimit
from app.core.outputs import WorkerReady


class TestOutput:
    def test_logs_as_its_repr(self) -> None:
        ready = WorkerReady(pid=42)
        assert str(ready) == repr(ready) == "worker 42 ready"

    def test_logs_its_class_name_as_its_event(self) -> None:
        assert WorkerReady(pid=42).event == "worker_ready"

    def test_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            WorkerReady(pid=42).pid = 7

    def test_needs_a_repr(self) -> None:
        class Silent(Output): ...

        with pytest.raises(TypeError, match="abstract"):
            Silent()  # type: ignore[abstract]  # the error is the point


class TestNoCpuLimit:
    def test_says_how_many_workers_and_how_to_bound_them(self) -> None:
        line = repr(NoCpuLimit(workers=64))

        assert line.startswith("no CPU limit and no GUNICORN_WORKERS: starting 64 workers")
        assert "Set a CPU limit on the pod, or GUNICORN_WORKERS" in line
