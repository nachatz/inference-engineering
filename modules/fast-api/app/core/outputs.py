from app.common.outputs import Output
from app.inference.runtime.info import ModelInfo


class WorkerReady(Output):
    pid: int

    def __repr__(self) -> str:
        return f"worker {self.pid} ready"


class ModelLoaded(Output):
    model: ModelInfo
    seconds: float

    def __repr__(self) -> str:
        m = self.model
        return (
            f"loaded {m.name} {m.version} ({m.size_bytes / 1e6:.0f} MB, sha256 {m.sha256[:12]}) in {self.seconds:.2f} s"
        )


class NoCpuLimit(Output):
    """No GUNICORN_WORKERS and no cgroup CPU quota: gunicorn sizes itself to every CPU it can run on, which on a
    Kubernetes node with CPU requests but no limit is the whole node's."""

    workers: int

    def __repr__(self) -> str:
        return (
            f"no CPU limit and no GUNICORN_WORKERS: starting {self.workers} workers, one per CPU this process can "
            "run on. Set a CPU limit on the pod, or GUNICORN_WORKERS"
        )


class InferenceStuck(Output):
    pid: int
    seconds: float

    def __repr__(self) -> str:
        return f"worker {self.pid}'s inference has run for {self.seconds:.1f} s: exiting so gunicorn replaces it"
