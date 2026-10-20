from app.common.outputs import Output


class LoadResult(Output):
    requests: int
    errors: int
    rps: float
    p50_ms: float
    p99_ms: float

    def __repr__(self) -> str:
        return f"{self.rps:.1f} req/s  p50 {self.p50_ms:.1f} ms  p99 {self.p99_ms:.1f} ms  errors {self.errors}"
