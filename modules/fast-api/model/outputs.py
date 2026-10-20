from pathlib import Path

from app.common.outputs import Output


class ModelExported(Output):
    path: Path
    size_bytes: int

    def __repr__(self) -> str:
        return f"wrote {self.path} ({self.size_bytes / 1e6:.0f} MB)"


class ExportVerified(Output):
    error: float
    worst: str
    cases: int
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.error <= self.tolerance

    def __repr__(self) -> str:
        verdict = f"tolerance {self.tolerance:.0e}" if self.passed else f"above the tolerance {self.tolerance:.0e}"
        return f"pytorch vs onnx over {self.cases} batches: worst {self.error:.2e} ({self.worst}), {verdict}"
