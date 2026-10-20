from pathlib import Path

from model.outputs import ExportVerified
from model.outputs import ModelExported


class TestModelExported:
    def test_reports_the_size_in_decimal_megabytes(self) -> None:
        assert repr(ModelExported(path=Path("model.onnx"), size_bytes=352_601_452)) == "wrote model.onnx (353 MB)"


class TestExportVerified:
    def test_passes_within_the_tolerance(self) -> None:
        verified = ExportVerified(error=1.12e-8, worst="normal, 64 rows", cases=20, tolerance=1e-4)
        assert verified.passed
        assert repr(verified) == "pytorch vs onnx over 20 batches: worst 1.12e-08 (normal, 64 rows), tolerance 1e-04"

    def test_fails_above_the_tolerance(self) -> None:
        verified = ExportVerified(error=1e-3, worst="zeros, 1 rows", cases=4, tolerance=1e-4)
        assert not verified.passed
        assert (
            repr(verified)
            == "pytorch vs onnx over 4 batches: worst 1.00e-03 (zeros, 1 rows), above the tolerance 1e-04"
        )
