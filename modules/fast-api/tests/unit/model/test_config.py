import pytest

from model.config import ExportSettings


class TestExportSettings:
    def test_reads_the_export_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXPORT_SEED", "7")
        monkeypatch.setenv("EXPORT_TOLERANCE", "1e-3")

        assert ExportSettings().model_dump() == {"seed": 7, "tolerance": 1e-3, "verify_batch_sizes": (1, 2, 7, 32)}
