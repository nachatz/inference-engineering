from pathlib import Path

import pytest
from pydantic import ValidationError

from app.common.enums.environment import Environment
from app.core.config import GunicornSettings
from app.core.config import ModelSettings
from app.core.config import Settings


class TestSettings:
    def test_reads_each_concern_from_its_own_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("GUNICORN_WORKERS", "4")
        monkeypatch.setenv("GUNICORN_PRELOAD", "0")
        monkeypatch.setenv("MODEL_PATH", "/models/model.onnx")
        monkeypatch.setenv("OTEL_ENABLED", "true")

        settings = Settings()

        assert settings.environment is Environment.PRODUCTION
        assert (settings.gunicorn.workers, settings.gunicorn.preload) == (4, False)
        assert settings.model.path == Path("/models/model.onnx")
        assert settings.otel.enabled

    def test_rejects_a_sample_ratio_above_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_SAMPLE_RATIO", "1.5")
        with pytest.raises(ValueError, match="sample_ratio"):
            Settings()

    def test_rejects_a_sha256_that_isnt_one(self) -> None:
        with pytest.raises(ValidationError, match="sha256"):
            ModelSettings(sha256="not-a-hash")


class TestForkSafety:
    def test_refuses_an_onnx_runtime_thread_pool_with_preload(self) -> None:
        with pytest.raises(ValidationError, match="MODEL_INTRA_OP_THREADS > 1 needs GUNICORN_PRELOAD=0"):
            Settings(gunicorn=GunicornSettings(preload=True), model=ModelSettings(intra_op_threads=2))

    def test_allows_one_without_preload(self) -> None:
        Settings(gunicorn=GunicornSettings(preload=False), model=ModelSettings(intra_op_threads=2))
