from pathlib import Path

import pytest

from app.common.enums.environment import Environment
from app.core.config import GunicornSettings
from app.core.config import PrometheusSettings
from app.core.config import Settings


class TestToEnv:
    def test_renders_each_field_under_its_prefix(self) -> None:
        env = GunicornSettings(workers=4, preload=False, host="127.0.0.1").to_env()
        assert env["GUNICORN_WORKERS"] == "4"
        assert env["GUNICORN_PRELOAD"] == "false"
        assert env["GUNICORN_HOST"] == "127.0.0.1"

    def test_leaves_out_unset_optional_fields(self) -> None:
        assert "GUNICORN_WORKERS" not in GunicornSettings(workers=None).to_env()
        assert PrometheusSettings(multiproc_dir=None).to_env() == {}

    def test_round_trips_nested_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = Settings(
            environment=Environment.STAGING,
            gunicorn=GunicornSettings(workers=3, preload=False),
            prometheus=PrometheusSettings(multiproc_dir=Path("/tmp/metrics")),
        )
        for name, value in settings.to_env().items():
            monkeypatch.setenv(name, value)
        assert Settings() == settings
