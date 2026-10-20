import gc
import importlib
import sys

import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY
from pytest_mock import MockerFixture

from app.core.config import ModelSettings
from app.core.config import settings
from app.core.preload import preload


class TestPreload:
    def test_loads_the_model_and_reports_it(self, tiny_model_settings: ModelSettings) -> None:
        served = preload(tiny_model_settings)
        labels = {"name": "tiny", "version": "test", "sha256": served.info.sha256}
        assert REGISTRY.get_sample_value("model_info", labels) == 1


class TestMain:
    def test_builds_the_app_with_the_gc_off_then_freezes_the_heap(
        self, tiny_model_settings: ModelSettings, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
    ) -> None:
        monkeypatch.setattr(settings, "model", tiny_model_settings)
        monkeypatch.delitem(sys.modules, "app.main", raising=False)
        disable = mocker.patch.object(gc, gc.disable.__name__)
        freeze = mocker.patch.object(gc, gc.freeze.__name__)
        main = importlib.import_module("app.main")
        assert isinstance(main.app, FastAPI)
        assert main.app.state.served_model.info.name == "tiny"
        disable.assert_called_once()
        freeze.assert_called_once()
