from http import HTTPStatus

import psutil
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytest_mock import MockerFixture

from app.apps.inference import create_app
from app.common.enums.environment import Environment
from app.common.memory import ProcessMemory
from app.core.config import Settings
from app.core.config import settings
from app.debug.models.debug import AllocatorResponse
from app.debug.models.debug import FlagsResponse
from app.debug.models.debug import ModelResponse
from app.debug.models.debug import RuntimeResponse
from app.exceptions.models.error import ErrorResponse
from app.inference.runtime.served import ServedModel


class TestDebug:
    @pytest.mark.parametrize(
        ("path", "model"),
        [
            ("/debug/flags", FlagsResponse),
            ("/debug/allocator", AllocatorResponse),
            ("/debug/runtime", RuntimeResponse),
            ("/debug/model", ModelResponse),
        ],
    )
    def test_answers(self, client: TestClient, path: str, model: type[BaseModel]) -> None:
        response = client.get(path)

        assert response.status_code == HTTPStatus.OK
        model.model_validate(response.json())

    def test_memory_on_linux(self, client: TestClient, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> None:
        worker = ProcessMemory(pid=1, rss=400, pss=100, private=10)
        monkeypatch.setattr(psutil, "LINUX", True)
        mocker.patch.object(ProcessMemory, ProcessMemory.of.__name__, return_value=worker)

        response = client.get("/debug/memory")

        assert response.status_code == HTTPStatus.OK
        assert ProcessMemory.model_validate(response.json()) == worker

    def test_memory_is_501_off_linux(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(psutil, "LINUX", False)

        response = client.get("/debug/memory")

        assert response.status_code == HTTPStatus.NOT_IMPLEMENTED
        assert ErrorResponse.model_validate(response.json()).error.startswith("this needs Linux")

    def test_flags_report_the_apps_settings(self, client: TestClient) -> None:
        assert FlagsResponse.model_validate(client.get("/debug/flags").json()).max_batch == settings.model.max_batch

    def test_is_not_in_the_api_schema(self, client: TestClient) -> None:
        assert not any(path.startswith("/debug") for path in client.get("/openapi.json").json()["paths"])

    def test_on_in_every_environment_with_the_api_docs(self, served: ServedModel) -> None:
        with TestClient(create_app(served, Settings(environment=Environment.PRODUCTION))) as client:
            for path in ["/debug/flags", "/docs", "/openapi.json"]:
                assert client.get(path).status_code == HTTPStatus.OK, path
