from collections.abc import Iterator
from http import HTTPStatus
from typing import Any

import pytest
import sentry_sdk
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from app.apps.inference import create_app
from app.inference.exceptions import Overloaded
from app.inference.runtime.served import ServedModel
from app.observability.sentry import report
from app.observability.sentry import tag_request_id


class Captured(Transport):
    """A Sentry transport that keeps every event instead of sending it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        self.events += [item.payload.json for item in envelope.items if item.type == "event" and item.payload.json]


@pytest.fixture
def sentry_events() -> Iterator[list[dict[str, Any]]]:
    transport = Captured()
    sentry_sdk.init(dsn="https://key@sentry.example/1", transport=transport, before_send=tag_request_id)
    try:
        yield transport.events
    finally:
        sentry_sdk.get_client().close()
        sentry_sdk.init()


@pytest.fixture
def app(served: ServedModel) -> FastAPI:
    return create_app(served)


def post(client: TestClient) -> int:
    return client.post("/v1/predict", json={"inputs": [[0.5] * 8]}).status_code


class TestSentry:
    def test_a_crash_is_reported_with_its_request_id(
        self, sentry_events: list[dict[str, Any]], app: FastAPI, served: ServedModel, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(served.model, served.model.run.__name__, side_effect=RuntimeError("inference crashed"))

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/v1/predict", json={"inputs": [[0.5] * 8]}, headers={"X-Request-ID": "a" * 32})

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        (event,) = sentry_events
        assert "inference crashed" in str(event["exception"])
        assert event["tags"]["request_id"] == "a" * 32

    def test_a_refused_request_is_not_an_error(
        self, sentry_events: list[dict[str, Any]], app: FastAPI, served: ServedModel, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(served.model, served.model.run.__name__, side_effect=Overloaded("busy"))

        with TestClient(app) as client:
            assert post(client) == HTTPStatus.SERVICE_UNAVAILABLE
            assert client.get("/healthz").status_code == HTTPStatus.OK

        assert sentry_events == []

    def test_report_sends_at_once(self, sentry_events: list[dict[str, Any]]) -> None:
        report(RuntimeError("the model didn't load"))

        (event,) = sentry_events
        assert "the model didn't load" in str(event["exception"])
        assert "request_id" not in event.get("tags", {})


class TestWithoutSentry:
    def test_report_does_nothing(self) -> None:
        report(RuntimeError("nobody is listening"))
