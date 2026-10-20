from http import HTTPStatus

from fastapi.testclient import TestClient

from app.common.enums.health import HealthStatus
from app.common.enums.health import ReadyStatus
from app.health.models.health import HealthResponse
from app.health.models.health import ReadyResponse
from app.inference.runtime.served import ServedModel


class TestHealthz:
    def test_ok(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == HTTPStatus.OK
        assert HealthResponse.model_validate(response.json()).status is HealthStatus.OK


class TestReadyz:
    def test_ready_once_the_worker_started(self, client: TestClient, served: ServedModel) -> None:
        response = client.get("/readyz")
        assert response.status_code == HTTPStatus.OK
        readiness = ReadyResponse.model_validate(response.json())
        assert readiness.status is ReadyStatus.READY
        assert readiness.model == served.info

    def test_503_before_the_worker_started(self, unstarted_client: TestClient) -> None:
        response = unstarted_client.get("/readyz")
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert ReadyResponse.model_validate(response.json()).status is ReadyStatus.NOT_READY
