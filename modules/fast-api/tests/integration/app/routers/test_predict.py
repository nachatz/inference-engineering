from collections.abc import Iterator
from http import HTTPStatus

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.apps.inference import create_app
from app.common.enums.adapter import AdapterName
from app.core.config import ModelSettings
from app.core.config import settings
from app.exceptions.models.error import ErrorResponse
from app.inference.adapters.classifier import ClassifierResponse
from app.inference.adapters.scorer import ScorerResponse
from app.inference.runtime.served import ServedModel
from tests.factories.inference import ClassifierRequestFactory
from tests.factories.inference import ScorerRequestFactory
from tests.utils import onnx_graphs
from tests.utils.config import with_model


def post(client: TestClient, rows: int = 1, n_features: int = 8) -> httpx2.Response:
    return client.post("/v1/predict", json={"inputs": [[0.5] * n_features] * rows})


def error(response: httpx2.Response) -> ErrorResponse:
    return ErrorResponse.model_validate(response.json())


@pytest.fixture(scope="module")
def sqrt_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """logits = sqrt(x) * 1e30: NaN for a negative feature, infinite for a large one."""
    path = onnx_graphs.scaled_sqrt(tmp_path_factory.mktemp("sqrt") / "sqrt.onnx")
    with TestClient(create_app(ServedModel.load(ModelSettings(path=path)))) as client:
        yield client


@pytest.fixture(scope="module")
def scorer_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    path = onnx_graphs.scorer(tmp_path_factory.mktemp("scorer") / "scorer.onnx", n_features=3)
    with TestClient(create_app(ServedModel.load(ModelSettings(path=path, adapter=AdapterName.SCORER)))) as client:
        yield client


class TestPredict:
    def test_matches_the_model(self, client: TestClient, served: ServedModel) -> None:
        request = ClassifierRequestFactory.for_model(n_features=8, rows=5)
        response = client.post("/v1/predict", json=request.model_dump())

        assert response.status_code == HTTPStatus.OK
        expected = served.adapter.from_outputs(served.model.run(served.adapter.to_feed(request, served.model)))
        assert ClassifierResponse.model_validate(response.json()) == expected

    def test_largest_batch_is_accepted(self, client: TestClient) -> None:
        assert post(client, rows=settings.model.max_batch).status_code == HTTPStatus.OK

    def test_echoes_the_correlation_id(self, client: TestClient) -> None:
        request_id = "8f4b7e0c2a9d4f6e9b1c3d5e7f9a1b2c"
        response = client.post("/v1/predict", json={"inputs": [[0.5] * 8]}, headers={"X-Request-ID": request_id})
        assert response.headers["X-Request-ID"] == request_id


class TestInvalidRequests:
    def test_wrong_feature_count_is_422(self, client: TestClient) -> None:
        response = post(client, n_features=7)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        (detail,) = error(response).details
        assert detail.message == "input 'x' takes float32 [batch, 8], got float32 [1, 7]"

    def test_empty_batch_is_422(self, client: TestClient) -> None:
        response = post(client, rows=0)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert error(response).details[0].location == ["body", "inputs"]

    def test_oversized_batch_is_422(self, client: TestClient) -> None:
        response = post(client, rows=settings.model.max_batch + 1)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert error(response).details[0].message == f"at most {settings.model.max_batch} rows per request"

    def test_ragged_rows_are_422(self, client: TestClient) -> None:
        response = client.post("/v1/predict", json={"inputs": [[0.5] * 8, [0.5] * 7]})
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert "same number of features" in error(response).details[0].message

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_numbers_are_422(self, client: TestClient, literal: str) -> None:
        body = f'{{"inputs": [[{literal}, 0, 0, 0, 0, 0, 0, 0]]}}'
        response = client.post("/v1/predict", content=body, headers={"content-type": "application/json"})
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert error(response).details[0].location == ["body", "inputs", 0, 0]

    def test_the_batch_limit_is_the_apps(self, served: ServedModel) -> None:
        with TestClient(create_app(served, with_model(max_batch=2))) as client:
            assert post(client, rows=2).status_code == HTTPStatus.OK
            assert post(client, rows=3).status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_body_over_the_limit_is_413(self, served: ServedModel) -> None:
        with TestClient(create_app(served, with_model(max_body_bytes=32))) as client:
            response = post(client)

        assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert response.text == "Content Too Large"

    def test_chunked_body_over_the_limit_is_413(self, served: ServedModel) -> None:
        body = b'{"inputs": [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]}'
        chunks = (body[i : i + 8] for i in range(0, len(body), 8))

        with TestClient(create_app(served, with_model(max_body_bytes=32))) as client:
            response = client.post("/v1/predict", content=chunks, headers={"Content-Type": "application/json"})

        assert "content-length" not in response.request.headers
        assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
        assert error(response).error == "Content Too Large"


class TestNumericalContract:
    """No NaN or infinity reaches the model, and none it computes becomes a prediction."""

    def test_a_value_that_overflows_float32_is_422(self, client: TestClient) -> None:
        # Finite as JSON, which is float64; infinite as the model's float32.
        response = client.post("/v1/predict", json={"inputs": [[1e39] + [0.0] * 7]})

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert error(response).details[0].message == "input 'x' has a value that isn't finite in float32"

    def test_a_finite_result_is_a_label(self, sqrt_client: TestClient) -> None:
        response = sqrt_client.post("/v1/predict", json={"inputs": [[1.0, 4.0, 0.0, 0.0]]})

        assert response.status_code == HTTPStatus.OK
        assert ClassifierResponse.model_validate(response.json()).predictions[0].label == 1

    @pytest.mark.parametrize("value", [-1.0, 1e20], ids=["NaN", "overflow"])
    def test_a_non_finite_result_is_a_500_not_a_label(self, sqrt_client: TestClient, value: float) -> None:
        response = sqrt_client.post("/v1/predict", json={"inputs": [[1.0, value, 0.0, 0.0]]})

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "predictions" not in response.json()
        assert error(response).error == "the model computed a non-finite result: output 'logits' isn't finite"


class TestScorer:
    """The same server with the scorer adapter: named records in, a score each out."""

    def test_scores_each_record(self, scorer_client: TestClient) -> None:
        body = {"rows": [{"features": [1, 0, 0], "segment": 0}, {"features": [0, 0, 1], "segment": 2}]}
        response = scorer_client.post("/v1/predict", json=body)

        assert response.status_code == HTTPStatus.OK
        assert ScorerResponse.model_validate(response.json()).scores == [1.0, 23.0]

    def test_takes_any_batch_up_to_the_limit(self, scorer_client: TestClient) -> None:
        request = ScorerRequestFactory.for_model(n_features=3, rows=settings.model.max_batch)
        response = scorer_client.post("/v1/predict", json=request.model_dump())

        assert len(ScorerResponse.model_validate(response.json()).scores) == settings.model.max_batch

    def test_refuses_a_classifier_request(self, scorer_client: TestClient) -> None:
        response = scorer_client.post("/v1/predict", json={"inputs": [[0.0, 0.0, 0.0]]})

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert error(response).details[0].location == ["body", "rows"]

    def test_publishes_its_own_schema(self, scorer_client: TestClient) -> None:
        schemas = scorer_client.get("/openapi.json").json()["components"]["schemas"]

        assert {"ScorerRequest", "ScorerResponse", "ScoredRow"} <= schemas.keys()
        assert "ClassifierRequest" not in schemas


class TestErrors:
    def test_503_before_the_worker_started(self, unstarted_client: TestClient) -> None:
        response = post(unstarted_client)
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert error(response).error == "the worker hasn't started"

    def test_crash_is_a_500_without_details_but_with_the_request_id(
        self, app: FastAPI, served: ServedModel, mocker: MockerFixture
    ) -> None:
        mocker.patch.object(served.model, served.model.run.__name__, side_effect=RuntimeError("secret"))

        with TestClient(app, raise_server_exceptions=False) as crashing:
            response = post(crashing)

        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        body = error(response)
        assert (body.error, body.details) == ("internal server error", [])
        assert body.request_id is not None
        assert response.headers["X-Request-ID"] == body.request_id

    def test_unknown_route_has_the_same_error_shape(self, client: TestClient) -> None:
        response = client.get("/v2/predict")
        assert response.status_code == HTTPStatus.NOT_FOUND
        assert error(response).error == "Not Found"
        assert error(response).request_id == response.headers["X-Request-ID"]

    def test_wrong_method_keeps_its_allow_header(self, client: TestClient) -> None:
        response = client.get("/v1/predict")
        assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED
        assert response.headers["Allow"] == "POST"
        assert error(response).error == "Method Not Allowed"
