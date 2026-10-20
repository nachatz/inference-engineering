from http import HTTPStatus

from fastapi.testclient import TestClient

from tests.factories.inference import ClassifierRequestFactory


class TestMetrics:
    def test_exposes_the_prediction_metrics(self, client: TestClient) -> None:
        client.post("/v1/predict", json=ClassifierRequestFactory.for_model(n_features=8, rows=3).model_dump())
        client.post("/v1/predict", json={"inputs": []})

        response = client.get("/metrics")

        assert response.status_code == HTTPStatus.OK
        assert response.headers["content-type"].startswith("text/plain")
        for sample in [
            'predict_requests_total{status="200"}',
            'predict_requests_total{status="422"}',
            "predict_request_seconds_bucket",
            "predict_inference_seconds_bucket",
            "predict_rows_bucket",
            "predict_queue_depth",
            "predict_inflight",
        ]:
            assert sample in response.text

    def test_is_not_in_the_api_schema(self, client: TestClient) -> None:
        assert "/metrics" not in client.get("/openapi.json").json()["paths"]
