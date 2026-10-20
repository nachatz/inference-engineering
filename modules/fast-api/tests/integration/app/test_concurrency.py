import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

import httpx2
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from prometheus_client import REGISTRY
from pytest_mock import MockerFixture

from app.apps.inference import create_app
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.served import ServedModel
from app.inference.runtime.tensor import Feed
from tests.utils.config import with_model


def post(client: TestClient) -> httpx2.Response:
    return client.post("/v1/predict", json={"inputs": [[0.5] * 8]})


class TestOneWorker:
    """Two promises: inference never blocks the event loop, and a worker runs one inference at a time."""

    def test_health_checks_answer_while_an_inference_runs(
        self, client: TestClient, loaded_model: OnnxModel, mocker: MockerFixture
    ) -> None:
        running, release = threading.Event(), threading.Event()
        run = loaded_model.run

        def held(feed: Feed) -> dict[str, NDArray[Any]]:
            running.set()
            release.wait(timeout=10)
            return run(feed)

        mocker.patch.object(loaded_model, loaded_model.run.__name__, side_effect=held)

        with ThreadPoolExecutor(1) as pool:
            predicting = pool.submit(post, client)
            try:
                assert running.wait(timeout=10)
                assert client.get("/healthz").status_code == HTTPStatus.OK
                assert not predicting.done(), "/healthz waited for the inference"
            finally:
                release.set()

            assert predicting.result(timeout=10).status_code == HTTPStatus.OK

    def test_runs_one_inference_at_a_time(
        self, client: TestClient, loaded_model: OnnxModel, mocker: MockerFixture
    ) -> None:
        lock, running, most = threading.Lock(), 0, 0
        run = loaded_model.run

        def counted(feed: Feed) -> dict[str, NDArray[Any]]:
            nonlocal running, most
            with lock:
                running += 1
                most = max(most, running)

            time.sleep(0.02)
            with lock:
                running -= 1

            return run(feed)

        mocker.patch.object(loaded_model, loaded_model.run.__name__, side_effect=counted)

        with ThreadPoolExecutor(8) as pool:
            responses = list(pool.map(lambda _: post(client), range(8)))

        assert all(response.status_code == HTTPStatus.OK for response in responses)
        assert most == 1


class TestAdmission:
    def test_a_request_over_the_limit_is_refused_and_counted_while_health_checks_answer(
        self, served: ServedModel, loaded_model: OnnxModel, mocker: MockerFixture
    ) -> None:
        running, release = threading.Event(), threading.Event()
        run = loaded_model.run

        def held(feed: Feed) -> dict[str, NDArray[Any]]:
            running.set()
            release.wait(timeout=10)
            return run(feed)

        mocker.patch.object(loaded_model, loaded_model.run.__name__, side_effect=held)
        refused_before = REGISTRY.get_sample_value("predict_requests_total", {"status": "503"}) or 0.0

        with TestClient(create_app(served, with_model(max_admitted=1))) as client, ThreadPoolExecutor(1) as pool:
            predicting = pool.submit(post, client)
            try:
                assert running.wait(timeout=10)

                refused = post(client)
                assert refused.status_code == HTTPStatus.SERVICE_UNAVAILABLE
                assert refused.json()["error"] == "this worker is at its limit of requests"
                assert client.get("/healthz").status_code == HTTPStatus.OK
            finally:
                release.set()

            assert predicting.result(timeout=10).status_code == HTTPStatus.OK
            assert post(client).status_code == HTTPStatus.OK

        assert REGISTRY.get_sample_value("predict_requests_total", {"status": "503"}) == refused_before + 1
