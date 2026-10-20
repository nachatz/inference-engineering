from pathlib import Path

import pytest
from prometheus_client import REGISTRY
from starlette.types import Message
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

from app.observability.prometheus import PredictMetricsMiddleware
from app.observability.prometheus import registry


class TestRegistry:
    def test_default_registry_in_a_single_process(self) -> None:
        assert registry(None) is REGISTRY

    def test_merges_the_worker_files_under_gunicorn(self, tmp_path: Path) -> None:
        assert registry(tmp_path) is not REGISTRY


class TestPredictMetricsMiddleware:
    async def crash(self, scope: Scope, receive: Receive, send: Send) -> None:
        raise RuntimeError("boom")

    async def receive(self) -> Message:
        return {"type": "http.request"}

    async def send(self, message: Message) -> None: ...

    def errors(self) -> float:
        return REGISTRY.get_sample_value("predict_requests_total", {"status": "500"}) or 0.0

    async def test_counts_an_escaped_exception_as_a_500(self) -> None:
        before = self.errors()
        with pytest.raises(RuntimeError):
            await PredictMetricsMiddleware(self.crash, path="/v1/predict")(
                {"type": "http", "path": "/v1/predict"}, self.receive, self.send
            )
        assert self.errors() == before + 1
