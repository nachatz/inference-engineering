import time
from http import HTTPStatus
from pathlib import Path

from prometheus_client import REGISTRY
from prometheus_client import CollectorRegistry
from prometheus_client import multiprocess
from starlette.types import ASGIApp
from starlette.types import Message
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

from app.observability.metrics import REQUEST_SECONDS
from app.observability.metrics import REQUESTS


def registry(multiproc_dir: Path | None) -> CollectorRegistry:
    """Under gunicorn every worker writes its own metric files, and the collector merges them on every scrape, so
    any worker can answer /metrics."""
    if multiproc_dir is None:
        return REGISTRY

    merged = CollectorRegistry()
    multiprocess.MultiProcessCollector(merged, path=str(multiproc_dir))  # type: ignore[no-untyped-call]  # unannotated
    return merged


class PredictMetricsMiddleware:
    """Counts and times the requests to one path, by status. An exception that escapes the app counts as a 500."""

    def __init__(self, app: ASGIApp, path: str) -> None:
        self.app = app
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return

        status = HTTPStatus.INTERNAL_SERVER_ERROR.value

        async def send_and_record_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_and_record_status)
        finally:
            REQUEST_SECONDS.observe(time.perf_counter() - start)
            REQUESTS.labels(status=str(status)).inc()
