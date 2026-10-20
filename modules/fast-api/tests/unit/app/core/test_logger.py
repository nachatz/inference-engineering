import json
import logging
import os
import sys
from collections.abc import Iterator

import pytest
from asgi_correlation_id import correlation_id
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.common.enums.logging import LogFormat
from app.core.logger import LOGGERS
from app.core.logger import JsonFormatter
from app.core.logger import TextFormatter
from app.core.logger import configure_logging
from app.core.outputs import InferenceStuck


def record(msg: object, name: str = "fast_api", args: tuple[object, ...] = ()) -> logging.LogRecord:
    return logging.LogRecord(name, logging.INFO, __file__, 1, msg, args, None)


@pytest.fixture
def request_id() -> Iterator[str]:
    token = correlation_id.set("8f4b7e0c2a9d4f6e9b1c3d5e7f9a1b2c")
    yield "8f4b7e0c2a9d4f6e9b1c3d5e7f9a1b2c"
    correlation_id.reset(token)


class TestConfigureLogging:
    @pytest.mark.parametrize(
        ("log_format", "formatter"), [(LogFormat.JSON, JsonFormatter), (LogFormat.TEXT, TextFormatter)]
    )
    def test_gives_every_logger_one_handler(self, log_format: LogFormat, formatter: type[logging.Formatter]) -> None:
        configure_logging(log_format)
        configure_logging(log_format)
        for name in LOGGERS:
            (handler,) = logging.getLogger(name).handlers
            assert isinstance(handler.formatter, formatter)
            assert not logging.getLogger(name).propagate


class TestJsonFormatter:
    def line(self, record: logging.LogRecord) -> dict[str, object]:
        parsed: dict[str, object] = json.loads(JsonFormatter().format(record))
        return parsed

    def test_one_object_per_line(self) -> None:
        line = self.line(record("hi"))
        assert (line["level"], line["logger"], line["message"]) == ("INFO", "fast_api", "hi")
        assert {"time", "pid"} <= line.keys()
        assert "request_id" not in line

    def test_carries_the_request_and_its_trace(self, request_id: str, spans: InMemorySpanExporter) -> None:
        with trace.get_tracer(__name__).start_as_current_span("request") as span:
            line = self.line(record("hi"))
        assert line["request_id"] == request_id
        assert line["trace_id"] == format(span.get_span_context().trace_id, "032x")
        assert line["span_id"] == format(span.get_span_context().span_id, "016x")

    def test_an_output_logs_its_event_and_fields(self) -> None:
        line = self.line(record(InferenceStuck(pid=42, seconds=31.2)))
        assert (line["event"], line["seconds"]) == ("inference_stuck", 31.2)
        assert line["message"] == "worker 42's inference has run for 31.2 s: exiting so gunicorn replaces it"
        # The common keys win: `pid` is always the process that wrote the line.
        assert line["pid"] == os.getpid()

    def test_the_access_log_gets_fields(self) -> None:
        access = record(
            '%s - "%s %s HTTP/%s" %d', "uvicorn.access", ("10.0.0.1:5000", "POST", "/v1/predict", "1.1", 200)
        )
        line = self.line(access)
        assert (line["method"], line["path"], line["status"]) == ("POST", "/v1/predict", 200)

    def test_includes_the_traceback(self) -> None:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            failed = logging.LogRecord("fast_api", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
        assert "RuntimeError: boom" in str(self.line(failed)["exception"])


class TestTextFormatter:
    def test_tags_the_line_with_the_request_id(self, request_id: str) -> None:
        assert TextFormatter().format(record("hi")).endswith(f"[fast_api] [{request_id}] hi")

    def test_a_dash_outside_a_request(self) -> None:
        assert TextFormatter().format(record("hi")).endswith("[fast_api] [-] hi")
