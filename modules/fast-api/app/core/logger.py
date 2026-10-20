import json
import logging
import logging.config
from datetime import UTC
from datetime import datetime
from typing import Any

from asgi_correlation_id import correlation_id
from opentelemetry import trace

from app.common.enums.logging import LogFormat
from app.common.outputs import Output
from app.core.config import settings

logger = logging.getLogger("fast_api")

LOGGERS = ["fast_api", "gunicorn.error", "gunicorn.access", "uvicorn", "uvicorn.error", "uvicorn.access"]

ACCESS_FIELDS = ("client", "method", "path", "http_version", "status")


def request_context() -> dict[str, str]:
    """The request ID and the active span of the request this task or thread is serving, if any."""
    context: dict[str, str] = {}
    if request_id := correlation_id.get():
        context["request_id"] = request_id

    span = trace.get_current_span().get_span_context()
    if span.is_valid:
        context["trace_id"] = format(span.trace_id, "032x")
        context["span_id"] = format(span.span_id, "016x")

    return context


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the request ID and the trace of the request it was logged in.

    An Output's fields become keys of their own, as do an access log's: uvicorn passes those as the record's args,
    in ACCESS_FIELDS order. The common keys (time, level, logger, pid, message) win over a field of the same name.
    """

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {}
        if isinstance(record.msg, Output):
            line |= {"event": record.msg.event, **record.msg.model_dump(mode="json")}
        elif record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            line |= dict(zip(ACCESS_FIELDS, record.args, strict=True))

        line |= {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "pid": record.process,
            "message": record.getMessage(),
            **request_context(),
        }
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)

        return json.dumps(line, default=str)


class TextFormatter(logging.Formatter):
    """For people: the benchmark and export tools, and `just dev`."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = correlation_id.get() or "-"
        return super().format(record)


def logging_config(level: str, log_format: LogFormat) -> dict[str, Any]:
    """One format for this service's logger, gunicorn's and uvicorn's, so an access log line and a 500's traceback
    carry the same request ID as everything else. gunicorn.conf.py hands it to gunicorn as logconfig_dict."""
    formatter = JsonFormatter if log_format is LogFormat.JSON else TextFormatter

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"()": formatter}},
        "handlers": {
            "stdout": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stdout"}
        },
        "root": {"level": "WARNING", "handlers": ["stdout"]},
        "loggers": {name: {"level": level, "handlers": ["stdout"], "propagate": False} for name in LOGGERS},
    }


def configure_logging(log_format: LogFormat | None = None) -> None:
    logging.config.dictConfig(logging_config(settings.log_level, log_format or settings.log_format))
