import os
from uuid import uuid4

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT
from opentelemetry.sdk.resources import PROCESS_PID
from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio

from app.common.enums.environment import Environment
from app.core.config import OpenTelemetrySettings


class OpenTelemetryInstrumentor:
    """Request tracing, exported over OTLP/HTTP.

    The app is instrumented when it's built, which with preload_app is in the gunicorn master. The tracer provider,
    and the exporter thread it starts, are created in each worker after the fork: a thread doesn't survive fork().
    """

    def __init__(self, otel: OpenTelemetrySettings, environment: Environment) -> None:
        self._otel = otel
        self._environment = environment
        self._provider: TracerProvider | None = None

    def instrument(self, app: FastAPI) -> None:
        if self._otel.enabled:
            FastAPIInstrumentor.instrument_app(app, excluded_urls=self._otel.excluded_urls)

    def start(self) -> None:
        if not self._otel.enabled:
            return

        resource = Resource.create(
            {
                SERVICE_NAME: self._otel.service_name,
                SERVICE_INSTANCE_ID: str(uuid4()),
                DEPLOYMENT_ENVIRONMENT: self._environment.value,
                PROCESS_PID: os.getpid(),
            }
        )

        self._provider = TracerProvider(resource=resource, sampler=ParentBasedTraceIdRatio(self._otel.sample_ratio))
        exporter = OTLPSpanExporter(endpoint=f"{self._otel.exporter_otlp_endpoint}/v1/traces")
        self._provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(self._provider)

    def shutdown(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()
            self._provider = None
