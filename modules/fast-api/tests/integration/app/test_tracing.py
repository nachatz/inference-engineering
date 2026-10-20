from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest_mock import MockerFixture

from app.apps.inference import create_app
from app.core.config import OpenTelemetrySettings
from app.core.config import settings
from app.inference.runtime.served import ServedModel
from app.observability.opentelemetry import OpenTelemetryInstrumentor
from tests.factories.inference import ClassifierRequestFactory


class TestTracing:
    def test_traces_a_prediction_down_to_onnx_runtime(
        self, spans: InMemorySpanExporter, served: ServedModel, mocker: MockerFixture
    ) -> None:
        # The session's in-memory tracer provider stands in for the one start() would install.
        mocker.patch.object(OpenTelemetryInstrumentor, OpenTelemetryInstrumentor.start.__name__)
        traced = settings.model_copy(update={"otel": OpenTelemetrySettings(enabled=True)})

        with TestClient(create_app(served, traced)) as client:
            client.get("/healthz")
            client.post("/v1/predict", json=ClassifierRequestFactory.for_model(n_features=8, rows=2).model_dump())

        by_name = {span.name: span for span in spans.get_finished_spans()}
        request, inference = by_name["POST /v1/predict"], by_name["onnxruntime.run"]
        assert inference.parent is not None
        assert inference.parent.span_id == request.context.span_id
        assert inference.attributes == {"inference.rows": 2}
        assert not any("healthz" in name for name in by_name)
