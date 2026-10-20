from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from pytest_mock import MockerFixture

from app.common.enums.environment import Environment
from app.core.config import OpenTelemetrySettings
from app.observability.opentelemetry import OpenTelemetryInstrumentor


class TestOpenTelemetryInstrumentor:
    def instrumentor(self, enabled: bool) -> OpenTelemetryInstrumentor:
        return OpenTelemetryInstrumentor(OpenTelemetrySettings(enabled=enabled), Environment.LOCAL)

    def test_does_nothing_when_disabled(self, mocker: MockerFixture) -> None:
        set_provider = mocker.patch.object(trace, trace.set_tracer_provider.__name__)
        instrument_app = mocker.patch.object(FastAPIInstrumentor, FastAPIInstrumentor.instrument_app.__name__)
        instrumentor = self.instrumentor(enabled=False)
        instrumentor.instrument(FastAPI())
        instrumentor.start()
        instrumentor.shutdown()
        set_provider.assert_not_called()
        instrument_app.assert_not_called()

    def test_instruments_the_app_without_its_health_checks(self, mocker: MockerFixture) -> None:
        instrument_app = mocker.patch.object(FastAPIInstrumentor, FastAPIInstrumentor.instrument_app.__name__)
        app = FastAPI()
        self.instrumentor(enabled=True).instrument(app)
        instrument_app.assert_called_once_with(app, excluded_urls="healthz,readyz,metrics,debug")

    def test_starts_a_tracer_provider_per_worker_and_shuts_it_down(self, mocker: MockerFixture) -> None:
        set_provider = mocker.patch.object(trace, trace.set_tracer_provider.__name__)
        instrumentor = self.instrumentor(enabled=True)
        instrumentor.start()
        (provider,) = set_provider.call_args.args
        assert isinstance(provider, TracerProvider)
        assert provider.resource.attributes["service.name"] == "fast-api"
        shutdown = mocker.spy(provider, provider.shutdown.__name__)
        instrumentor.shutdown()
        shutdown.assert_called_once()
