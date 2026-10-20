import gc
from unittest.mock import Mock

import anyio
from fastapi import FastAPI
from pytest_mock import MockerFixture

from app.apps.lifespan import worker_lifespan
from app.common.enums.environment import Environment
from app.core.config import ModelSettings
from app.core.config import OpenTelemetrySettings
from app.inference.capacity.slot import InferenceSlot
from app.observability.opentelemetry import OpenTelemetryInstrumentor


class TestWorkerLifespan:
    def instrumentor(self) -> OpenTelemetryInstrumentor:
        return OpenTelemetryInstrumentor(OpenTelemetrySettings(), Environment.LOCAL)

    async def test_makes_the_per_worker_state_and_tears_it_down(self, mocker: MockerFixture) -> None:
        instrumentor = self.instrumentor()
        start = mocker.spy(instrumentor, instrumentor.start.__name__)
        shutdown = mocker.spy(instrumentor, instrumentor.shutdown.__name__)
        async with worker_lifespan(instrumentor, ModelSettings())(FastAPI()) as state:
            assert isinstance(state["inference_slot"], InferenceSlot)
            start.assert_called_once()
            shutdown.assert_not_called()
        shutdown.assert_called_once()

    async def test_enables_the_gc_the_master_disabled(self, mocker: MockerFixture) -> None:
        enable = mocker.patch.object(gc, gc.enable.__name__)
        async with worker_lifespan(self.instrumentor(), ModelSettings())(FastAPI()):
            enable.assert_called_once()

    async def test_watches_for_a_stuck_inference(self) -> None:
        on_stuck = Mock()
        lifespan = worker_lifespan(self.instrumentor(), ModelSettings(inference_stuck_s=0.04), on_stuck)
        async with lifespan(FastAPI()) as state, state["inference_slot"].acquire():
            await anyio.sleep(0.2)
        on_stuck.assert_called_once()
