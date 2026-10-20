import threading
from typing import Any

import anyio
import pytest
from fastapi.exceptions import RequestValidationError
from numpy.typing import NDArray
from pytest_mock import MockerFixture

from app.inference.capacity.slot import InferenceSlot
from app.inference.exceptions import InputMismatch
from app.inference.exceptions import Overloaded
from app.inference.handlers.predict import PredictHandler
from app.inference.runtime.served import ServedModel
from app.inference.runtime.tensor import Feed
from tests.factories.inference import ClassifierRequestFactory


class TestPredictHandler:
    async def test_returns_the_adapters_response(self, served: ServedModel, slot: InferenceSlot) -> None:
        request = ClassifierRequestFactory.for_model(n_features=8, rows=5)
        response = await PredictHandler(served, slot, max_batch=64).predict(request)

        feed = served.adapter.to_feed(request, served.model)
        assert response == served.adapter.from_outputs(served.model.run(feed))

    async def test_rejects_the_wrong_feature_count_before_queueing(self, served: ServedModel) -> None:
        full = InferenceSlot(max_queue=0, max_queue_wait_s=1.0)
        async with full.acquire():
            with pytest.raises(InputMismatch, match=r"\[batch, 8\]"):
                await PredictHandler(served, full, max_batch=64).predict(
                    ClassifierRequestFactory.for_model(n_features=7)
                )

    async def test_refuses_when_the_worker_is_full(self, served: ServedModel) -> None:
        full = InferenceSlot(max_queue=0, max_queue_wait_s=1.0)
        async with full.acquire():
            with pytest.raises(Overloaded):
                await PredictHandler(served, full, max_batch=64).predict(
                    ClassifierRequestFactory.for_model(n_features=8)
                )

    async def test_rejects_a_batch_over_max_batch(self, served: ServedModel, slot: InferenceSlot) -> None:
        with pytest.raises(RequestValidationError, match="at most 2 rows per request"):
            await PredictHandler(served, slot, max_batch=2).predict(
                ClassifierRequestFactory.for_model(n_features=8, rows=3)
            )


class TestCancellation:
    async def test_a_cancelled_request_keeps_the_slot_until_its_inference_ends(
        self, served: ServedModel, slot: InferenceSlot, mocker: MockerFixture
    ) -> None:
        running, release = threading.Event(), threading.Event()
        run = served.model.run

        def held(feed: Feed) -> dict[str, NDArray[Any]]:
            running.set()
            release.wait(timeout=10)
            return run(feed)

        mocker.patch.object(served.model, served.model.run.__name__, side_effect=held)
        request = ClassifierRequestFactory.for_model(n_features=8)

        predicting = anyio.CancelScope()

        async def cancellable() -> None:
            with predicting:
                await PredictHandler(served, slot, max_batch=64).predict(request)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(cancellable)
            await anyio.to_thread.run_sync(running.wait, 10)

            # The client is gone; ONNX Runtime isn't. The slot stays taken until the inference returns.
            predicting.cancel()
            await anyio.sleep(0.1)
            assert slot.busy_for() > 0

            release.set()

        assert slot.busy_for() == 0
