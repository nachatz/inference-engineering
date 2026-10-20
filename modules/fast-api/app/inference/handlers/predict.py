from typing import Any

import anyio.to_thread
from fastapi.exceptions import RequestValidationError
from numpy.typing import NDArray
from opentelemetry import trace
from pydantic import BaseModel

from app.inference.capacity.slot import InferenceSlot
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.served import ServedModel
from app.inference.runtime.tensor import Feed
from app.observability.metrics import INFERENCE_SECONDS
from app.observability.metrics import ROWS

tracer = trace.get_tracer(__name__)


class PredictHandler:
    """One prediction: the adapter's feed, the batch limit, then the inference behind the worker's slot.

    Building the feed checks every input, so a malformed request is refused before it takes a place in the queue. The
    inference runs in a thread, so the event loop keeps answering health checks. A thread can't be stopped: a
    cancelled request (a client gone, a worker shutting down) still waits for its inference to end, so the slot is
    never free while the core is busy.
    """

    def __init__(self, served: ServedModel, slot: InferenceSlot, max_batch: int) -> None:
        self._model = served.model
        self._adapter = served.adapter
        self._slot = slot
        self._max_batch = max_batch

    async def predict(self, request: BaseModel) -> BaseModel:
        feed = self._adapter.to_feed(request, self._model)

        rows = len(next(iter(feed.values())))
        if rows > self._max_batch:
            message = f"at most {self._max_batch} rows per request"
            raise RequestValidationError([{"loc": ("body",), "msg": message, "type": "too_long"}])

        ROWS.observe(rows)

        async with self._slot.acquire():
            outputs = await anyio.to_thread.run_sync(self._run, self._model, feed, abandon_on_cancel=False)

        response: BaseModel = self._adapter.from_outputs(outputs)
        return response

    def _run(self, model: OnnxModel, feed: Feed) -> dict[str, NDArray[Any]]:
        with tracer.start_as_current_span("onnxruntime.run") as span, INFERENCE_SECONDS.time():
            span.set_attribute("inference.rows", len(next(iter(feed.values()))))
            return model.run(feed)
