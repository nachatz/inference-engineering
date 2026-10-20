import gc
import os
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI

from app.core.config import ModelSettings
from app.core.logger import logger
from app.core.outputs import WorkerReady
from app.inference.capacity.slot import InferenceSlot
from app.inference.capacity.slot import exit_worker
from app.inference.capacity.slot import watch
from app.observability.metrics import WORKERS
from app.observability.opentelemetry import OpenTelemetryInstrumentor

type Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[dict[str, Any]]]


def worker_lifespan(
    instrumentor: OpenTelemetryInstrumentor, model: ModelSettings, on_stuck: Callable[[float], None] = exit_worker
) -> Lifespan:
    """Each worker's own state, made after the fork: the tracer provider, the inference slot and its watchdog. The
    slot reaches requests as lifespan state. Never load the model here: that would be a copy per worker. It also
    enables the GC again, which app.main disabled while the shared heap was built."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[dict[str, Any]]:
        gc.enable()
        slot = InferenceSlot(model.max_queue, model.max_queue_wait_s)

        async with AsyncExitStack() as stack:
            instrumentor.start()
            stack.callback(instrumentor.shutdown)

            watchdog = await stack.enter_async_context(anyio.create_task_group())
            watchdog.start_soon(watch, slot, model.inference_stuck_s, on_stuck)
            stack.callback(watchdog.cancel_scope.cancel)

            WORKERS.set(1)
            logger.info(WorkerReady(pid=os.getpid()))
            yield {"inference_slot": slot}

    return lifespan
