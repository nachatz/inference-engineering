import os
import time
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import asynccontextmanager

import anyio

from app.core.logger import logger
from app.core.outputs import InferenceStuck
from app.inference.exceptions import Overloaded
from app.observability.metrics import INFLIGHT
from app.observability.metrics import QUEUE_DEPTH
from app.observability.metrics import SHED


class InferenceSlot:
    """One inference at a time in this worker, behind a bounded queue. Made per worker, in the lifespan.

    `acquire` holds the slot for one inference. A request that finds `max_queue` others already waiting, or waits
    longer than `max_queue_wait_s`, raises Overloaded: a fast 503 the client can retry elsewhere, instead of an answer
    it has likely stopped waiting for. `busy_for` is how long the running inference has run, 0 when idle: what the
    watchdog reads.
    """

    def __init__(self, max_queue: int, max_queue_wait_s: float) -> None:
        self._limiter = anyio.CapacityLimiter(1)
        self._max_queue = max_queue
        self._max_queue_wait_s = max_queue_wait_s
        self._busy_since: float | None = None

    @property
    def waiting(self) -> int:
        return self._limiter.statistics().tasks_waiting

    def busy_for(self) -> float:
        return 0.0 if self._busy_since is None else time.monotonic() - self._busy_since

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[None]:
        if self._limiter.borrowed_tokens and self.waiting >= self._max_queue:
            SHED.labels(reason="queue_full").inc()
            raise Overloaded("this worker's queue is full")

        try:
            with QUEUE_DEPTH.track_inprogress(), anyio.fail_after(self._max_queue_wait_s):
                await self._limiter.acquire()
        except TimeoutError:
            SHED.labels(reason="queue_timeout").inc()
            raise Overloaded("the request waited too long for this worker") from None

        try:
            self._busy_since = time.monotonic()
            with INFLIGHT.track_inprogress():
                yield
        finally:
            self._busy_since = None
            self._limiter.release()


def exit_worker(seconds: float) -> None:
    """A stuck inference holds the worker's one slot for good, and a thread can't be cancelled: exit, and gunicorn
    forks a replacement from the master, which still shares the model. Code 1, since gunicorn halts on 3 and 4."""
    logger.critical(InferenceStuck(pid=os.getpid(), seconds=round(seconds, 1)))
    os._exit(1)


async def watch(slot: InferenceSlot, stuck_after_s: float, on_stuck: Callable[[float], None]) -> None:
    """Call `on_stuck` once an inference has run longer than `stuck_after_s`. A liveness probe can't see this: the
    event loop still answers, and a probe reaches one random worker of many."""
    interval = min(1.0, stuck_after_s / 4)

    while True:
        await anyio.sleep(interval)
        if (seconds := slot.busy_for()) > stuck_after_s:
            on_stuck(seconds)
            return
