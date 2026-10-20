import os

from app.common.enums.health import ReadyStatus
from app.health.handlers.health import HealthHandler
from app.inference.capacity.slot import InferenceSlot
from app.inference.runtime.served import ServedModel


class TestHealthHandler:
    def test_ready_once_the_worker_has_its_slot(self, served: ServedModel, slot: InferenceSlot) -> None:
        readiness = HealthHandler(served, slot).readiness()
        assert (readiness.status, readiness.worker, readiness.ready) == (ReadyStatus.READY, os.getpid(), True)
        assert readiness.model == served.info

    def test_not_ready_before_the_worker_starts(self, served: ServedModel) -> None:
        assert HealthHandler(served, None).readiness().status is ReadyStatus.NOT_READY
