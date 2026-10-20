import os

from app.common.enums.health import ReadyStatus
from app.health.models.health import ReadyResponse
from app.inference.capacity.slot import InferenceSlot
from app.inference.runtime.served import ServedModel


class HealthHandler:
    """A worker is ready once its lifespan has made its inference slot: the model was loaded before the app existed."""

    def __init__(self, served: ServedModel, slot: InferenceSlot | None) -> None:
        self._served = served
        self._slot = slot

    def readiness(self) -> ReadyResponse:
        status = ReadyStatus.READY if self._slot is not None else ReadyStatus.NOT_READY
        return ReadyResponse(status=status, worker=os.getpid(), model=self._served.info)
