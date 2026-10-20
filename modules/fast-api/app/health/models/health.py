from pydantic import BaseModel

from app.common.enums.health import HealthStatus
from app.common.enums.health import ReadyStatus
from app.inference.runtime.info import ModelInfo


class HealthResponse(BaseModel):
    status: HealthStatus = HealthStatus.OK


class ReadyResponse(BaseModel):
    """`worker` is the answering worker's pid, so clients and the benchmarks can tell the workers apart."""

    status: ReadyStatus
    worker: int
    model: ModelInfo

    @property
    def ready(self) -> bool:
        return self.status is ReadyStatus.READY
