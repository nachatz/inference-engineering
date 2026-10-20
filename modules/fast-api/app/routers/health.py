from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Response

from app.common.enums.routes import RouteTags
from app.health.handlers.health import HealthHandler
from app.health.models.health import HealthResponse
from app.health.models.health import ReadyResponse
from app.inference.dependencies import OptionalInferenceSlotDependency
from app.inference.dependencies import ServedModelDependency

health_router = APIRouter(tags=[RouteTags.HEALTH])


def get_health_handler(served: ServedModelDependency, slot: OptionalInferenceSlotDependency) -> HealthHandler:
    return HealthHandler(served, slot)


HealthHandlerDependency = Annotated[HealthHandler, Depends(get_health_handler)]


@health_router.get("/healthz")
async def healthz() -> HealthResponse:
    """Liveness: this worker's event loop answers."""
    return HealthResponse()


@health_router.get("/readyz", responses={HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ReadyResponse}})
async def readyz(response: Response, handler: HealthHandlerDependency) -> ReadyResponse:
    """Readiness: this worker has started, so its model is loaded and warmed."""
    readiness = handler.readiness()
    if not readiness.ready:
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE

    return readiness
