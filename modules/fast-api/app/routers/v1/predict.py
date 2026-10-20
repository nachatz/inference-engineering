from http import HTTPStatus
from typing import Annotated
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel

from app.common.enums.routes import RouteTags
from app.core.dependencies import ConfigDependency
from app.exceptions.models.error import ErrorResponse
from app.inference.adapters.base import ModelAdapter
from app.inference.dependencies import InferenceSlotDependency
from app.inference.dependencies import ServedModelDependency
from app.inference.handlers.predict import PredictHandler


def get_predict_handler(
    served: ServedModelDependency, slot: InferenceSlotDependency, config: ConfigDependency
) -> PredictHandler:
    return PredictHandler(served, slot, config.model.max_batch)


PredictHandlerDependency = Annotated[PredictHandler, Depends(get_predict_handler)]


def predict_router(adapter: ModelAdapter[Any, Any]) -> APIRouter:
    """POST /v1/predict with the adapter's request and response models. FastAPI validates the body and builds the
    schema from the endpoint's annotations, so they are set to the adapter's models."""
    router = APIRouter(prefix="/v1", tags=[RouteTags.INFERENCE])

    async def predict(request: BaseModel, handler: PredictHandlerDependency) -> BaseModel:
        return await handler.predict(request)

    predict.__annotations__ |= {"request": adapter.request, "return": adapter.response}

    router.add_api_route(
        "/predict",
        predict,
        methods=["POST"],
        responses={
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value: {"model": ErrorResponse},
            HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
            HTTPStatus.SERVICE_UNAVAILABLE.value: {"model": ErrorResponse},
        },
    )
    return router
