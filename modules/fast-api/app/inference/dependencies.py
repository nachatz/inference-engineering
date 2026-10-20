from typing import Annotated

from fastapi import Depends
from fastapi import Request

from app.inference.capacity.slot import InferenceSlot
from app.inference.exceptions import WorkerNotStarted
from app.inference.runtime.served import ServedModel


def get_served_model(request: Request) -> ServedModel:
    served: ServedModel = request.app.state.served_model
    return served


def get_inference_slot_or_none(request: Request) -> InferenceSlot | None:
    slot: InferenceSlot | None = getattr(request.state, "inference_slot", None)
    return slot


def get_inference_slot(slot: Annotated[InferenceSlot | None, Depends(get_inference_slot_or_none)]) -> InferenceSlot:
    if slot is None:
        raise WorkerNotStarted
    return slot


ServedModelDependency = Annotated[ServedModel, Depends(get_served_model)]
InferenceSlotDependency = Annotated[InferenceSlot, Depends(get_inference_slot)]
OptionalInferenceSlotDependency = Annotated[InferenceSlot | None, Depends(get_inference_slot_or_none)]
