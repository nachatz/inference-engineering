from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends

from app.common.enums.routes import RouteTags
from app.common.memory import ProcessMemory
from app.core.dependencies import ConfigDependency
from app.debug.handlers.debug import DebugHandler
from app.debug.models.debug import AllocatorResponse
from app.debug.models.debug import FlagsResponse
from app.debug.models.debug import ModelResponse
from app.debug.models.debug import RuntimeResponse
from app.inference.dependencies import ServedModelDependency

debug_router = APIRouter(prefix="/debug", tags=[RouteTags.DEBUG], include_in_schema=False)


def get_debug_handler(served: ServedModelDependency, config: ConfigDependency) -> DebugHandler:
    return DebugHandler(served, config)


DebugHandlerDependency = Annotated[DebugHandler, Depends(get_debug_handler)]


@debug_router.get("/flags")
async def flags(handler: DebugHandlerDependency) -> FlagsResponse:
    return handler.flags()


@debug_router.get("/allocator")
async def allocator(handler: DebugHandlerDependency) -> AllocatorResponse:
    return handler.allocator()


@debug_router.get("/memory")
async def memory(handler: DebugHandlerDependency) -> ProcessMemory:
    return handler.memory()


@debug_router.get("/runtime")
async def runtime(handler: DebugHandlerDependency) -> RuntimeResponse:
    return handler.runtime()


@debug_router.get("/model")
async def model(handler: DebugHandlerDependency) -> ModelResponse:
    return handler.model()
