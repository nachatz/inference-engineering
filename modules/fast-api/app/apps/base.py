from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import ClassVar

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import APIRouter
from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.apps.lifespan import Lifespan
from app.core.config import Settings
from app.exceptions.handlers import ExceptionHandler
from app.exceptions.handlers import HTTPExceptionHandler
from app.exceptions.handlers import ProcUnavailableHandler
from app.exceptions.handlers import RequestValidationErrorHandler
from app.exceptions.handlers import UnhandledExceptionHandler
from app.observability.opentelemetry import OpenTelemetryInstrumentor
from app.observability.prometheus import registry
from app.routers.debug import debug_router
from app.routers.health import health_router


class BaseApp(ABC):
    """A FastAPI app with what every app here shares, built from class variables.

    Shared by every app
        - routers: /healthz, /readyz and /debug, plus /metrics from prometheus_client.
        - exception handlers: one error shape for validation errors, HTTP errors and the unhandled 500.
        - middleware: request IDs and tracing, outermost, so they cover everything a subclass adds.

    A subclass adds
        - `routers` and `exception_handlers`, registered after the shared ones. A handler for an exception a shared
          one covers takes its place.
        - `lifespan`: each worker's startup and shutdown, which must start and shut down the instrumentor.
        - `configure`, optionally: routes and middleware that need the instance, such as the served model.
    """

    shared_routers: ClassVar[list[APIRouter]] = [health_router, debug_router]
    shared_exception_handlers: ClassVar[list[type[ExceptionHandler[Any]]]] = [
        RequestValidationErrorHandler,
        HTTPExceptionHandler,
        ProcUnavailableHandler,
        UnhandledExceptionHandler,
    ]

    routers: ClassVar[list[APIRouter]] = []
    exception_handlers: ClassVar[list[type[ExceptionHandler[Any]]]] = []

    def __init__(self, config: Settings) -> None:
        instrumentor = OpenTelemetryInstrumentor(config.otel, config.environment)
        self.app = FastAPI(title=config.app_title, lifespan=self.lifespan(instrumentor, config))
        self.app.state.config = config

        for router in [*self.shared_routers, *self.routers]:
            self.app.include_router(router)

        self.app.mount("/metrics", make_asgi_app(registry(config.prometheus.multiproc_dir)))

        for handler in [*self.shared_exception_handlers, *self.exception_handlers]:
            self.app.add_exception_handler(handler.exception, handler())

        self.configure(config)
        self.app.add_middleware(CorrelationIdMiddleware)
        instrumentor.instrument(self.app)

    @abstractmethod
    def lifespan(self, instrumentor: OpenTelemetryInstrumentor, config: Settings) -> Lifespan: ...

    def configure(self, config: Settings) -> None: ...
