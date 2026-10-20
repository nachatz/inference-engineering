from typing import Any
from typing import ClassVar

from fastapi import FastAPI
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from app.apps.base import BaseApp
from app.apps.lifespan import Lifespan
from app.apps.lifespan import worker_lifespan
from app.core.config import Settings
from app.core.config import settings
from app.exceptions.handlers import ExceptionHandler
from app.exceptions.handlers import InputMismatchHandler
from app.exceptions.handlers import NonFiniteOutputHandler
from app.exceptions.handlers import NotReadyHandler
from app.exceptions.handlers import OverloadedHandler
from app.inference.capacity.admission import AdmissionMiddleware
from app.inference.runtime.served import ServedModel
from app.observability.opentelemetry import OpenTelemetryInstrumentor
from app.observability.prometheus import PredictMetricsMiddleware
from app.routers.v1.predict import predict_router


class InferenceApp(BaseApp):
    """The shared app, plus POST /v1/predict for the served model, with its adapter's schemas.

    Around the predict route, outermost first: the body-size limit, the request metrics, then admission. Admission
    sits inside the metrics, so a request it refuses is still counted.
    """

    exception_handlers: ClassVar[list[type[ExceptionHandler[Any]]]] = [
        InputMismatchHandler,
        NonFiniteOutputHandler,
        NotReadyHandler,
        OverloadedHandler,
    ]

    def __init__(self, served: ServedModel, config: Settings) -> None:
        self.served = served
        super().__init__(config)

    def lifespan(self, instrumentor: OpenTelemetryInstrumentor, config: Settings) -> Lifespan:
        return worker_lifespan(instrumentor, config.model)

    def configure(self, config: Settings) -> None:
        self.app.state.served_model = self.served
        self.app.include_router(predict_router(self.served.adapter))

        predict = self.app.url_path_for("predict")
        self.app.add_middleware(
            AdmissionMiddleware, path=predict, limit=config.model.max_admitted, body_read_s=config.model.max_body_read_s
        )
        self.app.add_middleware(PredictMetricsMiddleware, path=predict)
        self.app.add_middleware(RequestBodyLimitMiddleware, max_body_size=config.model.max_body_bytes)


def create_app(served: ServedModel, config: Settings = settings) -> FastAPI:
    return InferenceApp(served, config).app
