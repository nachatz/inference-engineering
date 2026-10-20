from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from http import HTTPStatus
from types import get_original_bases
from typing import Any
from typing import ClassVar
from typing import get_args
from typing import get_origin

from asgi_correlation_id import correlation_id
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.debug.exceptions import ProcUnavailable
from app.exceptions.models.error import ErrorDetail
from app.exceptions.models.error import ErrorResponse
from app.inference.exceptions import InputMismatch
from app.inference.exceptions import NonFiniteOutput
from app.inference.exceptions import NotReady
from app.inference.exceptions import Overloaded


class ExceptionHandler[E: Exception](ABC):
    """Turns one exception type into an ErrorResponse, stamped with the request ID.

    A subclass names the exception it handles once, as its type parameter: `class XHandler(ExceptionHandler[X])` sets
    `XHandler.exception` to X, which is what an app registers it for, so an app lists handler classes, not pairs. A
    subclass of a handler keeps the exception it inherits.
    """

    exception: ClassVar[type[Exception]]
    status_code: ClassVar[HTTPStatus]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if handled := [get_args(base)[0] for base in get_original_bases(cls) if get_origin(base) is ExceptionHandler]:
            (cls.exception,) = handled

    @abstractmethod
    def handle(self, exc: E) -> ErrorResponse: ...

    def status(self, exc: E) -> int:
        return self.status_code

    def headers(self, exc: E) -> Mapping[str, str] | None:
        return None

    async def __call__(self, _: Request, exc: E) -> JSONResponse:
        error = self.handle(exc).model_copy(update={"request_id": correlation_id.get()})

        return JSONResponse(
            status_code=self.status(exc),
            content=error.model_dump(),
            headers=self.headers(exc),
        )


class RequestValidationErrorHandler(ExceptionHandler[RequestValidationError]):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY

    def handle(self, exc: RequestValidationError) -> ErrorResponse:
        details = [ErrorDetail(location=list(error["loc"]), message=error["msg"]) for error in exc.errors()]
        return ErrorResponse(error="invalid request", details=details)


class InputMismatchHandler(ExceptionHandler[InputMismatch]):
    status_code = HTTPStatus.UNPROCESSABLE_ENTITY

    def handle(self, exc: InputMismatch) -> ErrorResponse:
        return ErrorResponse(error="invalid request", details=[ErrorDetail(location=["body"], message=str(exc))])


class HTTPExceptionHandler(ExceptionHandler[HTTPException]):
    """Starlette's own errors (404, 405, 413, ...) in the same shape as the rest, with their own status."""

    def status(self, exc: HTTPException) -> int:
        return exc.status_code

    def headers(self, exc: HTTPException) -> Mapping[str, str] | None:
        return exc.headers

    def handle(self, exc: HTTPException) -> ErrorResponse:
        return ErrorResponse(error=exc.detail)


class NotReadyHandler(ExceptionHandler[NotReady]):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE

    def handle(self, exc: NotReady) -> ErrorResponse:
        return ErrorResponse(error=str(exc))


class OverloadedHandler(ExceptionHandler[Overloaded]):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE

    def headers(self, exc: Overloaded) -> Mapping[str, str] | None:
        return {"Retry-After": "1"}

    def handle(self, exc: Overloaded) -> ErrorResponse:
        return ErrorResponse(error=str(exc))


class NonFiniteOutputHandler(ExceptionHandler[NonFiniteOutput]):
    """The model failed, not the request: a 500, which says why rather than hiding it behind a label."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR

    def handle(self, exc: NonFiniteOutput) -> ErrorResponse:
        return ErrorResponse(error=f"the model computed a non-finite result: {exc}")


class ProcUnavailableHandler(ExceptionHandler[ProcUnavailable]):
    status_code = HTTPStatus.NOT_IMPLEMENTED

    def handle(self, exc: ProcUnavailable) -> ErrorResponse:
        return ErrorResponse(error=str(exc))


class UnhandledExceptionHandler(ExceptionHandler[Exception]):
    """The 500. Starlette sends it from outside every middleware, so it sets X-Request-ID itself."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR

    def headers(self, exc: Exception) -> Mapping[str, str] | None:
        request_id = correlation_id.get()
        return {"X-Request-ID": request_id} if request_id else None

    def handle(self, _: Exception) -> ErrorResponse:
        return ErrorResponse(error="internal server error")
