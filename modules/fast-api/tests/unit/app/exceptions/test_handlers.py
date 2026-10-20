import json
from collections.abc import Iterator
from http import HTTPStatus

import pytest
from asgi_correlation_id import correlation_id
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from app.exceptions.handlers import ExceptionHandler
from app.exceptions.handlers import HTTPExceptionHandler
from app.exceptions.handlers import InputMismatchHandler
from app.exceptions.handlers import NotReadyHandler
from app.exceptions.handlers import OverloadedHandler
from app.exceptions.handlers import RequestValidationErrorHandler
from app.exceptions.handlers import UnhandledExceptionHandler
from app.exceptions.models.error import ErrorDetail
from app.exceptions.models.error import ErrorResponse
from app.inference.exceptions import InputMismatch
from app.inference.exceptions import Overloaded
from app.inference.exceptions import WorkerNotStarted

REQUEST = Request({"type": "http"})


@pytest.fixture
def request_id() -> Iterator[str]:
    token = correlation_id.set("8f4b7e0c2a9d4f6e9b1c3d5e7f9a1b2c")
    yield "8f4b7e0c2a9d4f6e9b1c3d5e7f9a1b2c"
    correlation_id.reset(token)


class TestRequestValidationErrorHandler:
    def test_lists_each_error_with_its_location(self) -> None:
        exc = RequestValidationError([{"loc": ("body", "inputs"), "msg": "too short", "type": "too_short"}])
        assert RequestValidationErrorHandler().handle(exc) == ErrorResponse(
            error="invalid request", details=[ErrorDetail(location=["body", "inputs"], message="too short")]
        )


class TestInputMismatchHandler:
    def test_points_at_the_body(self) -> None:
        (detail,) = InputMismatchHandler().handle(InputMismatch("input 'x' takes float32 [batch, 8]")).details
        assert detail == ErrorDetail(location=["body"], message="input 'x' takes float32 [batch, 8]")


class TestNotReadyHandler:
    def test_explains_why(self) -> None:
        assert NotReadyHandler().handle(WorkerNotStarted()).error == "the worker hasn't started"


class TestOverloadedHandler:
    async def test_is_a_503_to_retry(self) -> None:
        response = await OverloadedHandler()(REQUEST, Overloaded("this worker's queue is full"))
        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert response.headers["Retry-After"] == "1"


class TestHTTPExceptionHandler:
    async def test_keeps_the_status_and_headers(self) -> None:
        exc = HTTPException(HTTPStatus.METHOD_NOT_ALLOWED, headers={"Allow": "POST"})
        response = await HTTPExceptionHandler()(REQUEST, exc)
        assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED
        assert response.headers["Allow"] == "POST"
        assert json.loads(bytes(response.body))["error"] == "Method Not Allowed"


class TestUnhandledExceptionHandler:
    async def test_hides_the_exception_behind_a_500(self) -> None:
        response = await UnhandledExceptionHandler()(REQUEST, RuntimeError("secret"))
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert json.loads(bytes(response.body)) == {"error": "internal server error", "details": [], "request_id": None}
        assert "X-Request-ID" not in response.headers

    async def test_carries_the_request_id(self, request_id: str) -> None:
        response = await UnhandledExceptionHandler()(REQUEST, RuntimeError("secret"))
        assert json.loads(bytes(response.body))["request_id"] == request_id
        assert response.headers["X-Request-ID"] == request_id


class TestHandledException:
    @pytest.mark.parametrize(
        ("handler", "exception"),
        [
            (OverloadedHandler, Overloaded),
            (InputMismatchHandler, InputMismatch),
            (HTTPExceptionHandler, HTTPException),
            (UnhandledExceptionHandler, Exception),
        ],
    )
    def test_is_the_type_parameter(
        self, handler: type[ExceptionHandler[Exception]], exception: type[Exception]
    ) -> None:
        assert handler.exception is exception

    def test_a_subclass_of_a_handler_keeps_its_exception(self) -> None:
        class RetryLater(OverloadedHandler):
            pass

        assert RetryLater.exception is Overloaded
