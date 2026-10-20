from http import HTTPStatus
from pathlib import Path

from starlette.exceptions import HTTPException


class NotReady(Exception): ...


class WorkerNotStarted(NotReady):
    def __init__(self) -> None:
        super().__init__("the worker hasn't started")


class Overloaded(Exception):
    """Refused rather than queued: the client should retry, ideally on another worker or pod."""


class BodyReadTimeout(HTTPException):
    """Raised from inside the body read, where FastAPI turns any other exception into a 400: an HTTPException it passes
    on, to the handler that gives every error the same shape."""

    def __init__(self, seconds: float) -> None:
        super().__init__(HTTPStatus.REQUEST_TIMEOUT, f"the request body didn't arrive within {seconds:g} s")


class InputMismatch(Exception):
    """The request's tensors don't fit the model's inputs: a 422, rather than an error from inside ONNX Runtime."""


class NonFiniteOutput(Exception):
    """The model computed a NaN or an infinity: a 500, never a prediction made from it."""

    def __init__(self, output: str) -> None:
        super().__init__(f"output {output!r} isn't finite")


class IncompatibleModel(Exception):
    """A model file this server, or its adapter, can't serve. Raised at startup, before any worker is forked."""


class ModelChecksumMismatch(Exception):
    def __init__(self, path: Path, expected: str, actual: str) -> None:
        super().__init__(f"{path} has sha256 {actual}, but MODEL_SHA256 pins {expected}")
