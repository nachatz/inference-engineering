from pydantic import BaseModel


class ErrorDetail(BaseModel):
    location: list[str | int]
    message: str


class ErrorResponse(BaseModel):
    """Every error's body. `request_id` is the failed request's X-Request-ID: quoted by a client, it leads to its logs."""

    error: str
    details: list[ErrorDetail] = []
    request_id: str | None = None
