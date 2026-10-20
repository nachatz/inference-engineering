import anyio
from starlette.requests import Request
from starlette.types import ASGIApp
from starlette.types import Message
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

from app.exceptions.handlers import OverloadedHandler
from app.inference.exceptions import BodyReadTimeout
from app.inference.exceptions import Overloaded
from app.observability.metrics import ADMITTED
from app.observability.metrics import SHED


class AdmissionMiddleware:
    """At most `limit` requests to one path in this worker at once.

    A request is counted from before its body is read until the app returns. For a request whose client left
    mid-inference that is when the inference ends, since the handler waits for it. One over the limit is refused with
    a 503 before any of its body is read, so refusing costs next to nothing.

    An admitted request's whole body is due `body_read_s` after admission: past that, the read raises BodyReadTimeout,
    a 408, and the place is freed. Once the body has arrived, later messages such as a disconnect have no deadline.
    """

    def __init__(self, app: ASGIApp, path: str, limit: int, body_read_s: float) -> None:
        self.app = app
        self.path = path
        self.limit = limit
        self.body_read_s = body_read_s
        self.admitted = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return

        if self.admitted >= self.limit:
            SHED.labels(reason="admission").inc()
            refused = await OverloadedHandler()(Request(scope), Overloaded("this worker is at its limit of requests"))
            await refused(scope, receive, send)
            return

        self.admitted += 1
        try:
            with ADMITTED.track_inprogress():
                await self.app(scope, self.by_deadline(receive), send)
        finally:
            self.admitted -= 1

    def by_deadline(self, receive: Receive) -> Receive:
        deadline = anyio.current_time() + self.body_read_s
        body_read = False

        async def receive_body() -> Message:
            nonlocal body_read
            if body_read:
                return await receive()

            try:
                with anyio.fail_after(deadline - anyio.current_time()):
                    message = await receive()
            except TimeoutError:
                raise BodyReadTimeout(self.body_read_s) from None

            body_read = message["type"] != "http.request" or not message.get("more_body", False)
            return message

        return receive_body
