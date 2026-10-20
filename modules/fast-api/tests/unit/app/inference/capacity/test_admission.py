import json
from http import HTTPStatus

import anyio
import pytest
from prometheus_client import REGISTRY
from starlette.types import Message
from starlette.types import Receive
from starlette.types import Scope
from starlette.types import Send

from app.inference.capacity.admission import AdmissionMiddleware
from app.inference.exceptions import BodyReadTimeout

PATH = "/v1/predict"
HTTP: Scope = {"type": "http", "path": PATH, "method": "POST", "headers": []}


def receiving(*messages: Message, every_s: float = 0.0) -> Receive:
    """A client sending `messages`, one each `every_s`, then nothing more."""
    pending = list(messages)

    async def receive() -> Message:
        if not pending:
            await anyio.sleep_forever()
        await anyio.sleep(every_s)
        return pending.pop(0)

    return receive


def body(*chunks: bytes) -> list[Message]:
    last = len(chunks) - 1
    return [{"type": "http.request", "body": chunk, "more_body": i < last} for i, chunk in enumerate(chunks)]


async def unread() -> Message:
    raise AssertionError("the body was read")


async def unsent(_: Message) -> None:
    raise AssertionError("a response was sent")


async def read_body(_: Scope, receive: Receive, send: Send) -> None:
    while (await receive()).get("more_body"):
        pass
    await send({"type": "http.response.start", "status": HTTPStatus.OK, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def shed() -> float:
    return REGISTRY.get_sample_value("predict_shed_total", {"reason": "admission"}) or 0.0


class TestAdmission:
    async def test_other_paths_and_scopes_pass_through(self) -> None:
        seen: list[Scope] = []

        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            seen.append(scope)

        admission = AdmissionMiddleware(app, PATH, limit=1, body_read_s=0.01)
        await admission({"type": "lifespan"}, unread, unsent)
        await admission({**HTTP, "path": "/healthz"}, unread, unsent)

        assert [scope["type"] for scope in seen] == ["lifespan", "http"]

    async def test_refuses_beyond_the_limit_without_reading_the_body(self) -> None:
        entered, release = anyio.Event(), anyio.Event()

        async def held(scope: Scope, receive: Receive, send: Send) -> None:
            entered.set()
            await release.wait()

        admission = AdmissionMiddleware(held, PATH, limit=1, body_read_s=1.0)
        sent: list[Message] = []
        before = shed()

        async def record(message: Message) -> None:
            sent.append(message)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(admission, HTTP, receiving(*body(b"{}")), record)
            await entered.wait()

            await admission(HTTP, unread, record)
            release.set()

        start, response = sent
        assert start["status"] == HTTPStatus.SERVICE_UNAVAILABLE
        assert (b"retry-after", b"1") in start["headers"]
        assert json.loads(response["body"])["error"] == "this worker is at its limit of requests"
        assert shed() == before + 1
        assert admission.admitted == 0

    async def test_gives_the_place_back_when_the_app_raises(self) -> None:
        async def crash(scope: Scope, receive: Receive, send: Send) -> None:
            raise RuntimeError

        admission = AdmissionMiddleware(crash, PATH, limit=1, body_read_s=1.0)
        with pytest.raises(RuntimeError):
            await admission(HTTP, receiving(), unsent)

        assert admission.admitted == 0

    async def test_a_body_that_doesnt_arrive_is_a_408(self) -> None:
        admission = AdmissionMiddleware(read_body, PATH, limit=1, body_read_s=0.05)

        with anyio.fail_after(5), pytest.raises(BodyReadTimeout, match="didn't arrive within 0.05 s"):
            await admission(HTTP, receiving(), unsent)

        assert admission.admitted == 0

    async def test_one_deadline_covers_every_piece_of_the_body(self) -> None:
        # Each piece comes within 0.05 s of the last, but the whole body doesn't come within 0.05 s.
        admission = AdmissionMiddleware(read_body, PATH, limit=1, body_read_s=0.05)

        with pytest.raises(BodyReadTimeout):
            await admission(HTTP, receiving(*body(b"{", b'"a"', b": 1}"), every_s=0.03), unsent)

    async def test_after_the_body_receive_has_no_deadline(self) -> None:
        late: list[Message] = []

        async def wait_for_disconnect(scope: Scope, receive: Receive, send: Send) -> None:
            await receive()
            late.append(await receive())

        admission = AdmissionMiddleware(wait_for_disconnect, PATH, limit=1, body_read_s=0.05)
        disconnect: Message = {"type": "http.disconnect"}
        messages = [*body(b"{}"), disconnect]
        await admission(HTTP, receiving(*messages, every_s=0.04), unsent)

        assert late == [disconnect]
