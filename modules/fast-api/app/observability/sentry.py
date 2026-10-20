import sentry_sdk
from asgi_correlation_id import correlation_id
from sentry_sdk.types import Event
from sentry_sdk.types import Hint

from app.common.enums.environment import Environment
from app.core.config import SentrySettings


def start_sentry(sentry: SentrySettings, environment: Environment) -> None:
    """Error reporting to Sentry, if enabled. preload starts it in the gunicorn master, before the model loads, so a
    failed load is reported too. That's fork-safe: the SDK sends from a background thread, and starts its own in each
    worker the first time it sends from there. Every event is tagged with its request's ID, which leads to its logs."""
    if not sentry.enabled:
        return

    sentry_sdk.init(
        dsn=sentry.dsn,
        environment=environment.value,
        release=sentry.release,
        traces_sample_rate=sentry.traces_sample_rate,
        send_default_pii=False,
        before_send=tag_request_id,
    )


def tag_request_id(event: Event, _: Hint) -> Event:
    if request_id := correlation_id.get():
        event.setdefault("tags", {})["request_id"] = request_id

    return event


def report(error: BaseException) -> None:
    """Send `error` now, before the process exits on it. Without a started client it does nothing."""
    sentry_sdk.capture_exception(error)
    sentry_sdk.flush()
