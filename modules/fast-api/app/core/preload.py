import time

from app.core.config import ModelSettings
from app.core.config import settings
from app.core.logger import configure_logging
from app.core.logger import logger
from app.core.outputs import ModelLoaded
from app.inference.runtime.served import ServedModel
from app.observability.metrics import MODEL_INFO
from app.observability.sentry import report
from app.observability.sentry import start_sentry


def preload(model: ModelSettings | None = None) -> ServedModel:
    """Runs at import of app.main: with preload_app, once in the gunicorn master, before the workers are forked. A
    model that fails to load is reported to Sentry, if enabled, before gunicorn exits on it."""
    configure_logging()
    start_sentry(settings.sentry, settings.environment)

    start = time.perf_counter()
    try:
        served = ServedModel.load(model or settings.model)
    except Exception as error:
        report(error)
        raise
    seconds = round(time.perf_counter() - start, 2)

    MODEL_INFO.labels(name=served.info.name, version=served.info.version, sha256=served.info.sha256).set(1)
    logger.info(ModelLoaded(model=served.info, seconds=seconds))

    return served
