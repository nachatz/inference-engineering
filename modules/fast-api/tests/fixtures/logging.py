import logging
from collections.abc import Iterator

import pytest

from app.core.logger import LOGGERS


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """configure_logging() swaps handlers process-wide. Put them back, so no test logs into another's closed stream."""
    loggers = [logging.getLogger(name) for name in ["", *LOGGERS]]
    saved = [(logger, logger.handlers[:], logger.level, logger.propagate) for logger in loggers]
    yield
    for logger, handlers, level, propagate in saved:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
