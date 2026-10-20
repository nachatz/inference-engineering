from app.common.enums.logging import LogFormat
from app.core.logger import configure_logging
from app.core.logger import logger
from bench.config import bench_settings
from bench.load.runner import run

configure_logging(LogFormat.TEXT)

logger.info(run(bench_settings.load_test))
