from app.common.enums.logging import LogFormat
from app.core.logger import configure_logging
from app.core.logger import logger
from bench.common.proc import oldest_gunicorn
from bench.config import bench_settings
from bench.memory.pool import measure

configure_logging(LogFormat.TEXT)

logger.info(measure(bench_settings.mem.pid or oldest_gunicorn()))
