from app.common.enums.logging import LogFormat
from app.core.config import settings
from app.core.logger import configure_logging
from app.core.logger import logger
from bench.config import bench_settings
from bench.preload.benchmark import run_benchmark
from bench.preload.benchmark import write

configure_logging(LogFormat.TEXT)
config = bench_settings.preload_benchmark

results = run_benchmark(settings.model.path, config)
logger.info(write(results, config.out))
