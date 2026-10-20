import importlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Protocol

from prometheus_client import multiprocess

from app.common.cpu import available_cpus
from app.common.cpu import cgroup_cpus
from app.core.config import settings
from app.core.logger import logging_config
from app.core.outputs import NoCpuLimit

if (metrics_dir := settings.prometheus.multiproc_dir) is not None:
    shutil.rmtree(metrics_dir, ignore_errors=True)
    metrics_dir.mkdir(parents=True)

# uvicorn and anyio import these only once a worker starts serving. Imported here, in the master, the workers share
# them instead of each loading (and on a cold start, compiling) a private copy.
for module in [
    "anyio._backends._asyncio",
    "uvicorn.lifespan.on",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
]:
    importlib.import_module(module)

wsgi_app = "app.main:app"
worker_class = "uvicorn_worker.UvicornWorker"
bind = f"{settings.gunicorn.host}:{settings.gunicorn.port}"
workers = settings.gunicorn.workers or available_cpus()

# Import app.main, and so load the model, once in the master and then fork: the workers share it copy-on-write.
preload_app = settings.gunicorn.preload
reuse_port = sys.platform == "linux"

timeout = settings.gunicorn.timeout
graceful_timeout = settings.gunicorn.graceful_timeout
keepalive = settings.gunicorn.keepalive
max_requests = settings.gunicorn.max_requests
max_requests_jitter = settings.gunicorn.max_requests_jitter
control_socket_disable = not settings.gunicorn.control_socket
worker_tmp_dir = "/dev/shm" if Path("/dev/shm").is_dir() else None
logconfig_dict = logging_config(settings.log_level, settings.log_format)


class Worker(Protocol):
    pid: int


class Arbiter(Protocol):
    log: logging.Logger


def on_starting(server: Arbiter) -> None:
    if settings.gunicorn.workers is None and cgroup_cpus() is None:
        server.log.warning(NoCpuLimit(workers=workers))


def child_exit(server: object, worker: Worker) -> None:
    if metrics_dir is not None:
        multiprocess.mark_process_dead(worker.pid, path=str(metrics_dir))  # type: ignore[no-untyped-call]  # unannotated
