import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import msgspec

from app.core.config import PrometheusSettings
from app.core.config import settings


class Server(msgspec.Struct, frozen=True):
    """A gunicorn started by `serve`: its master `process`, its `port`, and `started`, the time.perf_counter() it
    started at, which the readiness waits measure from."""

    process: subprocess.Popen[bytes]
    port: int
    started: float

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def ready_worker(url: str) -> int | None:
    """The pid of whichever worker answered /readyz, or None while none is ready."""
    try:
        with urllib.request.urlopen(f"{url}/readyz", timeout=5) as response:
            worker: int = json.load(response)["worker"]
            return worker
    except (OSError, urllib.error.URLError):
        return None


def wait_until_ready(server: Server, timeout: float = 120) -> float:
    """Seconds from starting gunicorn until the first worker is ready."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if ready_worker(server.url) is not None:
            return time.perf_counter() - server.started
        time.sleep(0.05)

    raise TimeoutError(f"{server.url} wasn't ready after {timeout}s")


def wait_for_all_workers(server: Server, workers: int, timeout: float = 120) -> float:
    """Seconds from starting gunicorn until every worker has answered /readyz at least once."""
    seen: set[int] = set()
    deadline = time.perf_counter() + timeout
    while len(seen) < workers:
        if time.perf_counter() > deadline:
            raise TimeoutError(f"only {len(seen)} of {workers} workers answered after {timeout}s")

        if (worker := ready_worker(server.url)) is not None:
            seen.add(worker)
        else:
            time.sleep(0.05)

    return time.perf_counter() - server.started


@contextmanager
def serve(
    model: Path,
    workers: int,
    preload: bool,
    log: Path | None = None,
    **model_changes: Any,
) -> Generator[Server]:
    """Run the production server (gunicorn.conf.py) as a subprocess with these settings, and any MODEL_* changes,
    e.g. `inference_stuck_s=1.0`. Linux only."""
    port = free_port()
    gunicorn = settings.gunicorn.model_copy(
        update={"host": "127.0.0.1", "port": port, "workers": workers, "preload": preload}
    )
    model_settings = settings.model.model_copy(update={"path": model.resolve(), **model_changes})
    metrics = PrometheusSettings(multiproc_dir=Path(tempfile.mkdtemp(prefix="fast-api-metrics-")))
    server_settings = settings.model_copy(update={"gunicorn": gunicorn, "model": model_settings, "prometheus": metrics})

    command = [sys.executable, "-m", "gunicorn", "-c", "gunicorn.conf.py"]
    root = Path(__file__).resolve().parents[2]

    with (log or Path(os.devnull)).open("w") as out:
        started = time.perf_counter()
        process = subprocess.Popen(command, cwd=root, env=server_settings.to_env(), stdout=out, stderr=out)
        try:
            yield Server(process=process, port=port, started=started)
        finally:
            process.terminate()
            process.wait(timeout=60)
