from pathlib import Path

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from app.common.settings import EnvSettings


class LoadTestSettings(EnvSettings):
    """One headless locust run: `LOAD_TEST_*`.

    - `raw`: send a bare list of features, as the minimal example in examples/ takes, instead of `{"inputs": [[...]]}`.
    - `keep_alive`: off, every request opens a new connection, so SO_REUSEPORT spreads them over the workers. On, each
      user keeps one connection, as a load balancer or an ingress does: the shape of production traffic.
    """

    model_config = SettingsConfigDict(env_prefix="LOAD_TEST_")

    host: str = "http://localhost:8000"
    path: str = "/v1/predict"
    users: int = 16
    seconds: int = 10
    rows: int = 1
    features: int = 1024
    raw: bool = False
    keep_alive: bool = False


class MemSettings(EnvSettings):
    """`MEM_PID`: the gunicorn master to measure; unset, the oldest gunicorn running."""

    model_config = SettingsConfigDict(env_prefix="MEM_")

    pid: int | None = None


class PreloadBenchmarkSettings(EnvSettings):
    model_config = SettingsConfigDict(env_prefix="BENCH_")

    workers: tuple[int, ...] = (1, 2, 4, 8)
    repeats: int = 3
    users: int = 16
    seconds: int = 10
    rows: int = 1
    keep_alive: bool = False
    out: Path = Path("bench/results")


class BenchSettings(EnvSettings):
    load_test: LoadTestSettings = Field(default_factory=LoadTestSettings)
    mem: MemSettings = Field(default_factory=MemSettings)
    preload_benchmark: PreloadBenchmarkSettings = Field(default_factory=PreloadBenchmarkSettings)


bench_settings = BenchSettings()
