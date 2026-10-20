from pathlib import Path
from typing import Self

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from app.common.enums.adapter import AdapterName
from app.common.enums.environment import Environment
from app.common.enums.logging import LogFormat
from app.common.settings import EnvSettings


class GunicornSettings(EnvSettings):
    """How gunicorn runs the app: `GUNICORN_*`, read by gunicorn.conf.py.

    - `workers`: unset, one per CPU of the container's quota.
    - `preload`: load the model once in the master and fork, so every worker shares it.
    - `keepalive`: above the load balancer's idle timeout, so it never reuses a connection gunicorn already closed.
    - `control_socket`: gunicorn's runtime control socket lives under $HOME, which the image's user doesn't have.
    """

    model_config = SettingsConfigDict(env_prefix="GUNICORN_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int | None = Field(default=None, ge=1)
    preload: bool = True
    timeout: int = 30
    graceful_timeout: int = 20
    keepalive: int = 75
    max_requests: int = 0
    max_requests_jitter: int = 0
    control_socket: bool = False


class ModelSettings(EnvSettings):
    """The served model, and the limits every request to it meets: `MODEL_*`.

    The artifact
        - `path`, and the `adapter` that turns requests into its inputs and its outputs into responses.
        - `name` and `version` identify it at startup, on /readyz and as the model_info metric.
        - `sha256` pins the file at deploy time: a different file refuses to load. Unset, the hash is only reported.

    Inference
        - `intra_op_threads` is 1: the gunicorn workers are the parallelism, one core each.
        - `max_batch` rows per request; `warmup_runs` before the fork.

    Limits, in the order a request meets them
        - `max_body_bytes`: a larger body is a 413 before it is read. 64 rows of 1024 floats are about 1.4 MB of JSON.
        - `max_admitted`: requests a worker holds at once, from before the first byte of the body to the response.
          Any more get a 503 without their body being read. 40 is the one running, `max_queue` waiting and 7 being
          read; with the body limit it bounds a worker's request memory at about 6 x `max_body_bytes` each for JSON
          floats, as TestOverload measures.
        - `max_body_read_s`: an admitted request's body must arrive within this, or it is a 408.
        - `max_queue`, `max_queue_wait_s`: requests that may wait for the worker's one inference, and for how long,
          before a 503 the client can retry instead of an answer it has stopped waiting for.
        - `inference_stuck_s`: an inference running longer is stuck; the worker exits and gunicorn replaces it.
    """

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    path: Path = Path("model/model.onnx")
    adapter: AdapterName = AdapterName.CLASSIFIER
    name: str = "classifier"
    version: str = "dev"
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    intra_op_threads: int = Field(default=1, ge=1)
    max_batch: int = Field(default=64, ge=1)
    warmup_runs: int = Field(default=1, ge=0)

    max_body_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    max_admitted: int = Field(default=40, ge=1)
    max_body_read_s: float = Field(default=5.0, gt=0)
    max_queue: int = Field(default=32, ge=0)
    max_queue_wait_s: float = Field(default=2.0, gt=0)
    inference_stuck_s: float = Field(default=30.0, gt=0)


class OpenTelemetrySettings(EnvSettings):
    """Request tracing over OTLP/HTTP: `OTEL_*`. Off unless `enabled`."""

    model_config = SettingsConfigDict(env_prefix="OTEL_")

    enabled: bool = False
    service_name: str = "fast-api"
    exporter_otlp_endpoint: str = "http://localhost:4318"
    sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    excluded_urls: str = "healthz,readyz,metrics,debug"


class SentrySettings(EnvSettings):
    """Error reporting to Sentry: `SENTRY_*`. Off unless `enabled`, which needs a `dsn`.

    Errors only by default: `traces_sample_rate` is 0, since OpenTelemetry does the tracing. Raise it to also send
    Sentry performance traces. `release` tags every event, e.g. with the image's version.
    """

    model_config = SettingsConfigDict(env_prefix="SENTRY_")

    enabled: bool = False
    dsn: str | None = None
    release: str | None = None
    traces_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_dsn(self) -> Self:
        if self.enabled and not self.dsn:
            raise ValueError("SENTRY_ENABLED needs SENTRY_DSN")

        return self


class PrometheusSettings(EnvSettings):
    """`PROMETHEUS_MULTIPROC_DIR`, where every worker writes its metrics for any of them to serve merged.

    prometheus_client reads the variable itself when it is first imported, so it has to be in the environment gunicorn
    starts in: the Dockerfile and `just serve` set it. Unset, each worker reports only itself.
    """

    model_config = SettingsConfigDict(env_prefix="PROMETHEUS_")

    multiproc_dir: Path | None = None


class AllocatorSettings(EnvSettings):
    """`LD_PRELOAD` and `MALLOC_CONF`, read only to report them: the loader and jemalloc read them before Python."""

    ld_preload: str | None = None
    malloc_conf: str | None = None


class Settings(EnvSettings):
    """Every setting the server reads, one nested group per environment prefix.

    Refuses `MODEL_INTRA_OP_THREADS > 1` with preload on: ONNX Runtime's thread pool would be created in the gunicorn
    master, and its threads don't survive the fork. Better a startup error than a hang on the first request.
    """

    app_title: str = "fast-api"
    environment: Environment = Environment.LOCAL
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON

    gunicorn: GunicornSettings = Field(default_factory=GunicornSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    otel: OpenTelemetrySettings = Field(default_factory=OpenTelemetrySettings)
    sentry: SentrySettings = Field(default_factory=SentrySettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    allocator: AllocatorSettings = Field(default_factory=AllocatorSettings)

    @model_validator(mode="after")
    def check_fork_safety(self) -> Self:
        if self.gunicorn.preload and self.model.intra_op_threads > 1:
            raise ValueError(
                "MODEL_INTRA_OP_THREADS > 1 needs GUNICORN_PRELOAD=0: ONNX Runtime's thread pool would be created "
                "in the gunicorn master, and its threads don't survive the fork"
            )

        return self


settings = Settings()
