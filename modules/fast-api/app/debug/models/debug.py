from pathlib import Path

from pydantic import BaseModel

from app.common.enums.allocator import Allocator
from app.common.enums.environment import Environment
from app.common.enums.logging import LogFormat
from app.inference.runtime.info import ModelInfo


class FlagsResponse(BaseModel):
    environment: Environment
    log_level: str
    log_format: LogFormat
    preload: bool
    workers: int | None
    intra_op_threads: int
    max_batch: int
    max_queue: int
    max_queue_wait_s: float
    inference_stuck_s: float
    max_body_bytes: int
    warmup_runs: int
    otel_enabled: bool
    sentry_enabled: bool
    prometheus_multiprocess: bool


class AllocatorResponse(BaseModel):
    allocator: Allocator
    ld_preload: str | None
    malloc_conf: str | None


class Versions(BaseModel):
    python: str
    fastapi: str
    uvicorn: str
    gunicorn: str
    onnxruntime: str
    numpy: str


class RuntimeResponse(BaseModel):
    pid: int
    python_threads: int
    native_threads: int
    gc_frozen_objects: int
    versions: Versions


class TensorResponse(BaseModel):
    name: str
    dtype: str
    shape: list[int | str | None]


class ModelResponse(BaseModel):
    path: Path
    info: ModelInfo
    adapter: str
    inputs: list[TensorResponse]
    outputs: list[TensorResponse]
