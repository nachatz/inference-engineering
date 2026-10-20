import gc
import platform
import threading
from importlib.metadata import version

import psutil

from app.common.enums.allocator import Allocator
from app.common.memory import ProcessMemory
from app.core.config import Settings
from app.debug.exceptions import ProcUnavailable
from app.debug.models.debug import AllocatorResponse
from app.debug.models.debug import FlagsResponse
from app.debug.models.debug import ModelResponse
from app.debug.models.debug import RuntimeResponse
from app.debug.models.debug import TensorResponse
from app.debug.models.debug import Versions
from app.inference.runtime.served import ServedModel
from app.inference.runtime.tensor import TensorSpec


def tensor(spec: TensorSpec) -> TensorResponse:
    return TensorResponse(name=spec.name, dtype=str(spec.dtype), shape=list(spec.shape))


class DebugHandler:
    """What this worker is actually running with, including what its settings can't vouch for.

    The allocator is the one mapped into the process, since LD_PRELOAD only asks for it; with none of jemalloc,
    tcmalloc or mimalloc mapped on Linux it is glibc's malloc. Memory needs Linux's /proc, and is a 501 elsewhere.
    """

    def __init__(self, served: ServedModel, config: Settings, process: psutil.Process | None = None) -> None:
        self._served = served
        self._config = config
        self._process = process or psutil.Process()

    def flags(self) -> FlagsResponse:
        return FlagsResponse(
            environment=self._config.environment,
            log_level=self._config.log_level,
            log_format=self._config.log_format,
            preload=self._config.gunicorn.preload,
            workers=self._config.gunicorn.workers,
            intra_op_threads=self._config.model.intra_op_threads,
            max_batch=self._config.model.max_batch,
            max_queue=self._config.model.max_queue,
            max_queue_wait_s=self._config.model.max_queue_wait_s,
            inference_stuck_s=self._config.model.inference_stuck_s,
            max_body_bytes=self._config.model.max_body_bytes,
            warmup_runs=self._config.model.warmup_runs,
            otel_enabled=self._config.otel.enabled,
            sentry_enabled=self._config.sentry.enabled,
            prometheus_multiprocess=self._config.prometheus.multiproc_dir is not None,
        )

    def allocator(self) -> AllocatorResponse:
        return AllocatorResponse(
            allocator=self._loaded_allocator(),
            ld_preload=self._config.allocator.ld_preload,
            malloc_conf=self._config.allocator.malloc_conf,
        )

    def memory(self) -> ProcessMemory:
        if not psutil.LINUX:
            raise ProcUnavailable

        return ProcessMemory.of(self._process.pid)

    def runtime(self) -> RuntimeResponse:
        return RuntimeResponse(
            pid=self._process.pid,
            python_threads=threading.active_count(),
            native_threads=self._process.num_threads(),
            gc_frozen_objects=gc.get_freeze_count(),
            versions=Versions(
                python=platform.python_version(),
                fastapi=version("fastapi"),
                uvicorn=version("uvicorn"),
                gunicorn=version("gunicorn"),
                onnxruntime=version("onnxruntime"),
                numpy=version("numpy"),
            ),
        )

    def model(self) -> ModelResponse:
        model = self._served.model

        return ModelResponse(
            path=self._served.path,
            info=self._served.info,
            adapter=type(self._served.adapter).__name__,
            inputs=[tensor(spec) for spec in model.inputs.values()],
            outputs=[tensor(spec) for spec in model.outputs.values()],
        )

    def _loaded_allocator(self) -> Allocator:
        if not psutil.LINUX:
            return Allocator.UNKNOWN

        libraries = {
            Allocator.JEMALLOC: "libjemalloc",
            Allocator.TCMALLOC: "libtcmalloc",
            Allocator.MIMALLOC: "libmimalloc",
        }
        mapped = " ".join(region.path for region in self._process.memory_maps())
        loaded = (allocator for allocator, library in libraries.items() if library in mapped)
        return next(loaded, Allocator.GLIBC)
