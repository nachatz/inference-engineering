import hashlib
from pathlib import Path
from typing import Any
from typing import Self

import msgspec

from app.core.config import ModelSettings
from app.inference.adapters.base import ModelAdapter
from app.inference.exceptions import ModelChecksumMismatch
from app.inference.runtime.info import ModelInfo
from app.inference.runtime.onnx import OnnxModel


def sha256_of(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class ServedModel(msgspec.Struct, frozen=True):
    """The process's model and its adapter, loaded and checked before the app is built and handed to it.

    With preload_app that's in the gunicorn master, so every worker shares it copy-on-write, and a model that can't be
    served stops gunicorn before it forks a worker. Never load one in the lifespan: that runs in each worker, after the
    fork, and costs a copy per worker.
    """

    model: OnnxModel
    adapter: ModelAdapter[Any, Any]
    info: ModelInfo
    path: Path

    @classmethod
    def load(cls, settings: ModelSettings) -> Self:
        sha256 = sha256_of(settings.path)
        if settings.sha256 is not None and sha256 != settings.sha256:
            raise ModelChecksumMismatch(settings.path, expected=settings.sha256, actual=sha256)

        model = OnnxModel.load(settings.path, settings.intra_op_threads)
        adapter = ModelAdapter.for_name(settings.adapter)
        adapter.check(model)
        model.warm_up(settings.warmup_runs)

        info = ModelInfo(
            name=settings.name,
            version=settings.version,
            sha256=sha256,
            size_bytes=settings.path.stat().st_size,
        )
        return cls(model=model, adapter=adapter, info=info, path=settings.path)
