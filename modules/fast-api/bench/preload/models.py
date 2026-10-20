import statistics
from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel


class Metrics(BaseModel):
    """One run's numbers. `median` takes median_low of each, so it picks a real run's value and counts stay integers."""

    rps: float
    p50_ms: float
    p99_ms: float
    errors: int
    pss_total_mb: float
    rss_total_mb: float
    private_per_worker_mb: float
    first_response_s: float
    all_workers_ready_s: float

    @classmethod
    def median(cls, metrics: Sequence[Self]) -> Self:
        medians = {field: statistics.median_low(getattr(m, field) for m in metrics) for field in cls.model_fields}
        return cls.model_validate(medians)


class Summary(BaseModel):
    workers: int
    preload: bool
    repeats: int
    median: Metrics


class Machine(BaseModel):
    platform: str
    cpu: str
    cpus: int
    memory_gb: float
    python: str
    onnxruntime: str
    model: str
    model_mb: float
