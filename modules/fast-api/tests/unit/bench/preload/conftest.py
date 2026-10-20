import pytest

from bench.preload.models import Metrics


@pytest.fixture
def metrics() -> Metrics:
    return Metrics(
        rps=302.8,
        p50_ms=42,
        p99_ms=140,
        errors=0,
        pss_total_mb=502,
        rss_total_mb=2190,
        private_per_worker_mb=14.4,
        first_response_s=0.61,
        all_workers_ready_s=0.77,
    )
