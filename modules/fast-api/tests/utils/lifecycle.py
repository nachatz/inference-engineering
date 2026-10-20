import time
from collections.abc import Callable

import httpx2
from prometheus_client.parser import text_string_to_metric_families


def eventually[T](check: Callable[[], T | None], timeout: float, interval: float = 0.05) -> tuple[T, float]:
    """Poll `check` until it returns something truthy: that, and the seconds it took. A connection error counts as not
    yet, since a server that is replacing workers refuses some."""
    start = time.monotonic()

    while True:
        try:
            if result := check():
                return result, time.monotonic() - start
        except (OSError, httpx2.HTTPError):
            pass

        if time.monotonic() - start > timeout:
            raise TimeoutError(f"not true after {timeout} s")
        time.sleep(interval)


def predict(url: str, features: list[float], timeout: float = 30) -> httpx2.Response:
    """One row, on a new connection: the kernel picks the worker's socket for each request."""
    return httpx2.post(f"{url}/v1/predict", json={"inputs": [features]}, timeout=timeout)


def metric(url: str, name: str, **labels: str) -> float:
    """The sum of the samples called `name` (with these labels) on /metrics. Any worker answers: in multiprocess mode
    each merges every worker's files."""
    text = httpx2.get(f"{url}/metrics/", timeout=5).text
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name == name and labels.items() <= sample.labels.items()
    )
