import json
import random
from typing import ClassVar

from locust import FastHttpUser
from locust import constant
from locust import task

from bench.config import LoadTestSettings
from bench.config import bench_settings


def payload(rows: int, n_features: int, raw: bool) -> str:
    """`raw` is the minimal example's bare list of features; otherwise the production `{"inputs": [[...]]}`."""
    batch = [[random.random() for _ in range(n_features)] for _ in range(rows)]
    return json.dumps(batch[0] if raw else {"inputs": batch})


class PredictUser(FastHttpUser):
    """A closed-loop client: sends a prediction, waits for the answer, sends the next. Without keep_alive, each
    request is a new connection, so the kernel spreads the requests over the workers' sockets."""

    wait_time = constant(0)
    config: ClassVar[LoadTestSettings] = bench_settings.load_test
    body: ClassVar[str] = payload(config.rows, config.features, config.raw)
    headers: ClassVar[dict[str, str]] = {"Content-Type": "application/json"} | (
        {} if config.keep_alive else {"Connection": "close"}
    )

    @task
    def predict(self) -> None:
        self.client.post(self.config.path, data=self.body, headers=self.headers)
