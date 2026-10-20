import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path

import httpx2
import psutil
import pytest

from bench.common import server
from bench.common.proc import children
from bench.common.proc import cpu_seconds
from bench.common.proc import listening_sockets
from bench.common.units import MB
from bench.config import LoadTestSettings
from bench.load.runner import run
from bench.memory.pool import measure
from tests.utils import onnx_graphs
from tests.utils.lifecycle import eventually
from tests.utils.lifecycle import metric
from tests.utils.lifecycle import predict
from tests.utils.onnx_mlp import build_mlp

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "linux", reason="needs fork(), gunicorn and /proc"),
]

# The spinner's inference time is set by its input, inside ONNX Runtime: about 1.2 µs a loop on an x86 core.
QUICK = [0.5, 0.0, 0.0, 0.0]
ABOUT_A_SECOND = [1e6, 0.0, 0.0, 0.0]
FOREVER = [1e12, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # ~67 MB of float32 weights: big enough to tell a shared copy from one per worker, and slow enough that inference,
    # off the event loop, is most of a request. A model fast next to the HTTP work keeps the loop busy, and a busy loop
    # stops accepting connections: that would hide the imbalance TestBalance looks for.
    return build_mlp(tmp_path_factory.mktemp("model") / "model.onnx", d_in=16, d=4096, depth=2, n_classes=10)


@pytest.fixture(scope="module")
def spinner(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return onnx_graphs.spinner(tmp_path_factory.mktemp("spinner") / "spinner.onnx", d_in=4)


def replaced(running: server.Server, gone: set[int], workers: int) -> list[int] | None:
    """The pool, once gunicorn has replaced every worker in `gone` and is back to `workers`; else None."""
    pool = children(running.pid)
    return pool if len(pool) == workers and gone.isdisjoint(pool) else None


class TestPreload:
    """Real gunicorn servers, measured from /proc."""

    workers = 2

    def private_per_worker(self, model: Path, preload: bool, log: Path) -> list[int]:
        with server.serve(model, self.workers, preload, log) as running:
            server.wait_until_ready(running)
            server.wait_for_all_workers(running, self.workers)
            result = run(LoadTestSettings(host=running.url, users=4, seconds=2, features=16))
            pool = measure(running.pid)

        assert result.errors == 0
        assert result.requests
        assert len(pool.workers) == self.workers

        return [worker.private for worker in pool.workers]

    def test_workers_share_the_model(self, model: Path, tmp_path: Path) -> None:
        size = model.stat().st_size
        for private in self.private_per_worker(model, preload=True, log=tmp_path / "gunicorn.log"):
            assert private < 0.3 * size, f"{private / MB:.1f} MB private, the model is {size / MB:.1f} MB"

    def test_without_preload_each_worker_has_its_own_copy(self, model: Path, tmp_path: Path) -> None:
        size = model.stat().st_size
        for private in self.private_per_worker(model, preload=False, log=tmp_path / "gunicorn.log"):
            assert private > 0.9 * size, f"{private / MB:.1f} MB private, the model is {size / MB:.1f} MB"


class TestBalance:
    """the workers share the load, so 4 workers serve like 4 rather than 1."""

    workers = 4

    def test_each_worker_listens_on_its_own_socket(self, model: Path, tmp_path: Path) -> None:
        # The mechanism, checked directly: with one socket shared by every worker, whichever worker's event
        # loop is free accepts the connections. How badly that skews the load depends on timing a test can't control.
        with server.serve(model, self.workers, preload=True, log=tmp_path / "gunicorn.log") as running:
            server.wait_for_all_workers(running, self.workers)
            assert listening_sockets(running.port) == self.workers

    @pytest.mark.parametrize("keep_alive", [False, True], ids=["connection per request", "kept-alive connections"])
    def test_every_worker_does_its_share(self, model: Path, keep_alive: bool, tmp_path: Path) -> None:
        # The outcome. A worker's CPU time grows with the requests it serves, so its share of the pool's CPU time is its
        # share of the requests. Even is 25% each.
        with server.serve(model, self.workers, preload=True, log=tmp_path / "gunicorn.log") as running:
            server.wait_for_all_workers(running, self.workers)
            before = {pid: cpu_seconds(pid) for pid in children(running.pid)}

            # 32 kept-alive connections leave a worker without one about once in 2,500 runs.
            load = LoadTestSettings(host=running.url, users=32, seconds=4, features=16, keep_alive=keep_alive)
            result = run(load)
            used = {pid: cpu_seconds(pid) - seconds for pid, seconds in before.items()}

        assert result.errors == 0

        total = sum(used.values())
        for pid, seconds in used.items():
            assert seconds / total > 0.1, f"worker {pid} did {seconds / total:.0%} of the work: {used}"


class TestIncompatibleModel:
    def test_gunicorn_exits_before_forking_a_worker(self, tmp_path: Path) -> None:
        # A scorer's model, served with the default classifier adapter.
        scorer = onnx_graphs.scorer(tmp_path / "scorer.onnx", n_features=3)
        log = tmp_path / "gunicorn.log"

        with server.serve(scorer, workers=2, preload=True, log=log) as running:
            exit_code = running.process.wait(timeout=60)

        assert exit_code != 0
        assert (
            "IncompatibleModel: the adapter reads one input; the model has ['features', 'segment']" in log.read_text()
        )
        assert "Booting worker" not in log.read_text()


class TestStuckInference:
    stuck_after_s = 1.0

    def test_the_watchdog_replaces_the_worker_and_the_replacement_serves(self, spinner: Path, tmp_path: Path) -> None:
        log = tmp_path / "gunicorn.log"

        with (
            server.serve(spinner, workers=1, preload=True, log=log, inference_stuck_s=self.stuck_after_s) as running,
            ThreadPoolExecutor(1) as background,
        ):
            server.wait_until_ready(running)
            (stuck,) = children(running.pid)

            hung = background.submit(predict, running.url, FOREVER)
            eventually(lambda: metric(running.url, "predict_inflight") == 1, timeout=10)

            _, replaced_after = eventually(lambda: replaced(running, {stuck}, workers=1), timeout=30)
            _, serving_after = eventually(lambda: predict(running.url, QUICK).is_success, timeout=30)
            inflight = metric(running.url, "predict_inflight")

            # The stuck inference can't be cancelled: its client loses the connection when the worker exits.
            with pytest.raises(httpx2.HTTPError):
                hung.result(timeout=30)

        assert "exiting so gunicorn replaces it" in log.read_text()
        # Stuck after 1 s, noticed within a watchdog interval (0.25 s), replaced, then serving: seconds, not minutes.
        assert replaced_after < self.stuck_after_s + 5
        assert replaced_after + serving_after < self.stuck_after_s + 10
        # The dead worker's in-flight inference is no longer counted.
        assert inflight == 0


class TestClientCancellation:
    max_queue = 2

    def abandon(self, url: str) -> int | None:
        """Ask for a second of inference and hang up after 0.2 s: the status if refused by then, else None."""
        try:
            return predict(url, ABOUT_A_SECOND, timeout=0.2).status_code
        except httpx2.TimeoutException:
            return None

    def test_abandoned_requests_dont_leak_capacity(self, spinner: Path, tmp_path: Path) -> None:
        log = tmp_path / "gunicorn.log"

        with server.serve(spinner, 1, preload=True, log=log, max_queue=self.max_queue, max_queue_wait_s=30) as running:
            server.wait_until_ready(running)

            # Six at once at one worker: one runs, two queue, three are refused. Then every client hangs up.
            with ThreadPoolExecutor(6) as clients:
                outcomes = list(clients.map(lambda _: self.abandon(running.url), range(6)))

            def drained() -> bool:
                return metric(running.url, "predict_queue_depth") == 0 and metric(running.url, "predict_inflight") == 0

            _, drained_after = eventually(drained, timeout=30)

            # A full queue's worth again, all answered: no slot or queue place was lost to a client that left.
            with ThreadPoolExecutor(self.max_queue + 1) as clients:
                after = list(clients.map(lambda _: predict(running.url, QUICK).status_code, range(self.max_queue + 1)))
            shed = metric(running.url, "predict_shed_total", reason="queue_full")

        assert outcomes.count(HTTPStatus.SERVICE_UNAVAILABLE) == shed == 3
        assert outcomes.count(None) == 1 + self.max_queue
        # The three accepted inferences still run, one after another, after their clients left: about 3.6 s.
        assert drained_after < 3 * 1.2 + 5
        assert after == [HTTPStatus.OK] * (self.max_queue + 1)


class TestWorkerDeath:
    workers = 3

    def test_the_others_keep_serving_and_capacity_recovers(self, model: Path, tmp_path: Path) -> None:
        outcomes: list[tuple[float, bool]] = []
        stop = threading.Event()

        def traffic(url: str) -> None:
            while not stop.is_set():
                sent = time.monotonic()
                try:
                    ok = predict(url, [0.5] * 16, timeout=10).is_success
                except httpx2.HTTPError:
                    ok = False
                outcomes.append((sent, ok))

        def recovered(running: server.Server, victim: int) -> bool:
            return bool(replaced(running, {victim}, self.workers)) and listening_sockets(running.port) == self.workers

        with (
            server.serve(model, self.workers, preload=True, log=tmp_path / "gunicorn.log") as running,
            ThreadPoolExecutor(4) as clients,
        ):
            server.wait_for_all_workers(running, self.workers)
            for _ in range(4):
                clients.submit(traffic, running.url)

            time.sleep(1)
            victim = children(running.pid)[0]
            killed = time.monotonic()
            psutil.Process(victim).kill()

            _, recovered_after = eventually(lambda: recovered(running, victim), timeout=30)
            time.sleep(2)
            stop.set()

        while_down = [ok for sent, ok in outcomes if killed <= sent < killed + recovered_after]
        # Sent once the replacement was listening, with a margin for the requests then in flight.
        after = [ok for sent, ok in outcomes if sent > killed + recovered_after + 0.5]

        assert any(ok for sent, ok in outcomes if sent < killed)
        # Only requests the dead worker held, or its socket had queued, fail: the other workers keep answering.
        assert any(while_down), "no request succeeded while the worker was down"
        assert recovered_after < 5
        assert after
        assert all(after), f"{after.count(False)} of {len(after)} requests failed after the pool recovered"


class TestMetricsAfterReplacement:
    workers = 2

    def test_the_metrics_count_the_live_workers(self, model: Path, tmp_path: Path) -> None:
        with server.serve(model, self.workers, preload=True, log=tmp_path / "gunicorn.log") as running:
            server.wait_for_all_workers(running, self.workers)
            for _ in range(10):
                assert predict(running.url, [0.5] * 16).is_success

            requests_before = metric(running.url, "predict_requests_total")
            serving_before = metric(running.url, "workers_serving")

            victim, survivor = children(running.pid)
            psutil.Process(victim).kill()
            pool, _ = eventually(lambda: replaced(running, {victim}, self.workers), timeout=30)
            (new,) = set(pool) - {survivor}

            # Its lifespan has run, so its gauge is set, once the new worker has answered.
            eventually(lambda: server.ready_worker(running.url) == new, timeout=30)
            serving_after = metric(running.url, "workers_serving")
            requests_after = metric(running.url, "predict_requests_total")

        assert serving_before == self.workers
        # 2, not 3: gunicorn's child_exit hook removed the dead worker's gauge file. Without it the dead worker would
        # still count, and a live gauge like predict_inflight would keep whatever it last reported.
        assert serving_after == self.workers
        # Counters keep the dead worker's requests: the total never goes down when a worker is replaced.
        assert requests_after >= requests_before >= 10


class TestOverload:
    """One worker, at most 4 requests admitted: flooded with large bodies, and with clients that never finish theirs,
    while health checks run throughout."""

    max_admitted = 4
    max_body_read_s = 1.0
    seconds = 6.0

    # About 1.3 MB of JSON. Read, parsed and converted before the spinner refuses it, since it takes 4 features: the
    # memory and the event-loop time a request costs before it reaches the queue.
    body = json.dumps({"inputs": [[0.123456789] * 100_000]}).encode()

    def slow_loris(self, port: int) -> tuple[int, float]:
        """Headers, a promised 100 kB of body, one byte of it, then nothing: the status the server answers, and when."""
        start = time.monotonic()
        with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
            connection.sendall(
                b"POST /v1/predict HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
                b"Content-Length: 100000\r\n\r\n{"
            )
            status = int(connection.recv(1024).split()[1])

        return status, time.monotonic() - start

    def test_memory_and_latency_stay_bounded_and_the_worker_recovers(self, spinner: Path, tmp_path: Path) -> None:
        stop = threading.Event()
        floods: list[tuple[int, float]] = []
        health: list[float] = []
        rss: list[int] = []

        def flood(url: str) -> None:
            while not stop.is_set():
                sent = time.monotonic()
                try:
                    status = httpx2.post(
                        f"{url}/v1/predict", content=self.body, headers={"Content-Type": "application/json"}, timeout=30
                    ).status_code
                except httpx2.HTTPError:
                    status = 0
                floods.append((status, time.monotonic() - sent))

        def check_health(url: str) -> None:
            while not stop.is_set():
                sent = time.monotonic()
                assert httpx2.get(f"{url}/healthz", timeout=30).is_success
                health.append(time.monotonic() - sent)
                time.sleep(0.05)

        def sample(worker: psutil.Process) -> None:
            while not stop.is_set():
                rss.append(worker.memory_info().rss)
                time.sleep(0.02)

        log = tmp_path / "gunicorn.log"
        limits = {"max_admitted": self.max_admitted, "max_queue": 2, "max_body_read_s": self.max_body_read_s}

        with server.serve(spinner, 1, preload=True, log=log, **limits) as running:
            server.wait_until_ready(running)
            (pid,) = children(running.pid)
            worker = psutil.Process(pid)
            baseline = worker.memory_info().rss

            pool = ThreadPoolExecutor(40)
            watchers = [pool.submit(sample, worker), pool.submit(check_health, running.url)]
            # The slow clients first: admitted, they hold every place until their deadline, then give them back.
            lorises = [pool.submit(self.slow_loris, running.port) for _ in range(self.max_admitted)]
            time.sleep(0.2)
            for _ in range(24):
                pool.submit(flood, running.url)

            time.sleep(self.seconds)
            stop.set()
            pool.shutdown(wait=True)
            for watcher in watchers:
                watcher.result()

            def recovered() -> bool:
                return metric(running.url, "predict_admitted") == 0 and predict(running.url, QUICK).is_success

            _, recovered_after = eventually(recovered, timeout=10)

        statuses = {status for status, _ in floods}
        refusals = [seconds for status, seconds in floods if status == HTTPStatus.SERVICE_UNAVAILABLE]
        growth = max(rss) - baseline

        measured = (
            f"{len(floods)} flood requests {sorted(statuses)}, {len(refusals)} refused, slowest refusal "
            f"{max(refusals, default=0):.3f} s; slowest health check {max(health):.3f} s of {len(health)}; "
            f"RSS +{growth / MB:.1f} MB; lorises {[loris.result() for loris in lorises]}; "
            f"recovered in {recovered_after:.2f} s"
        )

        # The flood is answered, never dropped: a 422 once read and refused by the model, or a 503 at admission.
        assert statuses <= {HTTPStatus.UNPROCESSABLE_ENTITY, HTTPStatus.SERVICE_UNAVAILABLE}, measured
        assert HTTPStatus.SERVICE_UNAVAILABLE in statuses, measured
        # Refused before the body is read: fast however much is waiting.
        assert max(refusals) < 1.0, measured
        # At most 4 requests' bodies, parsed JSON and tensors at once: a few times the body each.
        assert growth < self.max_admitted * 10 * len(self.body), measured
        # The event loop never waits on inference, and parses at most 4 bodies at a time.
        assert max(health) < 1.0, measured
        # A body that never comes costs its client a 408 at the deadline, and its place goes back to the flood.
        for status, seconds in (loris.result() for loris in lorises):
            assert status == HTTPStatus.REQUEST_TIMEOUT, measured
            assert self.max_body_read_s <= seconds < self.max_body_read_s + 1, measured
        assert HTTPStatus.UNPROCESSABLE_ENTITY in statuses, measured
        assert recovered_after < 5, measured
