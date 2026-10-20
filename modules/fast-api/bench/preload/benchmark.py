import statistics
from pathlib import Path

from app.core.logger import logger
from app.inference.runtime.onnx import OnnxModel
from bench.common import server
from bench.common.units import MB
from bench.config import LoadTestSettings
from bench.config import PreloadBenchmarkSettings
from bench.load.runner import run
from bench.memory.pool import measure as measure_memory
from bench.preload.machine import machine
from bench.preload.models import Metrics
from bench.preload.models import Summary
from bench.preload.outputs import ResultsWritten
from bench.preload.outputs import Run
from bench.preload.report import Results


def measure(
    model: Path, n_features: int, workers: int, preload: bool, config: PreloadBenchmarkSettings, log: Path
) -> Metrics:
    """Start gunicorn, wait for every worker, warm up, load test, then read every process's memory."""
    with server.serve(model, workers, preload, log) as running:
        first_response = server.wait_until_ready(running)
        all_workers_ready = server.wait_for_all_workers(running, workers)

        load = LoadTestSettings(
            host=running.url,
            users=config.users,
            seconds=config.seconds,
            rows=config.rows,
            features=n_features,
            keep_alive=config.keep_alive,
        )
        run(load.model_copy(update={"users": workers * 2, "seconds": 2}))
        result = run(load)

        pool = measure_memory(running.pid)

    return Metrics(
        rps=round(result.rps, 1),
        p50_ms=result.p50_ms,
        p99_ms=result.p99_ms,
        errors=result.errors,
        pss_total_mb=round(pool.pss / MB, 1),
        rss_total_mb=round(pool.rss / MB, 1),
        private_per_worker_mb=round(statistics.mean(w.private for w in pool.workers) / MB, 1),
        first_response_s=round(first_response, 2),
        all_workers_ready_s=round(all_workers_ready, 2),
    )


def run_benchmark(model: Path, config: PreloadBenchmarkSettings) -> Results:
    """Every worker count in `config`, with and without preload, `config.repeats` times; each run logged as it ends.
    The load test sends rows of the model's feature count, so the model needs one input with a fixed one."""
    (spec,) = OnnxModel.load(model, intra_op_threads=1).inputs.values()
    if not isinstance(n_features := spec.shape[-1], int):
        raise TypeError(f"the benchmark needs a fixed feature count, got {spec}")

    (config.out / "logs").mkdir(parents=True, exist_ok=True)

    runs: list[Run] = []
    summaries: list[Summary] = []
    for workers in config.workers:
        for preload in (False, True):
            repeats = []
            for repeat in range(config.repeats):
                log = config.out / "logs" / f"workers{workers}-preload{int(preload)}-{repeat}.log"
                metrics = measure(model, n_features, workers, preload, config, log)
                repeats.append(Run(workers=workers, preload=preload, repeat=repeat, metrics=metrics))
                logger.info(repeats[-1])

            runs += repeats
            median = Metrics.median([run.metrics for run in repeats])
            summaries.append(Summary(workers=workers, preload=preload, repeats=config.repeats, median=median))

    return Results(machine=machine(model), config=config, summaries=summaries, runs=runs)


def write(results: Results, out: Path) -> ResultsWritten:
    written = ResultsWritten(json_path=out / "cpu-preload.json", markdown_path=out / "cpu-preload.md")
    written.json_path.write_text(results.model_dump_json(indent=2) + "\n")
    written.markdown_path.write_text(results.markdown())

    return written
