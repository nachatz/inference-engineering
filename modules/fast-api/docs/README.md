# Docs

Three kinds of document, each in its own folder, each numbered in the order it was written.
Start a new one by copying the folder's `template.md` to the next free number.

| folder | what it records | written when |
| --- | --- | --- |
| [`adrs/`](adrs/) | architecture decision records: a decision, why, and what it costs | a design choice is made |
| [`sops/`](sops/) | standard operating procedures: steps anyone can follow | a task is done more than once |
| [`pmrs/`](pmrs/) | post-mortem reports: what broke, why, and what changed | something went wrong |

Style: one sentence per line, no line over 100 characters, no em-dashes.
Prefer tables and code to paragraphs.
Every number comes from `just bench` or `bench/mem.py` on a named machine, never from memory.

## Architecture decision records

| # | decision | status |
| ---: | --- | --- |
| [0001](adrs/0001-serve-onnx-runtime-not-pytorch.md) | Serve with ONNX Runtime; PyTorch only at build time | accepted |
| [0002](adrs/0002-preload-the-model-and-fork-workers.md) | Preload the model and fork the workers | accepted |
| [0003](adrs/0003-one-core-per-worker.md) | One core, one thread and one inference per worker | accepted |
| [0004](adrs/0004-one-listening-socket-per-worker.md) | One listening socket per worker (`reuse_port`) | accepted |
| [0005](adrs/0005-per-worker-state-after-the-fork.md) | Create per-worker state after the fork | accepted |
| [0006](adrs/0006-jemalloc.md) | Run the container on jemalloc | accepted |

## Standard operating procedures: CPU model memory

| # | procedure |
| ---: | --- |
| [0001](sops/0001-measure-worker-memory.md) | Measure worker memory |
| [0002](sops/0002-size-a-cpu-pod.md) | Size a CPU pod |
| [0003](sops/0003-investigate-a-worker-that-copies-the-model.md) | Investigate a worker that copies the model |
| [0004](sops/0004-ship-a-new-model.md) | Ship a new model |
| [0005](sops/0005-benchmark-preload.md) | Benchmark preload |

## Post-mortem reports

| # | incident | severity |
| ---: | --- | --- |
| [0001](pmrs/0001-four-workers-served-like-one.md) | Four workers served like one | high |
| [0002](pmrs/0002-cold-start-workers-copied-server-modules.md) | Cold-start workers copied the server's modules | low |
