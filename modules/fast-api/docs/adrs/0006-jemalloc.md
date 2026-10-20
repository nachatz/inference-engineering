# 0006 · Run the container on jemalloc

| status | date | owner | supersedes |
| --- | --- | --- | --- |
| accepted | 2026-10-18 | @nachatz | none |

## Context

Every worker allocates per request: request bodies, numpy arrays, ONNX Runtime's output tensors.
glibc's malloc keeps freed memory in per-thread arenas and fragments under long-running churn.
jemalloc bounds fragmentation and can purge freed memory on a background thread.
Whatever the allocator, it must not break the sharing [ADR
0002](0002-preload-the-model-and-fork-workers.md) depends on.

## Decision

The runtime image installs `libjemalloc2` and preloads it into every process
([`Dockerfile`](../../Dockerfile)).

| variable | value |
| --- | --- |
| `LD_PRELOAD` | `/usr/local/lib/libjemalloc.so.2` (a symlink, so the path is the same on amd64 and arm64) |
| `MALLOC_CONF` | `background_thread:true,metadata_thp:auto,dirty_decay_ms:90000,muzzy_decay_ms:90000` |

`GET /debug/allocator` reports the allocator actually mapped into the worker, not the one requested.

## Alternatives

| option | why not |
| --- | --- |
| glibc malloc | measured below as slightly leaner over 10 s; no defence against fragmentation over days |
| tcmalloc | similar goals; jemalloc is the one already standard in this team's services |
| jemalloc with default decay | holds less freed memory (below); revisit after a soak test |

## Consequences

Measured on the benchmark machine ([ADR 0002](0002-preload-the-model-and-fork-workers.md)): the
image with `--cpus 4`, so 4 workers.
Three 10 s locust runs each (16 users), then PSS and private memory read from `/proc` in the
container.

| allocator | req/s | PSS, all processes | private per worker |
| --- | ---: | ---: | ---: |
| glibc | 323.8 to 339.5 | 517 to 518 MB | 14.8 MB |
| jemalloc, default decay | 315.0 to 318.3 | 525 to 527 MB | 16.1 to 16.5 MB |
| jemalloc, the configuration above | 309.9 to 323.1 | 535 to 538 MB | 17.0 to 17.1 MB |

- Workers still share the model: about 2 MB more private memory per worker, against 353 MB shared.
- The 90 s decay holds freed pages on purpose, about 1 MB per worker here; it saves `mmap` churn
  under steady load.
- A 10 s run cannot show fragmentation. The case for jemalloc rests on long uptimes, which is
  untested here.
- `background_thread` adds a native thread: a worker reports 2 in `/debug/runtime`, which is how to check it
  survived the fork.
- Follow-up: a 24 h soak of glibc against jemalloc, then keep, retune or supersede this ADR.
