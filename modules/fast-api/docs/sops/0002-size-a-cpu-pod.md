# 0002 · Size a CPU pod

| owner | last run | applies to |
| --- | --- | --- |
| @nachatz | 2026-10-18 | CPU pods with preload on; not GPU pods |

## When

- A new model, a new worker count, or a new machine type.
- A pod is OOMKilled, or its memory request is far above what it uses.

## Before you start

- Per-worker costs on the target machine: run [SOP 0005](0005-benchmark-preload.md), or [SOP
  0001](0001-measure-worker-memory.md) at two worker counts.

## Steps

1. Fit the pod's PSS as a line in the worker count, from `bench/results/cpu-preload.json`.

   ```
   pod PSS ≈ base + workers × private per worker
   ```

   Measured fit over 1 to 8 workers on the benchmark machine:

   | | base | per worker |
   | --- | ---: | ---: |
   | with preload | 444.2 MB | 14.0 MB |
   | without preload | 55.3 MB | 406.4 MB |

2. Pick the worker count: one per core.
   Set the pod's CPU request and limit to that count; leave `GUNICORN_WORKERS` unset to read it from
   the cgroup quota.

3. Project the pod's memory.
   For 16 workers with preload: 444.2 + 16 × 14.0 ≈ 668 MB.

4. Set the memory request above the projection, and the limit above the request.
   The request covers steady state; the limit covers bursts of `MODEL_MAX_BATCH` rows.
   For 16 workers: request 1Gi, limit 2Gi.

5. Write the numbers and where they came from next to the resources in the deployment manifest.
   Give the pod a CPU limit equal to its worker count: without one there is no cgroup quota, and
   gunicorn would start one worker per CPU of the node (it logs a warning when that happens).

## Done when

- The deployment's CPU equals its worker count, and its memory request is above the projection.
- After rollout, [SOP 0001](0001-measure-worker-memory.md) in a live pod is within 10% of the
  projection.

## Production scale

This is a projection, not a measurement: the fit above extrapolated to 20 workers, memory only.

| 200 workers of capacity | pods × workers | model copies | PSS per pod | PSS, all pods |
| --- | --- | ---: | ---: | ---: |
| without preload | 100 × 2 | 200 | ~868 MB | ~87 GB |
| with preload | 10 × 20 | 10 | ~724 MB | ~7.2 GB |

Without preload pods stay small, since each worker carries a model; with it, pods get big and few.

## If it doesn't work

| symptom | likely cause | next step |
| --- | --- | --- |
| live PSS far above the projection | workers are copying the model | [SOP 0003](0003-investigate-a-worker-that-copies-the-model.md) |
| OOMKilled under load only | bursts of large batches | raise the limit, or lower `MODEL_MAX_BATCH` |
| CPU throttled, latency up | more workers than the CPU quota | unset `GUNICORN_WORKERS`, or match it to the quota |
