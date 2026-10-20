# 0005 · Benchmark preload

| owner | last run | applies to |
| --- | --- | --- |
| @nachatz | 2026-10-18 | the published numbers in the ADRs, SOPs and READMEs |

## When

- The model, the serving code, or the machine changes, and the published numbers might no longer
  hold.

## Before you start

- Linux, with nothing else running: the load generator shares the machine with the server.
- An exported model: `just model`.

## Steps

1. Run the benchmark: every worker count, with and without preload, `BENCH_REPEATS` times.

   ```bash
   just bench
   ```

   | setting | default | what it does |
   | --- | --- | --- |
   | `BENCH_WORKERS` | `[1, 2, 4, 8]` | worker counts to try |
   | `BENCH_REPEATS` | `3` | runs per configuration; the table shows the median |
   | `BENCH_USERS` | `16` | locust users, each sending its next request when the last returns |
   | `BENCH_SECONDS` | `10` | length of each measured run |
   | `BENCH_ROWS` | `1` | rows per request |
   | `BENCH_OUT` | `bench/results` | where the results go |

   With the defaults it takes about 10 minutes and logs one line per run.

2. Check the results: `bench/results/cpu-preload.md` (the table) and `cpu-preload.json` (every run,
   the machine).
   `errors` must be 0 in every row.

3. Copy the numbers from the JSON into the documents that quote them; never type them from memory.

   | document | what it quotes |
   | --- | --- |
   | [ADR 0002](../adrs/0002-preload-the-model-and-fork-workers.md) | the full table and the machine |
   | [SOP 0002](0002-size-a-cpu-pod.md) | the fit and the projection |
   | the deployment manifest | the sizing comment |
   | the repo's `README.md` | the 4-worker row |

## Done when

- Every published number matches `bench/results/cpu-preload.json`, and the machine it came from is
  named.

## If it doesn't work

| symptom | likely cause | next step |
| --- | --- | --- |
| `errors` above 0 | the server rejected or dropped requests | read `bench/results/logs/` for that run |
| `TimeoutError: only N of M workers answered` | a worker didn't start, or connections didn't spread | the run's log; [ADR 0004](../adrs/0004-one-listening-socket-per-worker.md) |
| req/s barely changes with workers | connections land on one worker | [PMR 0001](../pmrs/0001-four-workers-served-like-one.md) |
| numbers swing between repeats | something else is using the CPU | stop it and re-run |
