# 0001 · Four workers served like one

| severity | date | detected by | status |
| --- | --- | --- | --- |
| high | 2026-10-18 | comparing the production server with the minimal example | resolved |

Blameless: this records how the system and the process let it happen, not who did it.

## Summary

With 4 workers, the production server served about as many requests per second as with 1.
One worker accepted every connection and queued them; the other three sat idle.
Giving each worker its own listening socket (`reuse_port`) fixed it.

## Impact

- Found in development, before the first benchmark was published; no deployment ran it.
- Shipped, a 16-CPU pod would have used one core for inference and failed to keep up at a fraction
  of its capacity.
- Measured with single load-test runs (5 to 8 s each, the load generator of the time):

| server, 4 workers unless noted | req/s |
| --- | ---: |
| production server, 1 worker | 152.6 |
| production server | 154.1 |
| the minimal example from the video | 421.4 |
| production server with `reuse_port` | 321.4 |

## Timeline

Times are UTC on the development container.

| when | what |
| --- | --- |
| 06:46 | 4 production workers measured at about the same req/s as 1 |
| 06:47 | `top -H` shows one worker's inference thread at 99.9% CPU and three workers idle |
| 06:47 | the access log shows all 1249 requests served by that one worker |
| 06:52 | `reuse_port = True`; 4 workers measured at 321.4 req/s |

## Root cause

1. Why did one worker get all the work? It accepted every connection.
2. Why? All workers share one listening socket, and asyncio accepts up to 100 waiting connections
   per wakeup.
3. Why did that worker keep accepting while busy? Its event loop was free: inference runs in a
   thread ([ADR 0003](../adrs/0003-one-core-per-worker.md)).
4. Why did the minimal example not do this? It runs inference on the event loop, so a busy worker
   stops accepting.
   That balances load by accident, at the cost of health checks.

The design fact: with a free event loop, a shared socket does not balance load between processes.

## Resolution

`reuse_port = True` on Linux in [`gunicorn.conf.py`](../../gunicorn.conf.py): each worker opens its
own `SO_REUSEPORT` socket and the kernel spreads connections ([ADR
0004](../adrs/0004-one-listening-socket-per-worker.md)).
The load test opens a new connection per request, so it measures the spread rather than one pinned
client.

## Lessons

| went well | went badly |
| --- | --- |
| the minimal example was a baseline that made the gap obvious | nothing checked that requests reach every worker |
| `top -H` and the access log found the cause in a minute | a benchmark run before the comparison would have published one worker's numbers as four |

## Action items

| action | owner | status |
| --- | --- | --- |
| one `SO_REUSEPORT` socket per worker | @nachatz | done |
| the benchmark waits until every worker has answered before measuring | @nachatz | done |
| a slow test that fails when one worker serves most requests: `TestBalance.test_every_worker_does_its_share` in [`tests/integration/test_gunicorn.py`](../../tests/integration/test_gunicorn.py) | @nachatz | done |
