# 0004 · One listening socket per worker (`reuse_port`)

| status | date | owner | supersedes |
| --- | --- | --- | --- |
| accepted | 2026-10-18 | @nachatz | none |

## Context

By default gunicorn's workers share one listening socket, and whichever worker the kernel wakes
accepts the connections.
asyncio accepts up to 100 waiting connections per wakeup.
With inference off the event loop ([ADR 0003](0003-one-core-per-worker.md)), a busy worker's loop is
always free to accept.
So one worker accepted every connection and queued them behind its one inference slot: [PMR
0001](../pmrs/0001-four-workers-served-like-one.md).

## Decision

Set `reuse_port = True` on Linux ([`gunicorn.conf.py`](../../gunicorn.conf.py)).
Each worker opens its own `SO_REUSEPORT` socket, and the kernel hashes new connections across them.

## Alternatives

| option | why not |
| --- | --- |
| block the event loop during inference | balances by accident (a busy worker stops accepting) but starves health checks |
| a reverse proxy in front of the workers | another process and hop per request, for what the kernel does for free |
| keep-alive connections from the load balancer | a long-lived connection pins a client to one worker regardless |

## Consequences

- Load spreads across workers; measured in [PMR 0001](../pmrs/0001-four-workers-served-like-one.md).
- The spread is by hash, not by load: a worker can still get two slow requests in a row.
- Linux only: on macOS `SO_REUSEPORT` does not balance, so the setting is off there.
- Clients should open new connections, or the load balancer should spread them: the load test sends
  `Connection: close`.
