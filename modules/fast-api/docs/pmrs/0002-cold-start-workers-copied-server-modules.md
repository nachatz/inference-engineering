# 0002 · Cold-start workers copied the server's modules

| severity | date | detected by | status |
| --- | --- | --- | --- |
| low | 2026-10-18 | `tests/integration/test_gunicorn.py`, once, on a fresh container | resolved |

Blameless: this records how the system and the process let it happen, not who did it.

## Summary

The slow test that checks preloaded workers share the model failed once, then passed on every
re-run.
On a cold start each worker compiled and imported uvicorn's and anyio's lazy modules itself, as
private memory.
Importing those modules in the master fixed it.

## Impact

- One flaky failure of the memory test; no deployment affected.
- The test's model is 67.6 MB, and it fails a worker above 30% of it (20.3 MB) private.

Private memory per worker, from repeated runs of the test's server (2 workers, preload on):

| | no `.pyc` files (cold) | `.pyc` files present (warm) |
| --- | ---: | ---: |
| before | 21.6 MB (32%) | 13.5 MB (20%) |
| after | 13.2 MB (20%) | 11.0 MB (16%) |

## Timeline

| when | what |
| --- | --- |
| first run | the full suite fails `test_workers_share_the_model` on a fresh container |
| re-runs | the test passes 7 times in a row |
| probe | 15 runs with `.pyc` files: 13.5 MB per worker every time; with them deleted: 21.6 MB |
| probe | the modules a worker imports after the fork: anyio's asyncio backend, uvicorn's lifespan, loop and HTTP protocol, h11 |
| fix | those modules imported in the master; cold start measured at 13.2 MB |

## Root cause

1. Why were workers over the threshold? They held 8 MB more private memory than usual.
2. Why? They imported modules after the fork, so the module objects lived in each worker's own
   pages.
3. Why only sometimes? Without `.pyc` files each worker also compiled those modules, and the
   allocator kept the memory.
4. Why were they imported after the fork? uvicorn and anyio import them lazily, when a worker starts
   serving.

The design fact: preload shares what is imported before the fork, and libraries decide some of that
at runtime.

## Resolution

[`gunicorn.conf.py`](../../gunicorn.conf.py) imports the lazily loaded modules in the master, before
it forks.
The image compiles bytecode at build time (`UV_COMPILE_BYTECODE=1` in the
[`Dockerfile`](../../Dockerfile)).

## Lessons

| went well | went badly |
| --- | --- |
| the memory test caught an 8 MB regression per worker | the first failure looked like flakiness, not a signal |
| deleting `.pyc` files reproduced it on demand | the list of modules depends on uvicorn's internals |

## Action items

| action | owner | status |
| --- | --- | --- |
| import uvicorn's and anyio's lazy modules in the master | @nachatz | done |
| compile bytecode in the image | @nachatz | done |
| a test that fails when a worker imports modules after the fork, so a uvicorn upgrade can't regress it | @nachatz | open |
