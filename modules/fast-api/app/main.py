import gc

from app.apps.inference import create_app
from app.core.preload import preload

# CPython's pattern for prefork servers: no collection while the shared heap is built (a collection frees objects
# and leaves holes that later allocations write into), then freeze it, so no worker's GC ever scans, writes to and
# copies those pages. Each worker enables the GC again in its lifespan.
gc.disable()
app = create_app(preload())
gc.freeze()
