from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram

REQUESTS = Counter("predict_requests_total", "Prediction requests, by HTTP status.", ["status"])
REQUEST_SECONDS = Histogram("predict_request_seconds", "Prediction request latency, queueing included.")
INFERENCE_SECONDS = Histogram("predict_inference_seconds", "Time spent in ONNX Runtime per request.")
ROWS = Histogram("predict_rows", "Rows per prediction request.", buckets=(1, 2, 4, 8, 16, 32, 64, 128))

# Under gunicorn, livesum adds up the live workers: the pod's queue, and how many of its workers are busy.
ADMITTED = Gauge(
    "predict_admitted",
    "Prediction requests a worker holds: reading, parsing, queued or running.",
    multiprocess_mode="livesum",
)
QUEUE_DEPTH = Gauge(
    "predict_queue_depth",
    "Requests waiting for their worker's inference slot.",
    multiprocess_mode="livesum",
)
INFLIGHT = Gauge("predict_inflight", "Inferences running.", multiprocess_mode="livesum")
WORKERS = Gauge("workers_serving", "Workers that have started serving: 1 per live worker.", multiprocess_mode="livesum")

SHED = Counter("predict_shed_total", "Requests refused with a 503 because their worker was overloaded.", ["reason"])

MODEL_INFO = Gauge(
    "model_info",
    "The model this server runs: 1, labelled with its identity.",
    ["name", "version", "sha256"],
    multiprocess_mode="max",
)
