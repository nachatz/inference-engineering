bind = "0.0.0.0:8000"
workers = 4   # start near one per CPU core
worker_class = "uvicorn_worker.UvicornWorker"   # FastAPI in each worker
preload_app = True   # import app.py (and the model) once, then fork