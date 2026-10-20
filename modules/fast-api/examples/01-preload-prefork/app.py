import gc, numpy as np, onnxruntime as ort
from fastapi import FastAPI

def load_model(path="model.onnx"):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1          # one core per worker
    session = ort.InferenceSession(path, opts)
    session.run(None, {"x": np.zeros((1, 1024), np.float32)})   # warm-up
    return session

session = load_model()   # with preload_app: runs once, in the master
gc.freeze()              # GC won't touch (and copy) the shared pages
app = FastAPI()

@app.post("/predict")
async def predict(features: list[float]):
    x = np.asarray([features], np.float32)
    return {"label": int(session.run(None, {"x": x})[0].argmax())}
