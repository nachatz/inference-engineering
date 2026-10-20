# 0004 · Ship a new model

| owner | last run | applies to |
| --- | --- | --- |
| @nachatz | 2026-10-18 | a new or retrained `model/model.py`, served on CPU |

## When

- The model's architecture, size or weights change.

## Before you start

- The model takes one input `x` of shape `[batch, n_features]` with a fixed `n_features`, and
  returns `logits`.
  The server reads both from the ONNX file ([ADR
  0001](../adrs/0001-serve-onnx-runtime-not-pytorch.md)).

## Steps

1. Export and check it against PyTorch.

   ```bash
   just model verify
   ```

   Expect two lines: `wrote model/model.onnx (… MB)` and `max |pytorch - onnx| = … (tolerance 1e-04)`.

2. Run the whole suite on Linux, slow tests included.

   ```bash
   just cov
   ```

3. Measure memory at the production worker count: [SOP 0001](0001-measure-worker-memory.md).

4. Resize the pod if the model size changed: [SOP 0002](0002-size-a-cpu-pod.md).

5. Re-run the benchmark if the published numbers no longer describe the model: [SOP
   0005](0005-benchmark-preload.md).

6. Build the image; the export and its check run inside it.

   ```bash
   docker build -t fast-api .
   docker run --rm fast-api python -c "import importlib.util as u; print(u.find_spec('torch'))"
   ```

   The second command prints `None`: the runtime image has no PyTorch.

## Done when

- `just model verify` and `just cov` pass, and the deployment's memory matches the new projection.

## If it doesn't work

| symptom | likely cause | next step |
| --- | --- | --- |
| `max \|pytorch - onnx\|` above the tolerance | an op ONNX exports differently | export with a newer opset, or raise `EXPORT_TOLERANCE` with a reason in an ADR |
| `TypeError: the model's feature dimension must be fixed` | a dynamic feature axis | export with a fixed `n_features` |
| private memory per worker ≈ the new model | the model now loads lazily or in the lifespan | [SOP 0003](0003-investigate-a-worker-that-copies-the-model.md) |
