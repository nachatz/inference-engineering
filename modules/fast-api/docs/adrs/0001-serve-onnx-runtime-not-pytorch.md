# 0001 · Serve with ONNX Runtime; PyTorch only at build time

| status | date | owner | supersedes |
| --- | --- | --- | --- |
| accepted | 2026-10-18 | @nachatz | none |

## Context

The model is a PyTorch module ([`model/model.py`](../../model/model.py)): an 88M-parameter MLP, 353
MB in float32.
The server needs a CPU runtime that loads it once, runs it read-only, and can be forked ([ADR
0002](0002-preload-the-model-and-fork-workers.md)).
PyTorch is a large dependency to ship and import, and its thread pools and allocator are built for
training.

## Decision

Export the model to ONNX once, at build time, and serve the `.onnx` file with ONNX Runtime.

| step | where |
| --- | --- |
| export with `torch.onnx.export`: input `x`, output `logits`, dynamic batch, one file | [`model/export.py`](../../model/export.py) |
| check max |PyTorch - ONNX| on a random batch, fail above `EXPORT_TOLERANCE` | [`model/verify.py`](../../model/verify.py) |
| run both in the image's build stage | [`Dockerfile`](../../Dockerfile) |
| load and run the session, never import torch | [`app/inference/runtime/onnx.py`](../../app/inference/runtime/onnx.py) |

## Alternatives

| option | why not |
| --- | --- |
| serve the PyTorch module | ships and imports torch in every pod; nothing in serving needs autograd |
| TorchScript | deprecated in favour of `torch.export`; still needs libtorch at runtime |
| export at startup | every pod would need PyTorch and pay the export time on every start |

## Consequences

- The runtime image has ONNX Runtime and no PyTorch; `torch` lives in the `export` dependency group
  only.
- The model's input name and feature count are read from the ONNX file, so the API validates against
  the file it serves.
- An export that drifts from PyTorch fails the build: `just model verify` exits non-zero.
- Changing the model follows [SOP 0004](../sops/0004-ship-a-new-model.md).
