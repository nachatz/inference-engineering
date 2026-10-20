from collections.abc import Iterable
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from model.model import Classifier
from model.outputs import ExportVerified


def representative_inputs(d_in: int, batch_sizes: Iterable[int]) -> Iterator[tuple[str, Tensor]]:
    """Every batch size, each with the inputs an export gets wrong: zeros (a bias-only path), the unit scale the
    weights were made for, all-negative (every first ReLU off), and a hundred times larger (an overflow or a
    precision loss shows up first in large activations)."""
    for rows in batch_sizes:
        yield f"zeros, {rows} rows", torch.zeros(rows, d_in)
        yield f"normal, {rows} rows", torch.randn(rows, d_in)
        yield f"negative, {rows} rows", -torch.rand(rows, d_in)
        yield f"normal x100, {rows} rows", 100 * torch.randn(rows, d_in)


def verify(
    model: Classifier, onnx_path: Path, inputs: Iterable[tuple[str, Tensor]], tolerance: float
) -> ExportVerified:
    """Compare `model` with its ONNX export on every input. The error is relative to the output's scale, so one
    tolerance fits both unit-scale and large inputs: max |pytorch - onnx| / max(1, max |pytorch|)."""
    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    errors: dict[str, float] = {}

    for case, x in inputs:
        with torch.no_grad():
            expected = model(x).numpy()
        (actual,) = session.run(None, {"x": x.numpy()})

        errors[case] = float(np.abs(expected - actual).max() / max(1.0, np.abs(expected).max()))

    worst = max(errors, key=errors.__getitem__)
    return ExportVerified(error=errors[worst], worst=worst, cases=len(errors), tolerance=tolerance)
