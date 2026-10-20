from pathlib import Path

import torch

from model.model import Classifier
from model.outputs import ModelExported


def export(model: Classifier, out: Path) -> ModelExported:
    """Write `model` to `out` as one ONNX file: input `x`, output `logits`, and a dynamic batch dimension."""
    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.randn(2, model.d_in),),
        out,
        input_names=["x"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        external_data=False,
        verbose=False,
    )

    return ModelExported(path=out, size_bytes=out.stat().st_size)
