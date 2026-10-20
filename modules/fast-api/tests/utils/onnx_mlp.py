from itertools import pairwise
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper


def build_mlp(path: Path, d_in: int, d: int, depth: int, n_classes: int, seed: int = 0) -> Path:
    """Write an MLP shaped like model.Classifier (input `x`, output `logits`) straight to ONNX, without PyTorch."""
    rng = np.random.default_rng(seed)
    nodes: list[onnx.NodeProto] = []
    weights: list[onnx.TensorProto] = []
    x = "x"

    for i, (w_in, w_out) in enumerate(pairwise([d_in] + [d] * depth + [n_classes])):
        w = (rng.standard_normal((w_in, w_out)) / np.sqrt(w_in)).astype(np.float32)
        b = np.zeros(w_out, np.float32)
        weights += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]

        if i == depth:
            nodes.append(helper.make_node("Gemm", [x, f"w{i}", f"b{i}"], ["logits"]))
        else:
            nodes += [
                helper.make_node("Gemm", [x, f"w{i}", f"b{i}"], [f"linear{i}"]),
                helper.make_node("Relu", [f"linear{i}"], [f"relu{i}"]),
            ]
            x = f"relu{i}"

    graph = helper.make_graph(
        nodes,
        "mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", d_in])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_classes])],
        weights,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # The IR version ONNX Runtime reads, rather than the newest one the onnx package writes by default.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path
