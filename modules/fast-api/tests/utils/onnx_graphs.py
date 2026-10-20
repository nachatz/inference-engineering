from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper

type Shape = list[int | str | None]


def save(
    path: Path,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] | None = None,
) -> Path:
    graph = helper.make_graph(nodes, path.stem, inputs, outputs, initializers or [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # The IR version ONNX Runtime reads, rather than the newest one the onnx package writes by default.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def tensor(name: str, elem_type: int, shape: Shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def identity(path: Path, elem_type: int, shape: Shape) -> Path:
    """`x` in, `y` out, of the given type and shape."""
    return save(
        path,
        [helper.make_node("Identity", ["x"], ["y"])],
        [tensor("x", elem_type, shape)],
        [tensor("y", elem_type, shape)],
    )


def two_inputs(path: Path) -> Path:
    """`a` + `b`: two inputs, which the classifier can't feed."""
    return save(
        path,
        [helper.make_node("Add", ["a", "b"], ["y"])],
        [tensor("a", TensorProto.FLOAT, ["batch", 4]), tensor("b", TensorProto.FLOAT, ["batch", 4])],
        [tensor("y", TensorProto.FLOAT, ["batch", 4])],
    )


def two_outputs(path: Path) -> Path:
    """Logits twice: an output the classifier wouldn't know to ignore."""
    return save(
        path,
        [helper.make_node("Identity", ["x"], ["a"]), helper.make_node("Identity", ["x"], ["b"])],
        [tensor("x", TensorProto.FLOAT, ["batch", 4])],
        [tensor("a", TensorProto.FLOAT, ["batch", 4]), tensor("b", TensorProto.FLOAT, ["batch", 4])],
    )


def int_logits(path: Path) -> Path:
    """Logits cast to int64: an output whose argmax is meaningless."""
    return save(
        path,
        [helper.make_node("Cast", ["x"], ["logits"], to=TensorProto.INT64)],
        [tensor("x", TensorProto.FLOAT, ["batch", 4])],
        [tensor("logits", TensorProto.INT64, ["batch", 4])],
    )


def scaled_sqrt(path: Path) -> Path:
    """logits = sqrt(x) * 1e30: finite for zeros and small positives, NaN for a negative, infinite for x >= 1e17."""
    return save(
        path,
        [
            helper.make_node("Sqrt", ["x"], ["root"]),
            helper.make_node("Mul", ["root", "scale"], ["logits"]),
        ],
        [tensor("x", TensorProto.FLOAT, ["batch", 4])],
        [tensor("logits", TensorProto.FLOAT, ["batch", 4])],
        [numpy_helper.from_array(np.array(1e30, np.float32), "scale")],
    )


def log(path: Path) -> Path:
    """logits = log(x): -infinity for the zeros warm-up feeds it."""
    return save(
        path,
        [helper.make_node("Log", ["x"], ["logits"])],
        [tensor("x", TensorProto.FLOAT, ["batch", 4])],
        [tensor("logits", TensorProto.FLOAT, ["batch", 4])],
    )


def scorer(path: Path, n_features: int, segment_type: int = TensorProto.INT64) -> Path:
    """score = features @ weights + 10 * segment: the scorer adapter's model, two inputs (one of them int64) and one
    output, without a classifier's argmax."""
    weights = np.arange(1, n_features + 1, dtype=np.float32).reshape(n_features, 1)
    return save(
        path,
        [
            helper.make_node("MatMul", ["features", "weights"], ["projected"]),
            helper.make_node("Squeeze", ["projected", "axis"], ["linear"]),
            helper.make_node("Cast", ["segment"], ["segment_float"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["segment_float", "ten"], ["offset"]),
            helper.make_node("Add", ["linear", "offset"], ["score"]),
        ],
        [tensor("features", TensorProto.FLOAT, ["batch", n_features]), tensor("segment", segment_type, ["batch"])],
        [tensor("score", TensorProto.FLOAT, ["batch"])],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(np.array([1], np.int64), "axis"),
            numpy_helper.from_array(np.array(10, np.float32), "ten"),
        ],
    )


def spinner(path: Path, d_in: int) -> Path:
    """A classifier whose inference runs a loop of max(x) iterations inside ONNX Runtime, holding no Python lock.

    Features in [-1, 1] run it no times, so it answers at once; a feature of 1e6 makes it run for about a second, and
    1e12 for longer than any test. A native hang, and a slow inference, on demand.
    """
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond_in"], ["cond_out"]),
            helper.make_node("Add", ["count_in", "one"], ["count_out"]),
        ],
        "body",
        [
            tensor("iteration", TensorProto.INT64, []),
            tensor("cond_in", TensorProto.BOOL, []),
            tensor("count_in", TensorProto.FLOAT, []),
        ],
        [tensor("cond_out", TensorProto.BOOL, []), tensor("count_out", TensorProto.FLOAT, [])],
        [numpy_helper.from_array(np.array(1, np.float32), "one")],
    )
    weights = np.eye(d_in, 2, dtype=np.float32)

    return save(
        path,
        [
            helper.make_node("ReduceMax", ["x"], ["largest"], keepdims=0),
            helper.make_node("Cast", ["largest"], ["trips"], to=TensorProto.INT64),
            helper.make_node("Loop", ["trips", "", "zero"], ["count"], body=body),
            helper.make_node("Mul", ["count", "zero"], ["nothing"]),
            helper.make_node("MatMul", ["x", "weights"], ["projected"]),
            helper.make_node("Add", ["projected", "nothing"], ["logits"]),
        ],
        [tensor("x", TensorProto.FLOAT, ["batch", d_in])],
        [tensor("logits", TensorProto.FLOAT, ["batch", 2])],
        [numpy_helper.from_array(np.array(0, np.float32), "zero"), numpy_helper.from_array(weights, "weights")],
    )
