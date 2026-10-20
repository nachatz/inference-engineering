import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto
from pydantic import ValidationError

from app.inference.adapters.classifier import ClassifierAdapter
from app.inference.adapters.classifier import ClassifierRequest
from app.inference.exceptions import IncompatibleModel
from app.inference.runtime.onnx import OnnxModel
from tests.factories.inference import ClassifierRequestFactory
from tests.utils import onnx_graphs

adapter = ClassifierAdapter()


class TestClassifierRequest:
    def test_rejects_an_empty_batch(self) -> None:
        with pytest.raises(ValidationError):
            ClassifierRequest(inputs=[])

    def test_rejects_ragged_rows(self) -> None:
        with pytest.raises(ValidationError, match="same number of features"):
            ClassifierRequest(inputs=[[0.0, 1.0], [0.0]])

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_numbers(self, value: float) -> None:
        with pytest.raises(ValidationError, match="finite"):
            ClassifierRequest(inputs=[[0.0, value]])


class TestCheck:
    def test_accepts_one_input_of_rows_and_one_output_of_logits(self, loaded_model: OnnxModel) -> None:
        adapter.check(loaded_model)

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (onnx_graphs.two_inputs, r"reads one input; the model has \['a', 'b'\]"),
            (onnx_graphs.two_outputs, r"reads one output; the model has \['a', 'b'\]"),
            (onnx_graphs.int_logits, r"'logits' is int64 \[batch, 4\]; the adapter needs a rank-2 float tensor"),
            (
                lambda path: onnx_graphs.identity(path, TensorProto.FLOAT, ["batch"]),
                r"'x' is float32 \[batch\]; the adapter needs a rank-2 float tensor",
            ),
            (
                lambda path: onnx_graphs.identity(path, TensorProto.INT64, ["batch", 4]),
                r"'x' is int64 \[batch, 4\]; the adapter needs a rank-2 float tensor",
            ),
        ],
        ids=["two inputs", "two outputs", "integer logits", "rank 1", "integer features"],
    )
    def test_refuses_any_other_model(self, tmp_path: Path, build: Callable[[Path], Path], match: str) -> None:
        with pytest.raises(IncompatibleModel, match=match):
            adapter.check(OnnxModel.load(build(tmp_path / "model.onnx"), intra_op_threads=1))


class TestConversions:
    def test_feeds_the_models_one_input_in_its_dtype(self, loaded_model: OnnxModel) -> None:
        (x,) = adapter.to_feed(ClassifierRequestFactory.for_model(n_features=8, rows=2), loaded_model).values()

        assert x.shape == (2, 8)
        assert x.dtype == np.dtype(np.float32)

    def test_labels_each_row_with_its_largest_logit(self) -> None:
        response = adapter.from_outputs({"logits": np.array([[0.1, 0.9, 0.0], [2.0, 0.0, 1.0]])})

        assert [p.label for p in response.predictions] == [1, 0]
