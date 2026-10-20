from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto
from pytest_mock import MockerFixture

from app.core.config import ModelSettings
from app.inference.exceptions import IncompatibleModel
from app.inference.exceptions import NonFiniteOutput
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.tensor import TensorSpec
from tests.utils import onnx_graphs


@pytest.fixture(scope="module")
def sqrt_model(tmp_path_factory: pytest.TempPathFactory) -> OnnxModel:
    return OnnxModel.load(onnx_graphs.scaled_sqrt(tmp_path_factory.mktemp("sqrt") / "sqrt.onnx"), 1)


class TestOnnxModel:
    @pytest.fixture
    def model(self, tiny_model_settings: ModelSettings) -> OnnxModel:
        return OnnxModel.load(tiny_model_settings.path, intra_op_threads=1)

    def test_reads_its_inputs_and_outputs_from_the_onnx_file(self, model: OnnxModel) -> None:
        assert model.inputs == {"x": TensorSpec("x", np.dtype(np.float32), ("batch", 8))}
        assert model.outputs == {"logits": TensorSpec("logits", np.dtype(np.float32), ("batch", 3))}

    def test_runs_to_named_outputs(self, model: OnnxModel) -> None:
        assert model.run({"x": np.zeros((4, 8), np.float32)})["logits"].shape == (4, 3)

    def test_warm_up_runs_the_model(self, model: OnnxModel, mocker: MockerFixture) -> None:
        run = mocker.spy(model, model.run.__name__)
        model.warm_up(3)
        assert run.call_count == 3

    def test_serves_other_dtypes_and_dynamic_dimensions(self, tmp_path: Path) -> None:
        # e.g. token IDs of any length: int64 [batch, sequence]. Warm-up feeds one token.
        model = OnnxModel.load(onnx_graphs.identity(tmp_path / "ids.onnx", TensorProto.INT64, ["batch", "seq"]), 1)
        model.warm_up(1)
        ids = model.inputs["x"].array([[0, 1, 2], [3, 4, 5]])

        assert (model.run({"x": ids})["y"] == ids).all()

    def test_refuses_a_type_it_cant_map(self, tmp_path: Path) -> None:
        with pytest.raises(IncompatibleModel, match=r"tensor\(string\)"):
            OnnxModel.load(onnx_graphs.identity(tmp_path / "text.onnx", TensorProto.STRING, ["batch"]), 1)


class TestNonFiniteOutputs:
    """logits = sqrt(x) * 1e30: every input below is finite, so only the computation can make it NaN or infinite."""

    def test_a_finite_result_is_returned(self, sqrt_model: OnnxModel) -> None:
        assert np.isfinite(sqrt_model.run({"x": np.ones((2, 4), np.float32)})["logits"]).all()

    @pytest.mark.parametrize("value", [-1.0, 1e20], ids=["NaN", "overflow"])
    def test_a_nan_or_an_infinity_raises(self, sqrt_model: OnnxModel, value: float) -> None:
        x = np.ones((2, 4), np.float32)
        x[1, 2] = value

        with pytest.raises(NonFiniteOutput, match="output 'logits' isn't finite"):
            sqrt_model.run({"x": x})
