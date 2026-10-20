import pytest

from app.core.config import ModelSettings
from app.inference.capacity.slot import InferenceSlot
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.served import ServedModel
from tests.utils.onnx_mlp import build_mlp


@pytest.fixture(scope="session")
def tiny_model_settings(tmp_path_factory: pytest.TempPathFactory) -> ModelSettings:
    path = build_mlp(tmp_path_factory.mktemp("model") / "tiny.onnx", d_in=8, d=16, depth=2, n_classes=3)
    return ModelSettings(path=path, name="tiny", version="test")


@pytest.fixture(scope="session")
def served(tiny_model_settings: ModelSettings) -> ServedModel:
    return ServedModel.load(tiny_model_settings)


@pytest.fixture(scope="session")
def loaded_model(served: ServedModel) -> OnnxModel:
    return served.model


@pytest.fixture
def slot() -> InferenceSlot:
    return InferenceSlot(max_queue=4, max_queue_wait_s=5.0)
