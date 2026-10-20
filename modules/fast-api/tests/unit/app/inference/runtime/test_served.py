import hashlib
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from app.common.enums.adapter import AdapterName
from app.core.config import ModelSettings
from app.inference.adapters.scorer import ScorerAdapter
from app.inference.exceptions import IncompatibleModel
from app.inference.exceptions import ModelChecksumMismatch
from app.inference.exceptions import NonFiniteOutput
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.served import ServedModel
from tests.utils import onnx_graphs


class TestServedModel:
    def test_reports_which_artifact_it_loaded(self, tiny_model_settings: ModelSettings) -> None:
        served = ServedModel.load(tiny_model_settings)
        path = tiny_model_settings.path
        assert (served.info.name, served.info.version) == ("tiny", "test")
        assert served.info.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert served.info.size_bytes == path.stat().st_size

    def test_warms_the_model_up(self, tiny_model_settings: ModelSettings, mocker: MockerFixture) -> None:
        warm_up = mocker.spy(OnnxModel, OnnxModel.warm_up.__name__)
        ServedModel.load(tiny_model_settings.model_copy(update={"warmup_runs": 2}))
        assert warm_up.call_args.args[1] == 2

    def test_loads_the_pinned_artifact(self, tiny_model_settings: ModelSettings) -> None:
        sha256 = hashlib.sha256(tiny_model_settings.path.read_bytes()).hexdigest()
        assert ServedModel.load(tiny_model_settings.model_copy(update={"sha256": sha256})).info.sha256 == sha256

    def test_refuses_any_other_file(self, tiny_model_settings: ModelSettings) -> None:
        with pytest.raises(ModelChecksumMismatch, match="MODEL_SHA256 pins 000"):
            ServedModel.load(tiny_model_settings.model_copy(update={"sha256": "0" * 64}))


class TestAdapter:
    def test_serves_the_adapter_its_settings_name(self, tmp_path: Path) -> None:
        path = onnx_graphs.scorer(tmp_path / "scorer.onnx", n_features=3)
        served = ServedModel.load(ModelSettings(path=path, adapter=AdapterName.SCORER))

        assert isinstance(served.adapter, ScorerAdapter)

    def test_a_model_the_adapter_cant_serve_fails_to_load(
        self, tiny_model_settings: ModelSettings, mocker: MockerFixture
    ) -> None:
        warm_up = mocker.spy(OnnxModel, OnnxModel.warm_up.__name__)

        with pytest.raises(IncompatibleModel, match=r"the adapter's inputs are \['features', 'segment'\]"):
            ServedModel.load(tiny_model_settings.model_copy(update={"adapter": AdapterName.SCORER}))

        # Checked before the model ever runs.
        warm_up.assert_not_called()

    def test_a_model_that_computes_a_non_finite_result_fails_to_load(self, tmp_path: Path) -> None:
        # log(0), from the zeros warm-up feeds it, is -infinity.
        with pytest.raises(NonFiniteOutput):
            ServedModel.load(ModelSettings(path=onnx_graphs.log(tmp_path / "log.onnx")))
