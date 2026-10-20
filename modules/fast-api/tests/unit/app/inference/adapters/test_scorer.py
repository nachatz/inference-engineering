from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto
from pydantic import ValidationError

from app.inference.adapters.scorer import ScoredRow
from app.inference.adapters.scorer import ScorerAdapter
from app.inference.adapters.scorer import ScorerRequest
from app.inference.exceptions import IncompatibleModel
from app.inference.runtime.onnx import OnnxModel
from tests.factories.inference import ScorerRequestFactory
from tests.utils import onnx_graphs
from tests.utils.onnx_mlp import build_mlp

adapter = ScorerAdapter()


@pytest.fixture(scope="module")
def scorer(tmp_path_factory: pytest.TempPathFactory) -> OnnxModel:
    return OnnxModel.load(onnx_graphs.scorer(tmp_path_factory.mktemp("scorer") / "scorer.onnx", n_features=3), 1)


class TestScorerRequest:
    def test_rejects_ragged_rows(self) -> None:
        rows = [ScoredRow(features=[0.0, 1.0], segment=0), ScoredRow(features=[0.0], segment=0)]
        with pytest.raises(ValidationError, match="same number of features"):
            ScorerRequest(rows=rows)

    def test_rejects_a_negative_segment(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ScoredRow(features=[0.0], segment=-1)


class TestCheck:
    def test_accepts_its_model(self, scorer: OnnxModel) -> None:
        adapter.check(scorer)

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (
                lambda path: build_mlp(path, d_in=3, d=4, depth=1, n_classes=2),
                r"inputs are \['features', 'segment'\]; the model's are \['x'\]",
            ),
            (
                lambda path: onnx_graphs.scorer(path, n_features=3, segment_type=TensorProto.FLOAT),
                r"'segment' is float32 \[batch\]; the adapter needs a rank-1 integer tensor",
            ),
        ],
        ids=["a classifier", "a float segment"],
    )
    def test_refuses_any_other_model(self, tmp_path: Path, build: Callable[[Path], Path], match: str) -> None:
        with pytest.raises(IncompatibleModel, match=match):
            adapter.check(OnnxModel.load(build(tmp_path / "model.onnx"), intra_op_threads=1))


class TestConversions:
    def test_feeds_both_inputs_in_their_dtypes(self, scorer: OnnxModel) -> None:
        feed = adapter.to_feed(ScorerRequestFactory.for_model(n_features=3, rows=2), scorer)

        assert (feed["features"].shape, feed["features"].dtype) == ((2, 3), np.dtype(np.float32))
        assert (feed["segment"].shape, feed["segment"].dtype) == ((2,), np.dtype(np.int64))

    def test_returns_each_score_as_it_is(self, scorer: OnnxModel) -> None:
        request = ScorerRequest(
            rows=[ScoredRow(features=[1, 0, 0], segment=0), ScoredRow(features=[0, 0, 1], segment=2)]
        )

        assert adapter.from_outputs(scorer.run(adapter.to_feed(request, scorer))).scores == [1.0, 23.0]
