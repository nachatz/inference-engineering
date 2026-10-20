from pathlib import Path

import onnxruntime as ort
import pytest
import torch

from model import seed_everything
from model.export import export
from model.model import Classifier
from model.verify import representative_inputs
from model.verify import verify


def small() -> Classifier:
    return Classifier(d_in=8, d=16, depth=2, n_classes=3)


class TestClassifier:
    def test_default_is_88m_parameters(self) -> None:
        assert 88_000_000 < sum(p.numel() for p in Classifier().parameters()) < 88_200_000

    def test_maps_rows_to_logits(self) -> None:
        assert small()(torch.zeros(4, 8)).shape == (4, 3)

    def test_without_hidden_layers(self) -> None:
        assert Classifier(d_in=8, depth=0, n_classes=3)(torch.zeros(2, 8)).shape == (2, 3)


class TestExport:
    @pytest.fixture
    def out(self, tmp_path: Path) -> Path:
        return tmp_path / "out" / "model.onnx"

    def test_reports_what_it_wrote(self, out: Path) -> None:
        exported = export(small(), out)

        assert exported.path == out
        assert exported.size_bytes == out.stat().st_size

    def test_has_a_dynamic_batch(self, out: Path) -> None:
        export(small(), out)

        session = ort.InferenceSession(str(out))
        (model_input,) = session.get_inputs()
        (output,) = session.get_outputs()
        assert (model_input.name, output.name) == ("x", "logits")
        assert isinstance(model_input.shape[0], str)
        assert model_input.shape[1] == 8


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Classifier, Path]:
    model = small()
    path = tmp_path_factory.mktemp("export") / "model.onnx"
    export(model, path)
    return model, path


class TestVerify:
    @pytest.mark.parametrize("rows", [1, 2, 7, 64])
    def test_the_export_matches_pytorch_at_every_batch_size(self, exported: tuple[Classifier, Path], rows: int) -> None:
        # Exported with a batch of 2: the dynamic batch dimension has to hold for 1 and for the largest batch too.
        model, path = exported
        verified = verify(model, path, representative_inputs(model.d_in, [rows]), tolerance=1e-5)

        assert verified.passed, verified
        assert verified.cases == 4

    def test_reports_the_worst_case(self, exported: tuple[Classifier, Path]) -> None:
        model, path = exported
        verified = verify(model, path, representative_inputs(model.d_in, [1, 3]), tolerance=-1.0)

        assert not verified.passed
        assert verified.cases == 8
        assert verified.worst.endswith(("1 rows", "3 rows"))

    def test_covers_zero_negative_and_large_inputs(self) -> None:
        cases = dict(representative_inputs(4, [2]))

        assert not cases["zeros, 2 rows"].any()
        assert (cases["negative, 2 rows"] <= 0).all()
        assert cases["normal x100, 2 rows"].abs().max() > 10


class TestSeedEverything:
    def test_the_same_seed_builds_the_same_weights(self) -> None:
        seed_everything(7)
        first = small().state_dict()
        seed_everything(7)
        second = small().state_dict()

        assert all(torch.equal(first[name], second[name]) for name in first)
