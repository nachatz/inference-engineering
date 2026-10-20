import numpy as np
import pytest

from app.inference.exceptions import InputMismatch
from app.inference.runtime.tensor import TensorSpec

rows = TensorSpec("x", np.dtype(np.float32), ("batch", 8))


class TestShape:
    def test_accepts_any_batch(self) -> None:
        assert rows.array(np.zeros((3, 8))).shape == (3, 8)

    @pytest.mark.parametrize("values", [[[0.0] * 7], [0.0] * 8], ids=["feature count", "rank"])
    def test_refuses_what_the_model_cant_take(self, values: list[object]) -> None:
        with pytest.raises(InputMismatch, match=r"input 'x' takes float32 \[batch, 8\]"):
            rows.array(values)

    def test_shows_unnamed_dynamic_dimensions_as_a_question_mark(self) -> None:
        assert str(TensorSpec("ids", np.dtype(np.int64), (None, 128))) == "int64 [?, 128]"


class TestConversion:
    """A JSON number is a float64 or an arbitrary-size int; the model's dtype may not hold it."""

    def test_converts_to_the_models_dtype(self) -> None:
        assert rows.array([[0.5] * 8]).dtype == np.dtype(np.float32)

    @pytest.mark.parametrize(
        ("dtype", "value"),
        [(np.float32, 1e39), (np.float32, -1e39), (np.float16, 70_000.0)],
        ids=["float32 overflow", "float32 negative overflow", "float16 overflow"],
    )
    def test_a_float_beyond_the_dtype_is_refused(self, dtype: type[np.floating], value: float) -> None:
        spec = TensorSpec("x", np.dtype(dtype), ("batch", 2))

        with pytest.raises(InputMismatch, match=f"input 'x' has a value that isn't finite in {np.dtype(dtype)}"):
            spec.array([[value, 0.0]])

    def test_the_largest_float_the_dtype_holds_is_accepted(self) -> None:
        TensorSpec("x", np.dtype(np.float32), ("batch", 1)).array([[float(np.finfo(np.float32).max)]])

    @pytest.mark.parametrize(("dtype", "value"), [(np.int32, 2**31), (np.int8, -129), (np.uint8, -1)])
    def test_an_integer_beyond_the_dtype_is_refused_not_wrapped(self, dtype: type[np.integer], value: int) -> None:
        with pytest.raises(InputMismatch, match="input 'ids': "):
            TensorSpec("ids", np.dtype(dtype), ("batch",)).array([value])

    def test_integers_are_never_checked_for_finiteness(self) -> None:
        TensorSpec("ids", np.dtype(np.int64), ("batch",)).array([2**62])


class TestZeros:
    def test_has_the_fixed_dimensions_and_one_for_each_dynamic_one(self) -> None:
        zeros = TensorSpec("ids", np.dtype(np.int64), ("batch", None, 4)).zeros()

        assert (zeros.shape, zeros.dtype) == ((1, 1, 4), np.dtype(np.int64))
