from collections.abc import Mapping
from typing import Any
from typing import Self

import msgspec
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from app.inference.exceptions import IncompatibleModel
from app.inference.exceptions import InputMismatch

DTYPES: dict[str, np.dtype[Any]] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(bool)": np.dtype(np.bool_),
}

type Feed = Mapping[str, NDArray[Any]]
type Outputs = Mapping[str, NDArray[Any]]


def is_finite(array: NDArray[Any]) -> bool:
    return array.dtype.kind != "f" or bool(np.isfinite(array).all())


class TensorSpec(msgspec.Struct, frozen=True):
    """One model input or output as the ONNX file declares it. An int dimension is fixed; a name or None is dynamic.

    `array` is how a request's values become an input: it converts them to the dtype and raises InputMismatch unless
    the result is one the model can take. That means the right rank and fixed dimensions, no integer wrapped around to
    fit the dtype, and no float that became an infinity in it (1e39 is finite in JSON, not in float32). `zeros` is an
    input of the right shape, every dynamic dimension 1, for warm-up.
    """

    name: str
    dtype: np.dtype[Any]
    shape: tuple[int | str | None, ...]

    @classmethod
    def from_node(cls, node: ort.NodeArg) -> Self:
        if node.type not in DTYPES:
            raise IncompatibleModel(f"{node.name!r} is a {node.type}, which this server doesn't map to a NumPy dtype")

        return cls(node.name, DTYPES[node.type], tuple(node.shape))

    def array(self, values: object) -> NDArray[Any]:
        try:
            with np.errstate(over="ignore"):
                array = np.asarray(values, dtype=self.dtype)
        except OverflowError as error:
            raise InputMismatch(f"input {self.name!r}: {error}") from None

        if not self.fits(array.shape):
            raise InputMismatch(f"input {self.name!r} takes {self}, got {array.dtype} {list(array.shape)}")

        if not is_finite(array):
            raise InputMismatch(f"input {self.name!r} has a value that isn't finite in {self.dtype}")

        return array

    def fits(self, shape: tuple[int, ...]) -> bool:
        fixed = [(want, got) for want, got in zip(self.shape, shape, strict=False) if isinstance(want, int)]
        return len(shape) == len(self.shape) and all(want == got for want, got in fixed)

    def zeros(self) -> NDArray[Any]:
        shape = [want if isinstance(want, int) else 1 for want in self.shape]
        return np.zeros(shape, self.dtype)

    def __str__(self) -> str:
        dims = ", ".join("?" if want is None else str(want) for want in self.shape)
        return f"{self.dtype} [{dims}]"
