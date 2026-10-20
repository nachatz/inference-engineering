# Just the slice of onnxruntime this repo uses: the package ships no type hints.
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray

__version__: str

class SessionOptions:
    intra_op_num_threads: int
    inter_op_num_threads: int
    def __init__(self) -> None: ...

class NodeArg:
    name: str
    type: str  # e.g. "tensor(float)", "tensor(int64)"
    shape: list[int | str | None]

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str] | None = None,
    ) -> None: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_outputs(self) -> list[NodeArg]: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, NDArray[Any]],
    ) -> list[NDArray[Any]]: ...
