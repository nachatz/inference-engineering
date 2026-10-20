from pathlib import Path
from typing import Any
from typing import Self

import onnxruntime as ort
from numpy.typing import NDArray

from app.inference.exceptions import NonFiniteOutput
from app.inference.runtime.tensor import Feed
from app.inference.runtime.tensor import TensorSpec
from app.inference.runtime.tensor import is_finite


class OnnxModel:
    """An ONNX Runtime session on the CPU, and the inputs and outputs its file declares.

    `run` returns the outputs by name, or raises NonFiniteOutput if any holds a NaN or an infinity: an argmax or a
    threshold would turn it into an answer that looks like any other. `warm_up` runs it on zeros, so first-run
    allocations happen at load, before the fork, rather than in a request.
    """

    def __init__(self, session: ort.InferenceSession) -> None:
        self.inputs = {node.name: TensorSpec.from_node(node) for node in session.get_inputs()}
        self.outputs = {node.name: TensorSpec.from_node(node) for node in session.get_outputs()}
        self._session = session

    @classmethod
    def load(cls, path: Path, intra_op_threads: int) -> Self:
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1

        return cls(ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]))

    def run(self, feed: Feed) -> dict[str, NDArray[Any]]:
        outputs = dict(zip(self.outputs, self._session.run(list(self.outputs), feed), strict=True))

        for name, output in outputs.items():
            if not is_finite(output):
                raise NonFiniteOutput(name)

        return outputs

    def warm_up(self, runs: int) -> None:
        feed = {name: spec.zeros() for name, spec in self.inputs.items()}

        for _ in range(runs):
            self.run(feed)
