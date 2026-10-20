from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any
from typing import ClassVar

from pydantic import BaseModel

from app.common.enums.adapter import AdapterName
from app.inference.exceptions import IncompatibleModel
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.tensor import Feed
from app.inference.runtime.tensor import Outputs
from app.inference.runtime.tensor import TensorSpec

KINDS = {"f": "float", "i": "integer"}


class ModelAdapter[Request: BaseModel, Response: BaseModel](ABC):
    """What a service writes for its model: the request and response schemas, and the conversions between them and
    the model's tensors. Queueing, limits, the numerical checks and the metrics around it are the same for any model.

    - `check` runs at startup and raises IncompatibleModel unless the model has the inputs `to_feed` builds and the
      outputs `from_outputs` reads, so a model the adapter can't serve never takes a request.
    - `to_feed` builds each input with `TensorSpec.array`, which refuses a malformed request before it is queued.
    - Every subclass registers itself under the name MODEL_ADAPTER selects it by:
      `class ClassifierAdapter(ModelAdapter[...], name=AdapterName.CLASSIFIER)`, found with `for_name`.
    """

    registry: ClassVar[dict[AdapterName, type["ModelAdapter[Any, Any]"]]] = {}

    request: type[Request]
    response: type[Response]

    def __init_subclass__(cls, name: AdapterName, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        ModelAdapter.registry[name] = cls

    @staticmethod
    def for_name(name: AdapterName) -> "ModelAdapter[Any, Any]":
        return ModelAdapter.registry[name]()

    @abstractmethod
    def check(self, model: OnnxModel) -> None: ...

    @abstractmethod
    def to_feed(self, request: Request, model: OnnxModel) -> Feed: ...

    @abstractmethod
    def from_outputs(self, outputs: Outputs) -> Response: ...


def only(specs: Mapping[str, TensorSpec], role: str) -> TensorSpec:
    if len(specs) != 1:
        raise IncompatibleModel(f"the adapter reads one {role}; the model has {sorted(specs)}")

    (spec,) = specs.values()
    return spec


def named(specs: Mapping[str, TensorSpec], names: set[str], role: str) -> None:
    if specs.keys() != names:
        raise IncompatibleModel(f"the adapter's {role}s are {sorted(names)}; the model's are {sorted(specs)}")


def require(spec: TensorSpec, kind: str, rank: int) -> None:
    if spec.dtype.kind != kind or len(spec.shape) != rank:
        raise IncompatibleModel(f"{spec.name!r} is {spec}; the adapter needs a rank-{rank} {KINDS[kind]} tensor")
