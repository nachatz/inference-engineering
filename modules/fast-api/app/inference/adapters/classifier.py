from typing import Self

from pydantic import BaseModel
from pydantic import Field
from pydantic import FiniteFloat
from pydantic import model_validator

from app.common.enums.adapter import AdapterName
from app.inference.adapters.base import ModelAdapter
from app.inference.adapters.base import only
from app.inference.adapters.base import require
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.tensor import Feed
from app.inference.runtime.tensor import Outputs


class ClassifierRequest(BaseModel):
    inputs: list[list[FiniteFloat]] = Field(min_length=1)

    @model_validator(mode="after")
    def check_rectangular(self) -> Self:
        if len({len(row) for row in self.inputs}) > 1:
            raise ValueError("every row needs the same number of features")

        return self


class Prediction(BaseModel):
    label: int


class ClassifierResponse(BaseModel):
    predictions: list[Prediction]


class ClassifierAdapter(ModelAdapter[ClassifierRequest, ClassifierResponse], name=AdapterName.CLASSIFIER):
    """Rows of features in, a label per row out: a model with one float input [batch, n_features] and one float output
    [batch, n_classes], whatever they're named. Each row's label is the argmax of its logits."""

    request = ClassifierRequest
    response = ClassifierResponse

    def check(self, model: OnnxModel) -> None:
        require(only(model.inputs, "input"), "f", rank=2)
        require(only(model.outputs, "output"), "f", rank=2)

    def to_feed(self, request: ClassifierRequest, model: OnnxModel) -> Feed:
        (spec,) = model.inputs.values()
        return {spec.name: spec.array(request.inputs)}

    def from_outputs(self, outputs: Outputs) -> ClassifierResponse:
        (logits,) = outputs.values()
        return ClassifierResponse(predictions=[Prediction(label=int(label)) for label in logits.argmax(axis=1)])
