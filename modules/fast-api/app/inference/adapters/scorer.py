from typing import Self

from pydantic import BaseModel
from pydantic import Field
from pydantic import FiniteFloat
from pydantic import model_validator

from app.common.enums.adapter import AdapterName
from app.inference.adapters.base import ModelAdapter
from app.inference.adapters.base import named
from app.inference.adapters.base import require
from app.inference.runtime.onnx import OnnxModel
from app.inference.runtime.tensor import Feed
from app.inference.runtime.tensor import Outputs


class ScoredRow(BaseModel):
    features: list[FiniteFloat] = Field(min_length=1)
    segment: int = Field(ge=0)


class ScorerRequest(BaseModel):
    rows: list[ScoredRow] = Field(min_length=1)

    @model_validator(mode="after")
    def check_rectangular(self) -> Self:
        if len({len(row.features) for row in self.rows}) > 1:
            raise ValueError("every row needs the same number of features")

        return self


class ScorerResponse(BaseModel):
    scores: list[float]


class ScorerAdapter(ModelAdapter[ScorerRequest, ScorerResponse], name=AdapterName.SCORER):
    """Records in, a number per record out: a model with two named inputs, `features` (float [batch, n_features]) and
    `segment` (integer [batch]), and one float output, `score` [batch], returned as it is."""

    request = ScorerRequest
    response = ScorerResponse

    def check(self, model: OnnxModel) -> None:
        named(model.inputs, {"features", "segment"}, "input")
        require(model.inputs["features"], "f", rank=2)
        require(model.inputs["segment"], "i", rank=1)

        named(model.outputs, {"score"}, "output")
        require(model.outputs["score"], "f", rank=1)

    def to_feed(self, request: ScorerRequest, model: OnnxModel) -> Feed:
        return {
            "features": model.inputs["features"].array([row.features for row in request.rows]),
            "segment": model.inputs["segment"].array([row.segment for row in request.rows]),
        }

    def from_outputs(self, outputs: Outputs) -> ScorerResponse:
        return ScorerResponse(scores=outputs["score"].tolist())
