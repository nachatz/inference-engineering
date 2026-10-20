from polyfactory.factories.pydantic_factory import ModelFactory

from app.inference.adapters.classifier import ClassifierRequest
from app.inference.adapters.scorer import ScoredRow
from app.inference.adapters.scorer import ScorerRequest


class ClassifierRequestFactory(ModelFactory[ClassifierRequest]):
    @classmethod
    def for_model(cls, n_features: int, rows: int = 1) -> ClassifierRequest:
        return cls.build(
            inputs=[[cls.__faker__.pyfloat(min_value=-1, max_value=1) for _ in range(n_features)] for _ in range(rows)]
        )


class ScorerRequestFactory(ModelFactory[ScorerRequest]):
    @classmethod
    def for_model(cls, n_features: int, rows: int = 1) -> ScorerRequest:
        return cls.build(
            rows=[
                ScoredRow(
                    features=[cls.__faker__.pyfloat(min_value=-1, max_value=1) for _ in range(n_features)],
                    segment=cls.__faker__.pyint(min_value=0, max_value=9),
                )
                for _ in range(rows)
            ]
        )
