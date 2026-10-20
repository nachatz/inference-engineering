from typing import Any

from app.core.config import Settings
from app.core.config import settings


def with_model(**changes: Any) -> Settings:
    """The process's settings with some MODEL_* settings changed."""
    return settings.model_copy(update={"model": settings.model.model_copy(update=changes)})
