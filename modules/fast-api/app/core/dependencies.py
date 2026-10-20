from typing import Annotated

from fastapi import Depends
from fastapi import Request

from app.core.config import Settings


def get_config(request: Request) -> Settings:
    config: Settings = request.app.state.config
    return config


ConfigDependency = Annotated[Settings, Depends(get_config)]
