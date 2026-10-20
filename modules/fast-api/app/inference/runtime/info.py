from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Which artifact this server runs: logged at startup, reported on /readyz and as the model_info metric."""

    name: str
    version: str
    sha256: str
    size_bytes: int
