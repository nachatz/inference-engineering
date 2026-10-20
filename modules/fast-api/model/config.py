from pydantic_settings import SettingsConfigDict

from app.common.settings import EnvSettings


class ExportSettings(EnvSettings):
    """The export and its check: `EXPORT_*`. The check runs every `verify_batch_sizes`, plus the server's
    MODEL_MAX_BATCH, the largest batch it takes, within `tolerance` of PyTorch."""

    model_config = SettingsConfigDict(env_prefix="EXPORT_")

    seed: int = 0
    tolerance: float = 1e-4
    verify_batch_sizes: tuple[int, ...] = (1, 2, 7, 32)


export_settings = ExportSettings()
