import json

from pydantic_settings import BaseSettings


class EnvSettings(BaseSettings):
    """Settings read from environment variables. `to_env` turns them back into those variables, nested settings
    included: the whole environment for a child process, such as a gunicorn the benchmarks start."""

    def to_env(self) -> dict[str, str]:
        prefix = self.model_config.get("env_prefix", "")
        values = self.model_dump(mode="json")

        env: dict[str, str] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)

            if isinstance(value, EnvSettings):
                env |= value.to_env()
            elif value is not None:
                rendered = values[name]
                env[f"{prefix}{name}".upper()] = rendered if isinstance(rendered, str) else json.dumps(rendered)

        return env
