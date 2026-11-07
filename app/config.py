"""Central configuration. Model names, thresholds and limits live only here / in env vars."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: SecretStr = SecretStr("")
    database_url: str = "postgresql://docspilot:docspilot@localhost:5432/docspilot"
    router_model: str = "gemini-2.5-flash-lite"
    synth_model: str = "gemini-2.5-flash"
    retrieval_threshold: float = Field(0.35, ge=0, le=1)
    daily_request_cap: int = Field(200, gt=0)
    max_output_tokens: int = Field(1024, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
