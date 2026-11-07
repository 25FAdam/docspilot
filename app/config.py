"""Central configuration. Model names, thresholds and limits live only here / in env vars."""

from functools import lru_cache

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: SecretStr = SecretStr("")
    database_url: str = "postgresql://docspilot:docspilot@127.0.0.1:5432/docspilot"
    router_model: str = "gemini-2.5-flash-lite"
    synth_model: str = "gemini-2.5-flash"
    retrieval_threshold: float = Field(0.35, ge=0, le=1)
    daily_request_cap: int = Field(200, gt=0)
    max_output_tokens: int = Field(1024, gt=0)

    # Embeddings (local fastembed model, no API calls)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(384, gt=0)
    embed_batch_size: int = Field(64, gt=0)

    # Handbook source (GitLab API). HANDBOOK_PATHS is comma-separated (see handbook_path_list).
    handbook_project: str = "gitlab-com/content-sites/handbook"
    handbook_ref: str = "main"
    handbook_paths: str = (
        "content/handbook/engineering,content/handbook/people-group/general-onboarding"
    )
    handbook_max_files_total: int = Field(300, gt=0)
    handbook_base_url: str = "https://handbook.gitlab.com"
    raw_data_dir: str = "data/raw"

    # Chunking (token counts use the embedding model's tokenizer)
    chunk_min_tokens: int = Field(200, gt=0)
    chunk_max_tokens: int = Field(500, gt=0)
    chunk_overlap_ratio: float = Field(0.12, ge=0, le=0.5)

    # HTTP
    http_timeout_seconds: float = Field(30, gt=0)
    http_max_retries: int = Field(5, ge=0)
    http_retry_after_max_seconds: float = Field(120, gt=0)  # cap on a server's Retry-After

    @model_validator(mode="after")
    def _check_chunk_bounds(self) -> "Settings":
        if self.chunk_max_tokens <= self.chunk_min_tokens:
            raise ValueError("CHUNK_MAX_TOKENS must be greater than CHUNK_MIN_TOKENS")
        return self

    @property
    def handbook_path_list(self) -> list[str]:
        """HANDBOOK_PATHS split on commas; whitespace and empty items dropped, order kept."""
        return [p.strip() for p in self.handbook_paths.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
