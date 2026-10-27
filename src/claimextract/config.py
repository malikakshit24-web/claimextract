"""Settings, read from environment variables (or a local .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAIMEXTRACT_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://claimextract:claimextract@localhost:5432/claimextract"
    default_extractor: str = "ocr"
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
