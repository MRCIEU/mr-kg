"""Configuration settings for the webapp."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Webapp configuration settings."""

    model_config = SettingsConfigDict(env_prefix="")

    api_url: str = "http://localhost:8000"
    default_model: str = "gpt-5"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
