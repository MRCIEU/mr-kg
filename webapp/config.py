"""Configuration settings for the webapp."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Webapp configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database paths
    vector_store_path: str = "data/db/vector_store.db"
    trait_profile_path: str = "data/db/trait_profile_db.db"
    evidence_profile_path: str = "data/db/evidence_profile_db.db"

    # Default model
    default_model: str = "gpt-5"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
