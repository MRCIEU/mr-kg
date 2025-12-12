"""Configuration settings for repository database access.

Provides settings for database paths with environment variable support.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RepositorySettings(BaseSettings):
    """Repository configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vector_store_path: str = "data/db/vector_store.db"
    trait_profile_path: str = "data/db/trait_profile_db.db"
    evidence_profile_path: str = "data/db/evidence_profile_db.db"
    default_model: str = "gpt-5"


@lru_cache
def get_settings() -> RepositorySettings:
    """Get cached settings instance.

    Returns:
        RepositorySettings instance with database paths
    """
    return RepositorySettings()
