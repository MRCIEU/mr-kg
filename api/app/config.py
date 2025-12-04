"""Configuration settings for the API service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="")

    vector_store_path: str = "data/db/vector_store.db"
    trait_profile_path: str = "data/db/trait_profile_db.db"
    evidence_profile_path: str = "data/db/evidence_profile_db.db"
    default_model: str = "gpt-5"
    log_level: str = "INFO"
    cache_ttl_autocomplete: int = 300
    cache_ttl_data: int = 3600


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
