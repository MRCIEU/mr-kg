"""Application configuration settings using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ---- Server configuration ----
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    HOST: str = Field(default="0.0.0.0", description="Host to bind server")
    PORT: int = Field(default=8000, description="Port to bind server")

    # ---- CORS configuration ----
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated allowed CORS origins",
    )

    @property
    def cors_origins(self) -> list[str]:
        """Convert comma-separated origins to list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    # ---- Database configuration ----
    DB_PROFILE: str = Field(
        default="local", description="Database profile (local or docker)"
    )
    VECTOR_STORE_PATH: str = Field(
        default="../data/db/vector_store.db",
        description="Path to vector store database",
    )
    TRAIT_PROFILE_PATH: str = Field(
        default="../data/db/trait_profile_db.db",
        description="Path to trait profile database",
    )

    # ---- API configuration ----
    API_V1_PREFIX: str = Field(
        default="/api/v1", description="API version 1 prefix"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
