"""Configuration settings for the FastAPI backend."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Profile settings
    profile: Literal["local", "docker"] = "local"
    
    # Database paths (will be set based on profile)
    vector_store_db_path: str = ""
    trait_profile_db_path: str = ""
    
    # CORS settings
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    class Config:
        env_file = ".env"
        env_prefix = "BACKEND_"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_database_paths()
    
    def _set_database_paths(self):
        """Set database paths based on profile."""
        if self.profile == "docker":
            # Docker paths
            self.vector_store_db_path = "/app/data/db/vector_store.db"
            self.trait_profile_db_path = "/app/data/db/trait_profile_db.db"
        else:
            # Local paths - find project root and set paths
            current_dir = Path(__file__).parent
            project_root = self._find_project_root(current_dir)
            
            if project_root:
                self.vector_store_db_path = str(project_root / "data" / "db" / "vector_store.db")
                self.trait_profile_db_path = str(project_root / "data" / "db" / "trait_profile_db.db")
            else:
                raise RuntimeError("Could not find project root directory")
    
    def _find_project_root(self, start_path: Path) -> Path | None:
        """Find project root by looking for specific marker files."""
        current = start_path
        while current != current.parent:
            if (current / "data" / "db").exists():
                return current
            current = current.parent
        return None


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
