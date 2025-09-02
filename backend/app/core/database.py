"""Database configuration and connection management for DuckDB vector stores."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

import duckdb
from pydantic import BaseModel

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    profile: str
    vector_store_path: Path
    trait_profile_path: Path
    max_connections: int = 10
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


class DatabaseConnectionPool:
    """Connection pool manager for DuckDB databases."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._vector_store_connections: list[duckdb.DuckDBPyConnection] = []
        self._trait_profile_connections: list[duckdb.DuckDBPyConnection] = []
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pools with health checks."""
        async with self._lock:
            if self._initialized:
                return

            logger.info(
                f"Initializing database pools with profile: {self.config.profile}"
            )

            # Validate database files exist
            await self._validate_database_files()

            # Create initial connections
            await self._create_initial_connections()

            # Perform health checks
            await self._perform_health_checks()

            self._initialized = True
            logger.info("Database pools initialized successfully")

    async def _validate_database_files(self) -> None:
        """Validate that database files exist and are accessible."""
        for db_name, path in [
            ("vector_store", self.config.vector_store_path),
            ("trait_profile", self.config.trait_profile_path),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{db_name} database not found at: {path}"
                )
            if not path.is_file():
                raise ValueError(
                    f"{db_name} database path is not a file: {path}"
                )
            logger.info(f"Validated {db_name} database at: {path}")

    async def _create_initial_connections(self) -> None:
        """Create initial connections for both databases."""
        # Create vector store connections
        for _ in range(min(3, self.config.max_connections)):
            conn = duckdb.connect(
                str(self.config.vector_store_path), read_only=True
            )
            self._vector_store_connections.append(conn)

        # Create trait profile connections
        for _ in range(min(3, self.config.max_connections)):
            conn = duckdb.connect(
                str(self.config.trait_profile_path), read_only=True
            )
            self._trait_profile_connections.append(conn)

        logger.info(
            f"Created {len(self._vector_store_connections)} vector store connections"
        )
        logger.info(
            f"Created {len(self._trait_profile_connections)} trait profile connections"
        )

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all connections."""
        # Check vector store connections
        if self._vector_store_connections:
            conn = self._vector_store_connections[0]
            result = conn.execute("SELECT 1 as health_check").fetchone()
            if result[0] != 1:
                raise RuntimeError("Vector store database health check failed")

        # Check trait profile connections
        if self._trait_profile_connections:
            conn = self._trait_profile_connections[0]
            result = conn.execute("SELECT 1 as health_check").fetchone()
            if result[0] != 1:
                raise RuntimeError("Trait profile database health check failed")

        logger.info("All database health checks passed")

    @asynccontextmanager
    async def get_vector_store_connection(
        self,
    ) -> AsyncGenerator[duckdb.DuckDBPyConnection, None]:
        """Get a connection to the vector store database with automatic cleanup."""
        if not self._initialized:
            await self.initialize()

        connection = None
        try:
            async with self._lock:
                if self._vector_store_connections:
                    connection = self._vector_store_connections.pop()
                else:
                    # Create new connection if pool is empty
                    connection = duckdb.connect(
                        str(self.config.vector_store_path), read_only=True
                    )

            yield connection

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            # Don't return potentially corrupted connection to pool
            if connection:
                connection.close()
                connection = None
            raise
        finally:
            if connection:
                async with self._lock:
                    if (
                        len(self._vector_store_connections)
                        < self.config.max_connections
                    ):
                        self._vector_store_connections.append(connection)
                    else:
                        connection.close()

    @asynccontextmanager
    async def get_trait_profile_connection(
        self,
    ) -> AsyncGenerator[duckdb.DuckDBPyConnection, None]:
        """Get a connection to the trait profile database with automatic cleanup."""
        if not self._initialized:
            await self.initialize()

        connection = None
        try:
            async with self._lock:
                if self._trait_profile_connections:
                    connection = self._trait_profile_connections.pop()
                else:
                    # Create new connection if pool is empty
                    connection = duckdb.connect(
                        str(self.config.trait_profile_path), read_only=True
                    )

            yield connection

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            # Don't return potentially corrupted connection to pool
            if connection:
                connection.close()
                connection = None
            raise
        finally:
            if connection:
                async with self._lock:
                    if (
                        len(self._trait_profile_connections)
                        < self.config.max_connections
                    ):
                        self._trait_profile_connections.append(connection)
                    else:
                        connection.close()

    async def close_all_connections(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            # Close vector store connections
            for conn in self._vector_store_connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(
                        f"Error closing vector store connection: {e}"
                    )
            self._vector_store_connections.clear()

            # Close trait profile connections
            for conn in self._trait_profile_connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(
                        f"Error closing trait profile connection: {e}"
                    )
            self._trait_profile_connections.clear()

            self._initialized = False
            logger.info("All database connections closed")

    async def get_pool_status(self) -> Dict[str, int]:
        """Get current pool status for monitoring."""
        async with self._lock:
            return {
                "vector_store_connections": len(self._vector_store_connections),
                "trait_profile_connections": len(
                    self._trait_profile_connections
                ),
                "max_connections": self.config.max_connections,
                "initialized": self._initialized,
            }


def _resolve_database_paths(profile: str) -> tuple[Path, Path]:
    """Resolve database paths based on profile configuration.

    Args:
        profile: Database profile (local or docker)

    Returns:
        Tuple of (vector_store_path, trait_profile_path)
    """
    if profile == "local":
        # Use settings paths for local development
        vector_store_path = Path(settings.VECTOR_STORE_PATH)
        trait_profile_path = Path(settings.TRAIT_PROFILE_PATH)

        # Make paths absolute if they're relative
        if not vector_store_path.is_absolute():
            vector_store_path = Path.cwd() / vector_store_path
        if not trait_profile_path.is_absolute():
            trait_profile_path = Path.cwd() / trait_profile_path
    else:
        # Docker environment - use container paths
        vector_store_path = Path("/app/data/db/vector_store.db")
        trait_profile_path = Path("/app/data/db/trait_profile_db.db")

    return vector_store_path, trait_profile_path


def create_database_config(
    profile: Optional[str] = None,
    max_connections: int = 10,
    connection_timeout: float = 30.0,
) -> DatabaseConfig:
    """Create database configuration from settings and parameters.

    Args:
        profile: Database profile, defaults to settings.DB_PROFILE
        max_connections: Maximum connections in pool
        connection_timeout: Connection timeout in seconds

    Returns:
        DatabaseConfig instance
    """
    resolved_profile = profile if profile is not None else settings.DB_PROFILE

    vector_store_path, trait_profile_path = _resolve_database_paths(
        resolved_profile
    )

    return DatabaseConfig(
        profile=resolved_profile,
        vector_store_path=vector_store_path,
        trait_profile_path=trait_profile_path,
        max_connections=max_connections,
        connection_timeout=connection_timeout,
    )


# Global connection pool instance
_connection_pool: Optional[DatabaseConnectionPool] = None


async def get_database_pool() -> DatabaseConnectionPool:
    """Get or create the global database connection pool.

    Returns:
        DatabaseConnectionPool instance
    """
    global _connection_pool

    if _connection_pool is None:
        config = create_database_config()
        _connection_pool = DatabaseConnectionPool(config)
        await _connection_pool.initialize()

    return _connection_pool


async def close_database_pool() -> None:
    """Close the global database connection pool."""
    global _connection_pool

    if _connection_pool is not None:
        await _connection_pool.close_all_connections()
        _connection_pool = None


# Dependency injection functions for FastAPI
async def get_vector_store_connection():
    """FastAPI dependency for vector store database connections."""
    pool = await get_database_pool()
    async with pool.get_vector_store_connection() as conn:
        yield conn


async def get_trait_profile_connection():
    """FastAPI dependency for trait profile database connections."""
    pool = await get_database_pool()
    async with pool.get_trait_profile_connection() as conn:
        yield conn
