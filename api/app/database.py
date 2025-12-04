"""Database connection management for MR-KG API.

Provides read-only connections to DuckDB databases with connection caching.
"""

from functools import lru_cache
from pathlib import Path

import duckdb

from app.config import get_settings


class DatabaseError(Exception):
    """Exception raised for database connection errors."""

    pass


@lru_cache(maxsize=1)
def get_vector_store_connection() -> duckdb.DuckDBPyConnection:
    """Get read-only connection to vector store database.

    Returns:
        Read-only DuckDB connection to vector_store.db

    Raises:
        DatabaseError: If database file not found or connection fails
    """
    settings = get_settings()
    db_path = Path(settings.vector_store_path)

    if not db_path.exists():
        raise DatabaseError(f"Vector store database not found: {db_path}")

    try:
        res = duckdb.connect(str(db_path), read_only=True)
        return res
    except Exception as e:
        raise DatabaseError(
            f"Failed to connect to vector store database: {e}"
        ) from e


@lru_cache(maxsize=1)
def get_trait_profile_connection() -> duckdb.DuckDBPyConnection:
    """Get read-only connection to trait profile database.

    Returns:
        Read-only DuckDB connection to trait_profile_db.db

    Raises:
        DatabaseError: If database file not found or connection fails
    """
    settings = get_settings()
    db_path = Path(settings.trait_profile_path)

    if not db_path.exists():
        raise DatabaseError(f"Trait profile database not found: {db_path}")

    try:
        res = duckdb.connect(str(db_path), read_only=True)
        return res
    except Exception as e:
        raise DatabaseError(
            f"Failed to connect to trait profile database: {e}"
        ) from e


@lru_cache(maxsize=1)
def get_evidence_profile_connection() -> duckdb.DuckDBPyConnection:
    """Get read-only connection to evidence profile database.

    Returns:
        Read-only DuckDB connection to evidence_profile_db.db

    Raises:
        DatabaseError: If database file not found or connection fails
    """
    settings = get_settings()
    db_path = Path(settings.evidence_profile_path)

    if not db_path.exists():
        raise DatabaseError(f"Evidence profile database not found: {db_path}")

    try:
        res = duckdb.connect(str(db_path), read_only=True)
        return res
    except Exception as e:
        raise DatabaseError(
            f"Failed to connect to evidence profile database: {e}"
        ) from e


def close_all_connections() -> None:
    """Close all cached database connections.

    Clears the connection cache and closes any open connections.
    Useful for cleanup during testing or application shutdown.
    """
    get_vector_store_connection.cache_clear()
    get_trait_profile_connection.cache_clear()
    get_evidence_profile_connection.cache_clear()
