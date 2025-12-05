"""Database connection management for MR-KG repositories.

Provides thread-safe read-only connections to DuckDB databases.
Each call creates a new connection to ensure thread safety with concurrent
requests.
"""

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from common_funcs.repositories.config import get_settings


class DatabaseError(Exception):
    """Exception raised for database connection errors."""

    pass


# Thread-local storage for connections
_thread_local = threading.local()


def _get_connection(db_path: Path, db_name: str) -> duckdb.DuckDBPyConnection:
    """Get a read-only connection for the current thread.

    Creates a new connection per thread to ensure thread safety.

    Args:
        db_path: Path to the database file
        db_name: Name of the database (for error messages and caching)

    Returns:
        Read-only DuckDB connection

    Raises:
        DatabaseError: If database file not found or connection fails
    """
    if not db_path.exists():
        raise DatabaseError(f"{db_name} database not found: {db_path}")

    # Check if we have a connection for this thread and database
    cache_key = f"{db_name}_connection"
    conn = getattr(_thread_local, cache_key, None)

    if conn is None:
        try:
            conn = duckdb.connect(str(db_path), read_only=True)
            setattr(_thread_local, cache_key, conn)
        except Exception as e:
            raise DatabaseError(
                f"Failed to connect to {db_name} database: {e}"
            ) from e

    return conn


def get_vector_store_connection() -> duckdb.DuckDBPyConnection:
    """Get read-only connection to vector store database.

    Returns:
        Read-only DuckDB connection to vector_store.db

    Raises:
        DatabaseError: If database file not found or connection fails
    """
    settings = get_settings()
    db_path = Path(settings.vector_store_path)
    return _get_connection(db_path, "Vector store")


def get_trait_profile_connection() -> duckdb.DuckDBPyConnection:
    """Get read-only connection to trait profile database.

    Returns:
        Read-only DuckDB connection to trait_profile_db.db

    Raises:
        DatabaseError: If database file not found or connection fails
    """
    settings = get_settings()
    db_path = Path(settings.trait_profile_path)
    return _get_connection(db_path, "Trait profile")


def get_evidence_profile_connection() -> duckdb.DuckDBPyConnection:
    """Get read-only connection to evidence profile database.

    Returns:
        Read-only DuckDB connection to evidence_profile_db.db

    Raises:
        DatabaseError: If database file not found or connection fails
    """
    settings = get_settings()
    db_path = Path(settings.evidence_profile_path)
    return _get_connection(db_path, "Evidence profile")


@contextmanager
def vector_store_connection() -> Generator[
    duckdb.DuckDBPyConnection, None, None
]:
    """Context manager for vector store database connection.

    Creates a new connection that is closed when the context exits.
    Use this when you need explicit connection lifecycle control.

    Yields:
        Read-only DuckDB connection
    """
    settings = get_settings()
    db_path = Path(settings.vector_store_path)

    if not db_path.exists():
        raise DatabaseError(f"Vector store database not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def trait_profile_connection() -> Generator[
    duckdb.DuckDBPyConnection, None, None
]:
    """Context manager for trait profile database connection.

    Creates a new connection that is closed when the context exits.
    Use this when you need explicit connection lifecycle control.

    Yields:
        Read-only DuckDB connection
    """
    settings = get_settings()
    db_path = Path(settings.trait_profile_path)

    if not db_path.exists():
        raise DatabaseError(f"Trait profile database not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def evidence_profile_connection() -> Generator[
    duckdb.DuckDBPyConnection, None, None
]:
    """Context manager for evidence profile database connection.

    Creates a new connection that is closed when the context exits.
    Use this when you need explicit connection lifecycle control.

    Yields:
        Read-only DuckDB connection
    """
    settings = get_settings()
    db_path = Path(settings.evidence_profile_path)

    if not db_path.exists():
        raise DatabaseError(f"Evidence profile database not found: {db_path}")

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        yield conn
    finally:
        conn.close()


def close_all_connections() -> None:
    """Close all cached database connections for the current thread.

    Useful for cleanup during testing or application shutdown.
    """
    for db_name in ["Vector store", "Trait profile", "Evidence profile"]:
        cache_key = f"{db_name}_connection"
        conn = getattr(_thread_local, cache_key, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            delattr(_thread_local, cache_key)
