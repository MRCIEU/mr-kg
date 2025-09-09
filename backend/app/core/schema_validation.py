"""Database schema integration and validation utilities."""

import logging

# Import schema definitions from common_funcs
import sys
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel

sys.path.append(
    str(Path(__file__).parent.parent.parent.parent / "src" / "common_funcs")
)

from common_funcs.schema.database_schema import (
    DATABASE_INDEXES,
    DATABASE_SCHEMA,
    DATABASE_VIEWS,
)
from common_funcs.schema.trait_profile_schema import (
    TRAIT_PROFILE_INDEXES,
    TRAIT_PROFILE_SCHEMA,
    TRAIT_PROFILE_VIEWS,
)

logger = logging.getLogger(__name__)


class DatabaseTableInfo(BaseModel):
    """Information about a database table."""

    name: str
    exists: bool
    row_count: int | None = None
    columns: list[str] = []
    error: str | None = None


class DatabaseSchemaStatus(BaseModel):
    """Overall database schema status."""

    database_path: str
    accessible: bool
    tables: list[DatabaseTableInfo] = []
    views: list[str] = []
    indexes: list[str] = []
    error: str | None = None


class SchemaValidator:
    """Validates database schema against expected definitions."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection

    def validate_vector_store_schema(self) -> DatabaseSchemaStatus:
        """Validate the vector store database schema.

        Returns:
            DatabaseSchemaStatus with validation results
        """
        try:
            # Check basic connectivity
            self.connection.execute("SELECT 1").fetchone()

            status = DatabaseSchemaStatus(
                database_path=str(
                    (
                        self.connection.execute(
                            "PRAGMA database_list"
                        ).fetchone()
                        or [None, None, "unknown"]
                    )[2]
                ),
                accessible=True,
            )

            # Validate tables
            for table_name, table_def in DATABASE_SCHEMA.items():
                table_info = self._validate_table(table_name, table_def.columns)
                status.tables.append(table_info)

            # Validate views
            for view_def in DATABASE_VIEWS:
                if self._view_exists(view_def.name):
                    status.views.append(view_def.name)

            # Validate indexes
            for index_def in DATABASE_INDEXES:
                if self._index_exists(index_def.name):
                    status.indexes.append(index_def.name)

            return status

        except Exception as e:
            logger.error(f"Error validating vector store schema: {e}")
            return DatabaseSchemaStatus(
                database_path="unknown", accessible=False, error=str(e)
            )

    def validate_trait_profile_schema(self) -> DatabaseSchemaStatus:
        """Validate the trait profile database schema.

        Returns:
            DatabaseSchemaStatus with validation results
        """
        try:
            # Check basic connectivity
            self.connection.execute("SELECT 1").fetchone()

            status = DatabaseSchemaStatus(
                database_path=str(
                    (
                        self.connection.execute(
                            "PRAGMA database_list"
                        ).fetchone()
                        or [None, None, "unknown"]
                    )[2]
                ),
                accessible=True,
            )

            # Validate tables
            for table_name, table_def in TRAIT_PROFILE_SCHEMA.items():
                table_info = self._validate_table(table_name, table_def.columns)
                status.tables.append(table_info)

            # Validate views
            for view_def in TRAIT_PROFILE_VIEWS:
                if self._view_exists(view_def.name):
                    status.views.append(view_def.name)

            # Validate indexes
            for index_def in TRAIT_PROFILE_INDEXES:
                if self._index_exists(index_def.name):
                    status.indexes.append(index_def.name)

            return status

        except Exception as e:
            logger.error(f"Error validating trait profile schema: {e}")
            return DatabaseSchemaStatus(
                database_path="unknown", accessible=False, error=str(e)
            )

    def _validate_table(
        self, table_name: str, expected_columns
    ) -> DatabaseTableInfo:
        """Validate a single table exists and has expected structure.

        Args:
            table_name: Name of the table to validate
            expected_columns: Expected column definitions

        Returns:
            DatabaseTableInfo with validation results
        """
        try:
            # Check if table exists
            result = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()

            if not result:
                return DatabaseTableInfo(
                    name=table_name, exists=False, error="Table does not exist"
                )

            # Get table info
            columns_info = self.connection.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
            actual_columns = [
                col[1] for col in columns_info
            ]  # col[1] is column name

            # Get row count
            count_result = self.connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()
            row_count = count_result[0] if count_result else 0

            return DatabaseTableInfo(
                name=table_name,
                exists=True,
                row_count=row_count,
                columns=actual_columns,
            )

        except Exception as e:
            logger.error(f"Error validating table {table_name}: {e}")
            return DatabaseTableInfo(
                name=table_name, exists=False, error=str(e)
            )

    def _view_exists(self, view_name: str) -> bool:
        """Check if a view exists.

        Args:
            view_name: Name of the view to check

        Returns:
            True if view exists, False otherwise
        """
        try:
            result = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='view' AND name=?",
                (view_name,),
            ).fetchone()
            return result is not None
        except Exception:
            return False

    def _index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            result = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
                (index_name,),
            ).fetchone()
            return result is not None
        except Exception:
            return False


class DatabaseHealthChecker:
    """Performs comprehensive database health checks."""

    def __init__(
        self,
        vector_store_conn: duckdb.DuckDBPyConnection,
        trait_profile_conn: duckdb.DuckDBPyConnection,
    ):
        self.vector_store_conn = vector_store_conn
        self.trait_profile_conn = trait_profile_conn

    async def perform_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check on both databases.

        Returns:
            Dictionary with health check results
        """
        health_status = {
            "timestamp": None,
            "overall_status": "unknown",
            "vector_store": {},
            "trait_profile": {},
            "performance_metrics": {},
        }

        try:
            import time

            start_time = time.time()
            health_status["timestamp"] = start_time

            # Validate vector store
            vs_validator = SchemaValidator(self.vector_store_conn)
            vs_schema = vs_validator.validate_vector_store_schema()
            health_status["vector_store"] = vs_schema.model_dump()

            # Validate trait profile
            tp_validator = SchemaValidator(self.trait_profile_conn)
            tp_schema = tp_validator.validate_trait_profile_schema()
            health_status["trait_profile"] = tp_schema.model_dump()

            # Performance tests
            perf_metrics = await self._run_performance_tests()
            health_status["performance_metrics"] = perf_metrics

            # Determine overall status
            vs_healthy = vs_schema.accessible and len(vs_schema.tables) > 0
            tp_healthy = tp_schema.accessible and len(tp_schema.tables) > 0

            # Check for errors in components
            errors = []
            if vs_schema.error:
                errors.append(f"Vector store: {vs_schema.error}")
            if tp_schema.error:
                errors.append(f"Trait profile: {tp_schema.error}")
            if isinstance(perf_metrics, dict) and perf_metrics.get("error"):
                errors.append(f"Performance: {perf_metrics['error']}")

            if vs_healthy and tp_healthy:
                health_status["overall_status"] = "healthy"
            elif vs_healthy or tp_healthy:
                health_status["overall_status"] = "partial"
            else:
                health_status["overall_status"] = "unhealthy"

            # Add top-level error if any component has errors
            if errors:
                health_status["error"] = "; ".join(errors)

            return health_status

        except Exception as e:
            logger.error(f"Error during health check: {e}")
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)
            return health_status

    async def _run_performance_tests(self) -> dict[str, Any]:
        """Run basic performance tests on databases.

        Returns:
            Dictionary with performance metrics
        """
        import time

        metrics = {}

        try:
            # Test vector store query performance
            start = time.time()
            self.vector_store_conn.execute(
                "SELECT COUNT(*) FROM trait_embeddings"
            ).fetchone()
            metrics["vector_store_query_time"] = time.time() - start

            # Test trait profile query performance
            start = time.time()
            self.trait_profile_conn.execute(
                "SELECT COUNT(*) FROM query_combinations"
            ).fetchone()
            metrics["trait_profile_query_time"] = time.time() - start

            # Test complex query performance
            start = time.time()
            self.vector_store_conn.execute("""
                SELECT t.trait_label, COUNT(*) as study_count
                FROM trait_embeddings t
                JOIN model_result_traits mrt ON t.trait_index = mrt.trait_index
                GROUP BY t.trait_label
                LIMIT 10
            """).fetchall()
            metrics["complex_query_time"] = time.time() - start

        except Exception as e:
            logger.warning(f"Error running performance tests: {e}")
            metrics["error"] = str(e)

        return metrics


def get_schema_validator(
    connection: duckdb.DuckDBPyConnection, db_type: str
) -> SchemaValidator:
    """Get appropriate schema validator for database type.

    Args:
        connection: Database connection
        db_type: Type of database ('vector_store' or 'trait_profile')

    Returns:
        SchemaValidator instance
    """
    return SchemaValidator(connection)


def validate_database_structure(
    db_path: str, db_type: str
) -> DatabaseSchemaStatus:
    """Validate database structure without using connection pool.

    Args:
        db_path: Path to database file
        db_type: Type of database ('vector_store' or 'trait_profile')

    Returns:
        DatabaseSchemaStatus with validation results
    """
    conn = None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        validator = SchemaValidator(conn)

        if db_type == "vector_store":
            return validator.validate_vector_store_schema()
        elif db_type == "trait_profile":
            return validator.validate_trait_profile_schema()
        else:
            raise ValueError(f"Unknown database type: {db_type}")

    except Exception as e:
        logger.error(f"Error validating database {db_path}: {e}")
        return DatabaseSchemaStatus(
            database_path=db_path, accessible=False, error=str(e)
        )
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
