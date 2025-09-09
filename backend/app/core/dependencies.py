"""Dependency injection for database services and health checks."""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime

from fastapi import Depends, HTTPException, Request, status

from app.core.database import (
    get_database_pool,
    get_trait_profile_connection,
    get_vector_store_connection,
)
from app.core.schema_validation import DatabaseHealthChecker
from app.models.database import DatabaseStatus, HealthCheckResponse
from app.services.database_service import (
    AnalyticsService,
    EFOService,
    SimilarityService,
    StudyService,
    TraitService,
)

logger = logging.getLogger(__name__)


# ==== Request Dependencies ====


def get_request_id(request: Request) -> str | None:
    """Get request ID from request state."""
    return getattr(request.state, "request_id", None)


# ==== Service Dependencies ====


async def get_database_service():
    """Generic database service dependency.

    Returns the appropriate service based on what's requested.
    In practice, this should be replaced with specific service dependencies.
    """
    # This is a fallback - in practice the specific services should be used
    pool = await get_database_pool()
    async with pool.get_vector_store_connection() as vs_conn:
        async with pool.get_trait_profile_connection() as tp_conn:
            # Return a generic service that can be used for different purposes
            # This is not ideal but maintains compatibility
            from app.services.database_service import DatabaseService

            return DatabaseService(vs_conn, tp_conn)


async def get_trait_service(
    vector_store_conn=Depends(get_vector_store_connection),
    trait_profile_conn=Depends(get_trait_profile_connection),
) -> AsyncGenerator[TraitService, None]:
    """Get TraitService instance with database connections.

    Args:
        vector_store_conn: Vector store database connection
        trait_profile_conn: Trait profile database connection

    Yields:
        TraitService instance
    """
    try:
        service = TraitService(vector_store_conn, trait_profile_conn)
        yield service
    except Exception as e:
        logger.error(f"Error creating TraitService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


async def get_study_service(
    vector_store_conn=Depends(get_vector_store_connection),
    trait_profile_conn=Depends(get_trait_profile_connection),
) -> AsyncGenerator[StudyService, None]:
    """Get StudyService instance with database connections.

    Args:
        vector_store_conn: Vector store database connection
        trait_profile_conn: Trait profile database connection

    Yields:
        StudyService instance
    """
    try:
        service = StudyService(vector_store_conn, trait_profile_conn)
        yield service
    except Exception as e:
        logger.error(f"Error creating StudyService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


async def get_similarity_service(
    vector_store_conn=Depends(get_vector_store_connection),
    trait_profile_conn=Depends(get_trait_profile_connection),
) -> AsyncGenerator[SimilarityService, None]:
    """Get SimilarityService instance with database connections.

    Args:
        vector_store_conn: Vector store database connection
        trait_profile_conn: Trait profile database connection

    Yields:
        SimilarityService instance
    """
    try:
        service = SimilarityService(vector_store_conn, trait_profile_conn)
        yield service
    except Exception as e:
        logger.error(f"Error creating SimilarityService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


async def get_efo_service(
    vector_store_conn=Depends(get_vector_store_connection),
    trait_profile_conn=Depends(get_trait_profile_connection),
) -> AsyncGenerator[EFOService, None]:
    """Get EFOService instance with database connections.

    Args:
        vector_store_conn: Vector store database connection
        trait_profile_conn: Trait profile database connection

    Yields:
        EFOService instance
    """
    try:
        service = EFOService(vector_store_conn, trait_profile_conn)
        yield service
    except Exception as e:
        logger.error(f"Error creating EFOService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


async def get_analytics_service(
    vector_store_conn=Depends(get_vector_store_connection),
    trait_profile_conn=Depends(get_trait_profile_connection),
) -> AsyncGenerator[AnalyticsService, None]:
    """Get AnalyticsService instance with database connections.

    Args:
        vector_store_conn: Vector store database connection
        trait_profile_conn: Trait profile database connection

    Yields:
        AnalyticsService instance
    """
    try:
        service = AnalyticsService(vector_store_conn, trait_profile_conn)
        yield service
    except Exception as e:
        logger.error(f"Error creating AnalyticsService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e


# ==== Health Check Functions ====


async def get_health_checker(
    vector_store_conn=Depends(get_vector_store_connection),
    trait_profile_conn=Depends(get_trait_profile_connection),
) -> DatabaseHealthChecker:
    """Get DatabaseHealthChecker instance.

    Args:
        vector_store_conn: Vector store database connection
        trait_profile_conn: Trait profile database connection

    Returns:
        DatabaseHealthChecker instance
    """
    try:
        return DatabaseHealthChecker(vector_store_conn, trait_profile_conn)
    except Exception as e:
        logger.error(f"Error creating DatabaseHealthChecker: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check service unavailable",
        ) from e


async def perform_database_health_check() -> HealthCheckResponse:
    """Perform comprehensive database health check.

    Returns:
        HealthCheckResponse with complete health status
    """
    try:
        # Get database pool and connections
        pool = await get_database_pool()

        # Get connections for health check
        async with pool.get_vector_store_connection() as vs_conn:
            async with pool.get_trait_profile_connection() as tp_conn:
                # Create health checker
                health_checker = DatabaseHealthChecker(vs_conn, tp_conn)

                # Perform health check
                health_data = await health_checker.perform_health_check()

                # Convert to response format
                return HealthCheckResponse(
                    status=health_data["overall_status"],
                    timestamp=datetime.fromtimestamp(health_data["timestamp"]),
                    vector_store=DatabaseStatus(
                        database_path=health_data["vector_store"][
                            "database_path"
                        ],
                        accessible=health_data["vector_store"]["accessible"],
                        table_count=len(health_data["vector_store"]["tables"]),
                        view_count=len(health_data["vector_store"]["views"]),
                        index_count=len(health_data["vector_store"]["indexes"]),
                        last_checked=datetime.fromtimestamp(
                            health_data["timestamp"]
                        ),
                        error=health_data["vector_store"].get("error"),
                    ),
                    trait_profile=DatabaseStatus(
                        database_path=health_data["trait_profile"][
                            "database_path"
                        ],
                        accessible=health_data["trait_profile"]["accessible"],
                        table_count=len(health_data["trait_profile"]["tables"]),
                        view_count=len(health_data["trait_profile"]["views"]),
                        index_count=len(
                            health_data["trait_profile"]["indexes"]
                        ),
                        last_checked=datetime.fromtimestamp(
                            health_data["timestamp"]
                        ),
                        error=health_data["trait_profile"].get("error"),
                    ),
                    performance_metrics=health_data["performance_metrics"],
                )

    except Exception as e:
        logger.error(f"Error performing health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        ) from e


async def check_database_connectivity() -> dict:
    """Quick database connectivity check for startup validation.

    Returns:
        Dictionary with connectivity status
    """
    try:
        pool = await get_database_pool()
        pool_status = await pool.get_pool_status()

        # Test basic queries
        async with pool.get_vector_store_connection() as vs_conn:
            vs_result = vs_conn.execute("SELECT 1 as test").fetchone()

        async with pool.get_trait_profile_connection() as tp_conn:
            tp_result = tp_conn.execute("SELECT 1 as test").fetchone()

        return {
            "status": "healthy",
            "vector_store_connection": vs_result is not None
            and vs_result[0] == 1,
            "trait_profile_connection": tp_result is not None
            and tp_result[0] == 1,
            "pool_status": pool_status,
        }

    except Exception as e:
        logger.error(f"Database connectivity check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "vector_store_connection": False,
            "trait_profile_connection": False,
        }


# ==== Connection Pool Management ====


async def get_pool_status() -> dict:
    """Get current database connection pool status.

    Returns:
        Dictionary with pool status information
    """
    try:
        pool = await get_database_pool()
        return await pool.get_pool_status()
    except Exception as e:
        logger.error(f"Error getting pool status: {e}")
        return {"error": str(e), "initialized": False}


# ==== Error Handling Utilities ====


def handle_database_error(error: Exception, operation: str) -> HTTPException:
    """Handle database errors and convert to appropriate HTTP exceptions.

    Args:
        error: The database error
        operation: Description of the operation that failed

    Returns:
        HTTPException with appropriate status code and message
    """
    logger.error(f"Database error in {operation}: {error}")

    error_msg = str(error).lower()

    if "connection" in error_msg or "database" in error_msg:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection error during {operation}",
        )
    elif "not found" in error_msg or "no such" in error_msg:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource not found during {operation}",
        )
    elif "timeout" in error_msg:
        return HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Database timeout during {operation}",
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal database error during {operation}",
        )


# ==== Validation Dependencies ====


def validate_trait_index(trait_index: int) -> int:
    """Validate trait index parameter.

    Args:
        trait_index: Trait index to validate

    Returns:
        Validated trait index

    Raises:
        HTTPException: If trait index is invalid
    """
    if trait_index < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trait index must be non-negative",
        )
    return trait_index


def validate_study_id(study_id: int) -> int:
    """Validate study ID parameter.

    Args:
        study_id: Study ID to validate

    Returns:
        Validated study ID

    Raises:
        HTTPException: If study ID is invalid
    """
    if study_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Study ID must be positive",
        )
    return study_id


def validate_pmid(pmid: str) -> str:
    """Validate PMID parameter.

    Args:
        pmid: PMID to validate

    Returns:
        Validated PMID

    Raises:
        HTTPException: If PMID is invalid
    """
    if not pmid or not pmid.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PMID cannot be empty",
        )
    pmid = pmid.strip()
    if not pmid.isdigit():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PMID must be numeric",
        )
    return pmid


def validate_similarity_threshold(threshold: float) -> float:
    """Validate similarity threshold parameter.

    Args:
        threshold: Similarity threshold to validate

    Returns:
        Validated similarity threshold

    Raises:
        HTTPException: If threshold is invalid
    """
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Similarity threshold must be between 0.0 and 1.0",
        )
    return threshold
