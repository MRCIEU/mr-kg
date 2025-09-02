"""Health check endpoints for monitoring and diagnostics."""

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import settings
from app.core.dependencies import (
    check_database_connectivity,
    get_database_service,
    get_pool_status,
    perform_database_health_check,
)
from app.models.database import HealthCheckResponse
from app.models.responses import (
    DataResponse,
    HealthStatus,
    MetricsInfo,
    SystemInfo,
)
from app.services.database_service import DatabaseService

router = APIRouter()

# Global metrics storage (use Redis or proper metrics backend in production)
_app_start_time = time.time()
_request_metrics = {
    "total_requests": 0,
    "database_queries": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}


@router.get("/", response_model=DataResponse[HealthStatus])
async def basic_health_check(request: Request) -> DataResponse[HealthStatus]:
    """Basic health check endpoint for load balancers and monitoring.

    This endpoint provides a simple health status without heavy operations.
    """
    uptime = time.time() - _app_start_time

    health_data = HealthStatus(
        status="healthy",
        version="0.1.0",
        uptime=uptime,
        timestamp=datetime.utcnow(),
    )

    return DataResponse(data=health_data)


@router.get("/detailed", response_model=HealthCheckResponse)
async def detailed_health_check() -> HealthCheckResponse:
    """Comprehensive health check including database connectivity.

    This endpoint performs thorough health checks including database
    connectivity, schema validation, and performance metrics.
    """
    try:
        return await perform_database_health_check()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        )


@router.get("/database", response_model=DataResponse[dict[str, Any]])
async def database_health_check() -> DataResponse[dict[str, Any]]:
    """Quick database connectivity check.

    Returns basic connectivity status for both vector store and trait
    profile databases without performance metrics.
    """
    try:
        connectivity_data = await check_database_connectivity()
        return DataResponse(data=connectivity_data)
    except Exception as e:
        # Return structured error instead of raising exception
        error_data = {
            "status": "error",
            "error": str(e),
            "vector_store_connection": False,
            "trait_profile_connection": False,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return DataResponse(data=error_data)


@router.get("/pool", response_model=DataResponse[dict[str, Any]])
async def connection_pool_status() -> DataResponse[dict[str, Any]]:
    """Get database connection pool status and metrics."""
    try:
        pool_data = await get_pool_status()
        return DataResponse(data=pool_data)
    except Exception as e:
        error_data = {
            "error": str(e),
            "initialized": False,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return DataResponse(data=error_data)


@router.get("/system", response_model=DataResponse[SystemInfo])
async def system_info() -> DataResponse[SystemInfo]:
    """Get system information and capabilities."""
    features = [
        "vector_search",
        "trait_analysis",
        "similarity_computation",
        "database_health_monitoring",
        "api_versioning",
        "request_logging",
        "error_handling",
    ]

    system_data = SystemInfo(
        application="MR-KG API",
        version="0.1.0",
        environment="development" if settings.DEBUG else "production",
        api_version="v1",
        database_profile=settings.DB_PROFILE,
        features=features,
    )

    return DataResponse(data=system_data)


@router.get("/metrics", response_model=DataResponse[MetricsInfo])
async def application_metrics() -> DataResponse[MetricsInfo]:
    """Get application performance metrics.

    Note: This is a basic implementation. In production, use proper
    metrics collection systems like Prometheus.
    """
    # Get metrics from middleware if available
    metrics_data = MetricsInfo(
        requests_total=_request_metrics["total_requests"],
        active_connections=0,  # Would come from connection pool
        database_queries=_request_metrics["database_queries"],
        cache_hits=_request_metrics["cache_hits"],
        cache_misses=_request_metrics["cache_misses"],
        response_time_avg=0.0,  # Would come from middleware
    )

    return DataResponse(data=metrics_data)


@router.get("/ready", response_model=DataResponse[dict[str, Any]])
async def readiness_check(
    db_service: DatabaseService = Depends(get_database_service),
) -> DataResponse[dict[str, Any]]:
    """Kubernetes-style readiness probe.

    Checks if the application is ready to serve traffic by verifying
    all dependencies are available and functional.
    """
    checks = {}
    overall_ready = True

    # Check database connectivity
    try:
        connectivity = await check_database_connectivity()
        checks["database"] = {
            "ready": connectivity["status"] == "healthy",
            "details": connectivity,
        }
        if connectivity["status"] != "healthy":
            overall_ready = False
    except Exception as e:
        checks["database"] = {
            "ready": False,
            "error": str(e),
        }
        overall_ready = False

    # Check if we can perform basic database operations
    try:
        # Simple query to verify database functionality
        vector_store_count = await db_service.get_trait_count()
        checks["database_operations"] = {
            "ready": True,
            "trait_count": vector_store_count,
        }
    except Exception as e:
        checks["database_operations"] = {
            "ready": False,
            "error": str(e),
        }
        overall_ready = False

    readiness_data = {
        "ready": overall_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }

    # Return 503 if not ready
    if not overall_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness_data,
        )

    return DataResponse(data=readiness_data)


@router.get("/live", response_model=DataResponse[dict[str, str]])
async def liveness_check() -> DataResponse[dict[str, str]]:
    """Kubernetes-style liveness probe.

    Simple check to verify the application process is running and responsive.
    """
    liveness_data = {
        "alive": "true",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": str(time.time() - _app_start_time),
    }

    return DataResponse(data=liveness_data)


# ==== Utility Functions ====


def increment_request_metric(metric_name: str) -> None:
    """Increment a request metric counter."""
    if metric_name in _request_metrics:
        _request_metrics[metric_name] += 1


def get_app_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - _app_start_time
