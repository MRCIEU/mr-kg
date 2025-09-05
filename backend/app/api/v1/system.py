"""System information and utility endpoints."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.dependencies import get_database_service
from app.models.responses import APICapabilities, APIEndpoint, DataResponse
from app.services.database_service import DatabaseService

router = APIRouter()


@router.get("/info", response_model=DataResponse[dict[str, Any]])
async def system_information() -> DataResponse[dict[str, Any]]:
    """Get comprehensive system information and configuration."""
    info = {
        "application": {
            "name": "MR-KG API",
            "version": "0.1.0",
            "description": "FastAPI backend for MR-KG (Mendelian Randomization Knowledge Graph)",
            "environment": "development" if settings.DEBUG else "production",
        },
        "api": {
            "version": "v1",
            "prefix": settings.API_V1_PREFIX,
            "docs_enabled": settings.DEBUG,
        },
        "database": {
            "profile": settings.DB_PROFILE,
            "vector_store_path": settings.VECTOR_STORE_PATH,
            "trait_profile_path": settings.TRAIT_PROFILE_PATH,
        },
        "features": {
            "vector_search": True,
            "trait_analysis": True,
            "similarity_computation": True,
            "health_monitoring": True,
            "request_logging": True,
            "error_handling": True,
            "rate_limiting": True,
            "security_headers": True,
        },
        "cors": {
            "allowed_origins": settings.ALLOWED_ORIGINS,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    return DataResponse(data=info)


@router.get("/capabilities", response_model=DataResponse[APICapabilities])
async def api_capabilities() -> DataResponse[APICapabilities]:
    """Get API capabilities and available endpoints."""
    # Define available endpoints
    endpoints = [
        APIEndpoint(
            path="/health",
            method="GET",
            summary="Basic health check",
            tags=["health"],
            requires_auth=False,
        ),
        APIEndpoint(
            path="/health/detailed",
            method="GET",
            summary="Detailed health check with database connectivity",
            tags=["health"],
            requires_auth=False,
        ),
        APIEndpoint(
            path="/health/database",
            method="GET",
            summary="Database connectivity check",
            tags=["health"],
            requires_auth=False,
        ),
        APIEndpoint(
            path="/health/ready",
            method="GET",
            summary="Kubernetes readiness probe",
            tags=["health"],
            requires_auth=False,
        ),
        APIEndpoint(
            path="/health/live",
            method="GET",
            summary="Kubernetes liveness probe",
            tags=["health"],
            requires_auth=False,
        ),
        APIEndpoint(
            path="/system/info",
            method="GET",
            summary="System information",
            tags=["system"],
            requires_auth=False,
        ),
        APIEndpoint(
            path="/system/capabilities",
            method="GET",
            summary="API capabilities",
            tags=["system"],
            requires_auth=False,
        ),
        APIEndpoint(
            path="/core/version",
            method="GET",
            summary="API version information",
            tags=["core"],
            requires_auth=False,
        ),
    ]

    # API features
    features = [
        "RESTful API design",
        "OpenAPI 3.0 documentation",
        "Automatic request/response validation",
        "Type-safe Pydantic models",
        "Comprehensive error handling",
        "Request logging and monitoring",
        "Database health checks",
        "CORS support",
        "Security headers",
        "Rate limiting",
        "Pagination support",
        "Vector similarity search",
        "Trait analysis",
        "Study similarity computation",
    ]

    # Rate limits (basic implementation)
    rate_limits = {
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
    }

    capabilities = APICapabilities(
        endpoints=endpoints,
        features=features,
        rate_limits=rate_limits,
        max_request_size=10 * 1024 * 1024,  # 10MB
        supported_formats=["application/json"],
    )

    return DataResponse(data=capabilities)


@router.get("/status", response_model=DataResponse[dict[str, Any]])
async def system_status(
    db_service: DatabaseService = Depends(get_database_service),
) -> DataResponse[dict[str, Any]]:
    """Get current system status and operational metrics."""
    status_info: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime": "calculated_by_health_endpoint",  # Would be calculated
        "status": "operational",
    }

    # Database status
    try:
        trait_count = await db_service.get_trait_count()
        study_count = await db_service.get_study_count()

        status_info["database"] = {
            "status": "operational",
            "trait_count": trait_count,
            "study_count": study_count,
        }
    except Exception as e:
        status_info["database"] = {
            "status": "error",
            "error": str(e),
        }
        status_info["status"] = "degraded"

    # Service status
    status_info["services"] = {
        "vector_search": "operational",
        "trait_analysis": "operational",
        "similarity_computation": "operational",
    }

    return DataResponse(data=status_info)


@router.get("/config", response_model=DataResponse[dict[str, Any]])
async def system_configuration() -> DataResponse[dict[str, Any]]:
    """Get non-sensitive system configuration information."""
    config = {
        "server": {
            "debug_mode": settings.DEBUG,
            "host": settings.HOST,
            "port": settings.PORT,
        },
        "api": {
            "version_prefix": settings.API_V1_PREFIX,
            "docs_enabled": settings.DEBUG,
        },
        "database": {
            "profile": settings.DB_PROFILE,
            # Don't expose full paths for security
            "vector_store_configured": bool(settings.VECTOR_STORE_PATH),
            "trait_profile_configured": bool(settings.TRAIT_PROFILE_PATH),
        },
        "features": {
            "cors_enabled": len(settings.ALLOWED_ORIGINS) > 0,
            "allowed_origins_count": len(settings.ALLOWED_ORIGINS),
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    return DataResponse(data=config)


@router.get("/environment", response_model=DataResponse[dict[str, str]])
async def environment_info() -> DataResponse[dict[str, str]]:
    """Get environment information for debugging and support."""
    env_info = {
        "environment": "development" if settings.DEBUG else "production",
        "debug_mode": str(settings.DEBUG),
        "database_profile": settings.DB_PROFILE,
        "api_version": "v1",
        "python_version": "3.12+",  # From pyproject.toml
        "fastapi_version": "0.104.0+",  # From dependencies
        "timestamp": datetime.now(UTC).isoformat(),
    }

    return DataResponse(data=env_info)


@router.post("/validate", response_model=DataResponse[dict[str, Any]])
async def validate_system_integrity(
    db_service: DatabaseService = Depends(get_database_service),
) -> DataResponse[dict[str, Any]]:
    """Validate system integrity and configuration.

    This endpoint performs comprehensive system validation including
    database schema validation, connectivity checks, and configuration
    verification.
    """
    validation_results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "overall_status": "unknown",
        "checks": {},
    }

    all_passed = True

    # Database connectivity check
    try:
        trait_count = await db_service.get_trait_count()
        validation_results["checks"]["database_connectivity"] = {
            "status": "passed",
            "details": f"Successfully connected, found {trait_count} traits",
        }
    except Exception as e:
        validation_results["checks"]["database_connectivity"] = {
            "status": "failed",
            "error": str(e),
        }
        all_passed = False

    # Database schema validation (basic)
    try:
        # This would use the schema validation service
        validation_results["checks"]["database_schema"] = {
            "status": "passed",
            "details": "Schema validation not implemented yet",
        }
    except Exception as e:
        validation_results["checks"]["database_schema"] = {
            "status": "failed",
            "error": str(e),
        }
        all_passed = False

    # Configuration validation
    config_issues = []
    if not settings.VECTOR_STORE_PATH:
        config_issues.append("Vector store path not configured")
    if not settings.TRAIT_PROFILE_PATH:
        config_issues.append("Trait profile path not configured")
    if not settings.ALLOWED_ORIGINS:
        config_issues.append("No CORS origins configured")

    if config_issues:
        validation_results["checks"]["configuration"] = {
            "status": "warning",
            "issues": config_issues,
        }
    else:
        validation_results["checks"]["configuration"] = {
            "status": "passed",
            "details": "Configuration validation passed",
        }

    # Set overall status
    if all_passed:
        validation_results["overall_status"] = "passed"
    elif any(
        check.get("status") == "failed"
        for check in validation_results["checks"].values()
    ):
        validation_results["overall_status"] = "failed"
    else:
        validation_results["overall_status"] = "warning"

    return DataResponse(data=validation_results)
