"""Core API endpoints for version and basic functionality."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request

from app.models.responses import DataResponse

router = APIRouter()


@router.get("/version", response_model=DataResponse[dict[str, str]])
async def api_version() -> DataResponse[dict[str, str]]:
    """Get API version information."""
    version_info = {
        "api_version": "v1",
        "application_version": "0.1.0",
        "build_date": "2025-09-01",  # Would be set during build
        "commit_hash": "unknown",  # Would be set during build
        "timestamp": datetime.utcnow().isoformat(),
    }

    return DataResponse(data=version_info)


@router.get("/ping", response_model=DataResponse[dict[str, str]])
async def ping() -> DataResponse[dict[str, str]]:
    """Simple ping endpoint for connectivity testing."""
    ping_data = {
        "message": "pong",
        "timestamp": datetime.utcnow().isoformat(),
    }

    return DataResponse(data=ping_data)


@router.get("/echo", response_model=DataResponse[dict[str, Any]])
async def echo(request: Request) -> DataResponse[dict[str, Any]]:
    """Echo request information for debugging purposes."""
    echo_data = {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "headers": {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in ["authorization", "cookie"]  # Security
        },
        "client": {
            "host": getattr(request.client, "host", "unknown"),
            "port": getattr(request.client, "port", "unknown"),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    return DataResponse(data=echo_data)


@router.options("/", include_in_schema=False)
async def options_handler() -> dict[str, str]:
    """Handle OPTIONS requests for CORS preflight."""
    return {"message": "OK"}


@router.get("/", response_model=DataResponse[dict[str, str]])
async def api_root() -> DataResponse[dict[str, str]]:
    """API root endpoint with basic information."""
    root_data = {
        "message": "MR-KG API v1",
        "description": "FastAPI backend for MR-KG (Mendelian Randomization Knowledge Graph)",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "timestamp": datetime.utcnow().isoformat(),
    }

    return DataResponse(data=root_data)
