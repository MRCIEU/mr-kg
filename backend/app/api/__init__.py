"""Main API router configuration."""

from fastapi import APIRouter

from app.api.v1 import core, health, similarities, studies, system, traits


def create_api_router() -> APIRouter:
    """Create the main API router with all sub-routers."""
    api_router = APIRouter()

    # ---- API v1 routes ----
    api_router.include_router(
        health.router,
        prefix="/health",
        tags=["health"],
    )

    api_router.include_router(
        system.router,
        prefix="/system",
        tags=["system"],
    )

    api_router.include_router(
        core.router,
        prefix="/core",
        tags=["core"],
    )

    api_router.include_router(
        traits.router,
        prefix="/traits",
        tags=["traits"],
    )

    api_router.include_router(
        studies.router,
        prefix="/studies",
        tags=["studies"],
    )

    api_router.include_router(
        similarities.router,
        prefix="/similarities",
        tags=["similarities"],
    )

    return api_router
