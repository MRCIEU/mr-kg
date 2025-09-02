"""Main FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import create_api_router
from app.core.config import settings
from app.core.database import close_database_pool, get_database_pool
from app.core.dependencies import check_database_connectivity
from app.core.error_handlers import register_exception_handlers
from app.core.middleware import (
    MetricsMiddleware,
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting MR-KG API...")

    try:
        # Initialize database connection pool
        await get_database_pool()
        logger.info("Database connection pool initialized")

        # Perform initial connectivity check
        connectivity = await check_database_connectivity()
        if connectivity["status"] != "healthy":
            logger.warning(f"Database connectivity issues: {connectivity}")
        else:
            logger.info("Database connectivity verified")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Don't raise here to allow app to start and show health check status

    yield

    # Shutdown
    logger.info("Shutting down MR-KG API...")
    try:
        await close_database_pool()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations.

    Returns:
        Configured FastAPI application instance
    """
    # ---- Create FastAPI app ----
    app = FastAPI(
        title="MR-KG API",
        description="FastAPI backend for MR-KG (Mendelian Randomization Knowledge Graph)",
        version="0.1.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ---- Register exception handlers ----
    register_exception_handlers(app)

    # ---- Add middleware stack ----
    # Note: Middleware is processed in reverse order of addition

    # Metrics collection (innermost)
    app.add_middleware(MetricsMiddleware)

    # Rate limiting
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=60,
        requests_per_hour=1000,
        enabled=not settings.DEBUG,  # Disable in debug mode
    )

    # Security headers
    app.add_middleware(
        SecurityHeadersMiddleware,
        include_csp=True,
        include_hsts=False,  # Disabled for development
        custom_headers={
            "X-API-Version": "v1",
            "X-Service": "mr-kg-api",
        },
    )

    # Request logging
    app.add_middleware(
        RequestLoggingMiddleware,
        include_request_body=settings.DEBUG,
        include_response_body=False,
        exclude_paths=["/health", "/docs", "/openapi.json", "/favicon.ico"],
    )

    # CORS (outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # ---- Register API routes ----
    api_router = create_api_router()
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # ---- Root endpoint ----
    @app.get("/", include_in_schema=False)
    async def root():
        """API root endpoint."""
        return {
            "message": "MR-KG API",
            "version": "0.1.0",
            "docs": "/docs" if settings.DEBUG else "disabled",
            "health": "/api/v1/health",
        }

    return app


# Create the app instance
app = create_app()
