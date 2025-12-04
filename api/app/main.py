"""FastAPI application entry point for MR-KG API.

Provides RESTful endpoints for accessing Mendelian Randomization study data,
extraction results, and similarity metrics.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import (
    DatabaseError,
    get_evidence_profile_connection,
    get_trait_profile_connection,
    get_vector_store_connection,
)
from app.models import HealthResponse
from app.routers import similar, studies


# ==== Custom exceptions ====


class StudyNotFoundError(Exception):
    """Exception raised when a study is not found."""

    def __init__(self, pmid: str, model: str):
        self.pmid = pmid
        self.model = model
        self.message = f"Study {pmid} not found for model {model}"
        super().__init__(self.message)


class InvalidParameterError(Exception):
    """Exception raised for invalid request parameters."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# ==== Application setup ====


app = FastAPI(
    title="MR-KG API",
    description="API for Mendelian Randomization Knowledge Graph",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---- Configure CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Include routers ----
app.include_router(studies.router, prefix="/api")
app.include_router(similar.router, prefix="/api")


# ==== Exception handlers ====


@app.exception_handler(StudyNotFoundError)
async def study_not_found_handler(
    request: Request,
    exc: StudyNotFoundError,
) -> JSONResponse:
    """Handle study not found errors."""
    res = JSONResponse(
        status_code=404,
        content={"detail": exc.message},
    )
    return res


@app.exception_handler(InvalidParameterError)
async def invalid_parameter_handler(
    request: Request,
    exc: InvalidParameterError,
) -> JSONResponse:
    """Handle invalid parameter errors."""
    res = JSONResponse(
        status_code=400,
        content={"detail": exc.message},
    )
    return res


@app.exception_handler(DatabaseError)
async def database_error_handler(
    request: Request,
    exc: DatabaseError,
) -> JSONResponse:
    """Handle database errors."""
    res = JSONResponse(
        status_code=500,
        content={"detail": f"Database error: {str(exc)}"},
    )
    return res


# ==== Health check endpoint ====


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Verifies database connectivity for all three databases and returns
    their status.
    """
    settings = get_settings()

    databases = {
        "vector_store": False,
        "trait_profile": False,
        "evidence_profile": False,
    }

    # ---- Check vector store database ----
    try:
        vector_store_path = Path(settings.vector_store_path)
        if vector_store_path.exists():
            conn = get_vector_store_connection()
            conn.execute("SELECT 1").fetchone()
            databases["vector_store"] = True
    except Exception:
        pass

    # ---- Check trait profile database ----
    try:
        trait_profile_path = Path(settings.trait_profile_path)
        if trait_profile_path.exists():
            conn = get_trait_profile_connection()
            conn.execute("SELECT 1").fetchone()
            databases["trait_profile"] = True
    except Exception:
        pass

    # ---- Check evidence profile database ----
    try:
        evidence_profile_path = Path(settings.evidence_profile_path)
        if evidence_profile_path.exists():
            conn = get_evidence_profile_connection()
            conn.execute("SELECT 1").fetchone()
            databases["evidence_profile"] = True
    except Exception:
        pass

    # ---- Determine overall status ----
    all_healthy = all(databases.values())
    status = "healthy" if all_healthy else "degraded"

    res = HealthResponse(
        status=status,
        databases=databases,
    )
    return res
