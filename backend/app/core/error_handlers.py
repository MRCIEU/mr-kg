"""Global exception handlers for the FastAPI application."""

import logging
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import (
    MRKGException,
    ValidationError,
    mrkg_exception_to_http_exception,
)
from app.models.responses import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


async def mrkg_exception_handler(
    request: Request, exc: MRKGException
) -> JSONResponse:
    """Handle custom MR-KG exceptions."""
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)

    # Log the exception
    logger.error(
        f"MR-KG exception: {exc.code}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "exception_type": type(exc).__name__,
            "exception_message": exc.message,
            "exception_context": exc.context,
        },
    )

    # Convert to HTTP exception
    http_exc = mrkg_exception_to_http_exception(exc)

    # Create error response
    error_detail = ErrorDetail(
        code=exc.code,
        message=exc.message,
        context=exc.context,
    )

    # Add specific fields for certain exception types
    if isinstance(exc, ValidationError) and exc.field:
        error_detail.field = exc.field

    error_response = ErrorResponse(
        request_id=request_id,
        error=error_detail,
    )

    return JSONResponse(
        status_code=http_exc.status_code,
        content=error_response.model_dump(),
        headers=http_exc.headers,
    )


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)

    # Log the exception
    logger.warning(
        f"HTTP exception: {exc.status_code}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )

    # Handle structured error details (from our custom exceptions)
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
                "request_id": request_id,
                **exc.detail,
            },
            headers=exc.headers,
        )

    # Handle simple error messages
    error_detail = ErrorDetail(
        code="HTTP_ERROR",
        message=str(exc.detail) if exc.detail else "An error occurred",
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=error_detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)

    # Log validation errors
    logger.warning(
        "Request validation error",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "errors": exc.errors(),
        },
    )

    # Convert Pydantic errors to our error format
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        error_detail = ErrorDetail(
            code="VALIDATION_ERROR",
            message=error["msg"],
            field=field_path,
            context={"type": error["type"]},
        )
        errors.append(error_detail)

    # Use first error as primary error
    primary_error = (
        errors[0]
        if errors
        else ErrorDetail(
            code="VALIDATION_ERROR",
            message="Validation failed",
        )
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=primary_error,
        errors=errors if len(errors) > 1 else None,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(),
    )


async def starlette_http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)

    # Log the exception
    logger.warning(
        f"Starlette HTTP exception: {exc.status_code}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )

    error_detail = ErrorDetail(
        code="HTTP_ERROR",
        message=str(exc.detail) if exc.detail else "An error occurred",
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=error_detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    # Get request ID if available
    request_id = getattr(request.state, "request_id", None)

    # Log the exception with full traceback
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
    )

    # Don't expose internal error details in production
    error_detail = ErrorDetail(
        code="INTERNAL_SERVER_ERROR",
        message="An internal server error occurred",
        context={
            "type": type(exc).__name__,
            # Only include exception message in debug mode
            "debug_message": str(exc),
        },
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=error_detail,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


# ==== Exception Handler Registry ====


def register_exception_handlers(app) -> None:
    """Register all exception handlers with the FastAPI app."""

    # Custom MR-KG exceptions
    app.add_exception_handler(MRKGException, mrkg_exception_handler)

    # FastAPI exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(
        RequestValidationError, validation_exception_handler
    )

    # Starlette exceptions
    app.add_exception_handler(
        StarletteHTTPException, starlette_http_exception_handler
    )

    # Catch-all for unexpected exceptions
    app.add_exception_handler(Exception, general_exception_handler)


# ==== Utility Functions ====


def create_error_response(
    message: str,
    code: str = "ERROR",
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    context: dict[str, Any] = None,
    request_id: str = None,
) -> JSONResponse:
    """Create a standardized error response."""
    error_detail = ErrorDetail(
        code=code,
        message=message,
        context=context,
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=error_detail,
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(),
    )


def create_validation_error_response(
    field: str,
    message: str,
    request_id: str = None,
) -> JSONResponse:
    """Create a validation error response."""
    error_detail = ErrorDetail(
        code="VALIDATION_ERROR",
        message=message,
        field=field,
    )

    error_response = ErrorResponse(
        request_id=request_id,
        error=error_detail,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(),
    )
