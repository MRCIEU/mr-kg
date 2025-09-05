"""Standardized response models for API endpoints."""

from datetime import UTC, datetime
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ==== Base Response Models ====


class BaseResponse(BaseModel):
    """Base response model with common fields."""

    success: bool = Field(default=True, description="Request success status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp",
    )
    request_id: str | None = Field(None, description="Request correlation ID")


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Human-readable error message")
    field: str | None = Field(None, description="Field that caused error")
    context: dict[str, Any] | None = Field(
        None, description="Additional error context"
    )


class ErrorResponse(BaseResponse):
    """Standardized error response model."""

    success: bool = Field(default=False, description="Request success status")
    error: ErrorDetail = Field(..., description="Error details")
    errors: list[ErrorDetail] | None = Field(
        None, description="Multiple validation errors"
    )


class DataResponse[T](BaseResponse):
    """Generic response wrapper for data payloads."""

    data: T = Field(..., description="Response data payload")


class ListResponse[T](BaseResponse):
    """Response model for list data with pagination."""

    data: list[T] = Field(default_factory=list, description="List of items")
    pagination: Optional["PaginationInfo"] = Field(
        None, description="Pagination information"
    )
    filters: dict[str, Any] | None = Field(None, description="Applied filters")


class PaginatedDataResponse[T](BaseResponse):
    """Paginated response model for API data with pagination metadata."""

    data: list[T] = Field(
        default_factory=list, description="Paginated data items"
    )
    pagination: "PaginationInfo" = Field(
        ..., description="Pagination information"
    )
    filters: dict[str, Any] | None = Field(None, description="Applied filters")


class PaginationInfo(BaseModel):
    """Pagination metadata for list responses."""

    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether next page exists")
    has_previous: bool = Field(..., description="Whether previous page exists")

    @classmethod
    def create(
        cls, page: int, page_size: int, total_items: int
    ) -> "PaginationInfo":
        """Create pagination info from parameters."""
        total_pages = (total_items + page_size - 1) // page_size
        return cls(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


# ==== System Response Models ====


class HealthStatus(BaseModel):
    """System health status information."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Application uptime in seconds")
    timestamp: datetime = Field(..., description="Health check timestamp")


class SystemInfo(BaseModel):
    """System information and capabilities."""

    application: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    api_version: str = Field(..., description="API version")
    database_profile: str = Field(..., description="Database profile")
    features: list[str] = Field(
        default_factory=list, description="Available features"
    )


class MetricsInfo(BaseModel):
    """Application metrics information."""

    requests_total: int = Field(..., description="Total requests processed")
    active_connections: int = Field(..., description="Active connections")
    database_queries: int = Field(..., description="Database queries executed")
    cache_hits: int = Field(..., description="Cache hit count")
    cache_misses: int = Field(..., description="Cache miss count")
    response_time_avg: float = Field(
        ..., description="Average response time in ms"
    )


# ==== Search and Filter Models ====


class SearchMeta(BaseModel):
    """Metadata for search operations."""

    query: str | None = Field(None, description="Search query")
    total_results: int = Field(..., description="Total matching results")
    search_time: float = Field(..., description="Search time in seconds")
    filters_applied: list[str] = Field(
        default_factory=list, description="Applied filter types"
    )


class SortInfo(BaseModel):
    """Sorting information for list responses."""

    field: str = Field(..., description="Sort field")
    direction: str = Field(..., description="Sort direction (asc/desc)")


# ==== Validation Models ====


class ValidationResult(BaseModel):
    """Result of data validation operations."""

    valid: bool = Field(..., description="Validation success status")
    errors: list[ErrorDetail] = Field(
        default_factory=list, description="Validation errors"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Validation warnings"
    )


# ==== Async Operation Models ====


class TaskInfo(BaseModel):
    """Information about background tasks."""

    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Task creation time")
    started_at: datetime | None = Field(None, description="Task start time")
    completed_at: datetime | None = Field(
        None, description="Task completion time"
    )
    progress: float | None = Field(
        None, ge=0, le=1, description="Task progress (0.0-1.0)"
    )
    result: Any | None = Field(None, description="Task result data")
    error: str | None = Field(None, description="Task error message")


# ==== API Documentation Models ====


class APIEndpoint(BaseModel):
    """Information about API endpoints."""

    path: str = Field(..., description="Endpoint path")
    method: str = Field(..., description="HTTP method")
    summary: str = Field(..., description="Endpoint summary")
    tags: list[str] = Field(default_factory=list, description="Endpoint tags")
    requires_auth: bool = Field(
        default=False, description="Authentication required"
    )


class APICapabilities(BaseModel):
    """API capabilities and feature information."""

    endpoints: list[APIEndpoint] = Field(
        default_factory=list, description="Available endpoints"
    )
    features: list[str] = Field(
        default_factory=list, description="Supported features"
    )
    rate_limits: dict[str, int] = Field(
        default_factory=dict, description="Rate limit information"
    )
    max_request_size: int = Field(..., description="Maximum request size")
    supported_formats: list[str] = Field(
        default_factory=list, description="Supported response formats"
    )


# Update forward references
ListResponse.model_rebuild()
