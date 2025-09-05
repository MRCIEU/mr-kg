"""Custom exception classes and error handling utilities."""

from typing import Any

from fastapi import HTTPException, status


class MRKGException(Exception):
    """Base exception class for MR-KG API errors."""

    def __init__(
        self,
        message: str,
        code: str = "MRKG_ERROR",
        context: dict[str, Any] | None = None,
    ):
        self.message = message
        self.code = code
        self.context = context or {}
        super().__init__(message)


class ValidationError(MRKGException):
    """Exception for data validation errors."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.field = field
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            context=context,
        )


class DatabaseError(MRKGException):
    """Exception for database-related errors."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.operation = operation
        super().__init__(
            message=message,
            code="DATABASE_ERROR",
            context=context,
        )


class NotFoundError(MRKGException):
    """Exception for resource not found errors."""

    def __init__(
        self,
        resource: str,
        identifier: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.resource = resource
        self.identifier = identifier
        message = f"{resource} not found"
        if identifier:
            message += f" with identifier: {identifier}"
        super().__init__(
            message=message,
            code="NOT_FOUND",
            context=context,
        )


class BusinessLogicError(MRKGException):
    """Exception for business logic validation errors."""

    def __init__(
        self,
        message: str,
        rule: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.rule = rule
        super().__init__(
            message=message,
            code="BUSINESS_LOGIC_ERROR",
            context=context,
        )


class ExternalServiceError(MRKGException):
    """Exception for external service communication errors."""

    def __init__(
        self,
        service: str,
        message: str,
        status_code: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.service = service
        self.status_code = status_code
        super().__init__(
            message=f"{service}: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            context=context,
        )


class RateLimitError(MRKGException):
    """Exception for rate limiting errors."""

    def __init__(
        self,
        limit: int,
        window: str,
        retry_after: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        message = f"Rate limit exceeded: {limit} requests per {window}"
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            context=context,
        )


class AuthenticationError(MRKGException):
    """Exception for authentication errors."""

    def __init__(
        self,
        message: str = "Authentication required",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            context=context,
        )


class AuthorizationError(MRKGException):
    """Exception for authorization errors."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.required_permission = required_permission
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            context=context,
        )


# ==== HTTP Exception Mapping ====


def mrkg_exception_to_http_exception(exc: MRKGException) -> HTTPException:
    """Convert MR-KG exception to FastAPI HTTPException."""
    # Map exception types to HTTP status codes
    status_map = {
        ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        NotFoundError: status.HTTP_404_NOT_FOUND,
        BusinessLogicError: status.HTTP_400_BAD_REQUEST,
        DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ExternalServiceError: status.HTTP_502_BAD_GATEWAY,
        RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        AuthorizationError: status.HTTP_403_FORBIDDEN,
    }

    http_status = status_map.get(
        type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    # Create error detail
    detail = {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "context": exc.context,
        }
    }

    # Add specific fields for certain exception types
    if isinstance(exc, ValidationError) and exc.field:
        detail["error"]["field"] = exc.field

    if isinstance(exc, NotFoundError):
        detail["error"]["resource"] = exc.resource
        if exc.identifier:
            detail["error"]["identifier"] = exc.identifier

    if isinstance(exc, BusinessLogicError) and exc.rule:
        detail["error"]["rule"] = exc.rule

    if isinstance(exc, ExternalServiceError):
        detail["error"]["service"] = exc.service
        if exc.status_code:
            detail["error"]["upstream_status"] = exc.status_code

    if isinstance(exc, RateLimitError):
        detail["error"]["limit"] = exc.limit
        detail["error"]["window"] = exc.window
        if exc.retry_after:
            detail["error"]["retry_after"] = exc.retry_after

    if isinstance(exc, AuthorizationError) and exc.required_permission:
        detail["error"]["required_permission"] = exc.required_permission

    # Add headers for rate limiting
    headers = None
    if isinstance(exc, RateLimitError) and exc.retry_after:
        headers = {"Retry-After": str(exc.retry_after)}

    return HTTPException(
        status_code=http_status,
        detail=detail,
        headers=headers,
    )


# ==== Validation Helpers ====


class MultipleValidationError(MRKGException):
    """Exception for multiple validation errors."""

    def __init__(
        self,
        errors: list[ValidationError],
        message: str = "Multiple validation errors",
    ):
        self.errors = errors
        super().__init__(
            message=message,
            code="MULTIPLE_VALIDATION_ERRORS",
            context={"error_count": len(errors)},
        )


def validate_and_raise(validations: list[tuple[bool, str, str | None]]) -> None:
    """Validate multiple conditions and raise MultipleValidationError if any fail.

    Args:
        validations: List of (condition, message, field) tuples
    """
    errors = []
    for condition, message, field in validations:
        if not condition:
            errors.append(ValidationError(message, field))

    if errors:
        raise MultipleValidationError(errors)


# ==== Error Context Helpers ====


def add_request_context(
    exc: MRKGException,
    request_id: str | None = None,
    endpoint: str | None = None,
    method: str | None = None,
    user_id: str | None = None,
) -> MRKGException:
    """Add request context to exception."""
    context = exc.context.copy()

    if request_id:
        context["request_id"] = request_id
    if endpoint:
        context["endpoint"] = endpoint
    if method:
        context["method"] = method
    if user_id:
        context["user_id"] = user_id

    exc.context = context
    return exc


def create_database_error(
    operation: str,
    error: Exception,
    context: dict[str, Any] | None = None,
) -> DatabaseError:
    """Create database error with proper context."""
    error_context = context or {}
    error_context.update(
        {
            "original_error": str(error),
            "error_type": type(error).__name__,
        }
    )

    return DatabaseError(
        message=f"Database operation failed: {operation}",
        operation=operation,
        context=error_context,
    )
