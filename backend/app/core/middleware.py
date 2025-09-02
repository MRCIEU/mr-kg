"""Middleware components for the FastAPI application."""

import time
import uuid
from typing import Any, Callable, Optional

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.types import ASGIApp

import logging

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""

    def __init__(
        self,
        app: ASGIApp,
        include_request_body: bool = False,
        include_response_body: bool = False,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.include_request_body = include_request_body
        self.include_response_body = include_response_body
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and response with logging."""
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Log request
        await self._log_request(request, request_id)

        # Process request
        try:
            response = await call_next(request)
        except Exception as exc:
            # Log exception
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": self._get_client_ip(request),
                    "user_agent": request.headers.get("user-agent"),
                    "process_time": process_time,
                    "exception": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
            raise

        # Calculate process time
        process_time = time.time() - start_time

        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)

        # Log response
        await self._log_response(request, response, request_id, process_time)

        return response

    async def _log_request(self, request: Request, request_id: str) -> None:
        """Log incoming request details."""
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }

        # Include request body if configured
        if self.include_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_data["body_size"] = len(body)
                    # Don't log the actual body for security
                    # request_data["body"] = body.decode("utf-8")[:1000]
            except Exception as e:
                request_data["body_read_error"] = str(e)

        logger.info("Request received", extra=request_data)

    async def _log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        process_time: float,
    ) -> None:
        """Log response details."""
        response_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
            "process_time": process_time,
        }

        # Log level based on status code
        if response.status_code >= 500:
            logger.error("Request completed", extra=response_data)
        elif response.status_code >= 400:
            logger.warning("Request completed", extra=response_data)
        else:
            logger.info("Request completed", extra=response_data)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if hasattr(request.client, "host"):
            return request.client.host

        return "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""

    def __init__(
        self,
        app: ASGIApp,
        include_csp: bool = True,
        include_hsts: bool = False,  # Disabled by default for development
        custom_headers: Optional[dict[str, str]] = None,
    ):
        super().__init__(app)
        self.include_csp = include_csp
        self.include_hsts = include_hsts
        self.custom_headers = custom_headers or {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Basic security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy
        if self.include_csp:
            csp_directives = [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: https:",
                "font-src 'self'",
                "connect-src 'self'",
                "frame-ancestors 'none'",
            ]
            response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # HTTP Strict Transport Security (for production)
        if self.include_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

        # Custom headers
        for header, value in self.custom_headers.items():
            response.headers[header] = value

        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        enabled: bool = True,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.enabled = enabled
        
        # Simple in-memory storage (use Redis in production)
        self.request_counts: dict[str, dict[str, Any]] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply rate limiting to requests."""
        if not self.enabled:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Check rate limits
        if self._is_rate_limited(client_ip, current_time):
            from app.core.exceptions import RateLimitError, mrkg_exception_to_http_exception
            
            error = RateLimitError(
                limit=self.requests_per_minute,
                window="minute",
                retry_after=60,
            )
            raise mrkg_exception_to_http_exception(error)

        # Record request
        self._record_request(client_ip, current_time)

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        if hasattr(request.client, "host"):
            return request.client.host

        return "unknown"

    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limits."""
        if client_ip not in self.request_counts:
            return False

        client_data = self.request_counts[client_ip]
        
        # Clean old entries
        self._clean_old_entries(client_data, current_time)

        # Check minute limit
        minute_key = int(current_time // 60)
        minute_requests = client_data.get("minutes", {}).get(minute_key, 0)
        if minute_requests >= self.requests_per_minute:
            return True

        # Check hour limit
        hour_key = int(current_time // 3600)
        hour_requests = client_data.get("hours", {}).get(hour_key, 0)
        if hour_requests >= self.requests_per_hour:
            return True

        return False

    def _record_request(self, client_ip: str, current_time: float) -> None:
        """Record a request for rate limiting."""
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {"minutes": {}, "hours": {}}

        client_data = self.request_counts[client_ip]

        # Record minute
        minute_key = int(current_time // 60)
        client_data["minutes"][minute_key] = (
            client_data["minutes"].get(minute_key, 0) + 1
        )

        # Record hour
        hour_key = int(current_time // 3600)
        client_data["hours"][hour_key] = (
            client_data["hours"].get(hour_key, 0) + 1
        )

    def _clean_old_entries(self, client_data: dict, current_time: float) -> None:
        """Remove old entries to prevent memory leaks."""
        current_minute = int(current_time // 60)
        current_hour = int(current_time // 3600)

        # Keep only last 2 minutes and 2 hours
        minute_keys = list(client_data["minutes"].keys())
        for key in minute_keys:
            if key < current_minute - 1:
                del client_data["minutes"][key]

        hour_keys = list(client_data["hours"].keys())
        for key in hour_keys:
            if key < current_hour - 1:
                del client_data["hours"][key]


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting application metrics."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "requests_total": 0,
            "requests_by_method": {},
            "requests_by_status": {},
            "response_times": [],
            "active_requests": 0,
        }

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Collect metrics for request processing."""
        start_time = time.time()
        
        # Increment active requests
        self.metrics["active_requests"] += 1
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(request, response, response_time)
            
            return response
            
        finally:
            # Decrement active requests
            self.metrics["active_requests"] -= 1

    def _update_metrics(
        self, request: Request, response: Response, response_time: float
    ) -> None:
        """Update collected metrics."""
        # Total requests
        self.metrics["requests_total"] += 1

        # Requests by method
        method = request.method
        self.metrics["requests_by_method"][method] = (
            self.metrics["requests_by_method"].get(method, 0) + 1
        )

        # Requests by status code
        status_code = response.status_code
        self.metrics["requests_by_status"][status_code] = (
            self.metrics["requests_by_status"].get(status_code, 0) + 1
        )

        # Response times (keep only last 1000)
        self.metrics["response_times"].append(response_time)
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        response_times = self.metrics["response_times"]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        return {
            "requests_total": self.metrics["requests_total"],
            "requests_by_method": self.metrics["requests_by_method"].copy(),
            "requests_by_status": self.metrics["requests_by_status"].copy(),
            "active_requests": self.metrics["active_requests"],
            "average_response_time": avg_response_time,
            "total_response_time_samples": len(response_times),
        }


# ==== Middleware Factory Functions ====


def create_request_logging_middleware(
    include_request_body: bool = False,
    include_response_body: bool = False,
    exclude_paths: Optional[list[str]] = None,
) -> Callable[[ASGIApp], RequestLoggingMiddleware]:
    """Create request logging middleware factory."""
    def middleware_factory(app: ASGIApp) -> RequestLoggingMiddleware:
        return RequestLoggingMiddleware(
            app=app,
            include_request_body=include_request_body,
            include_response_body=include_response_body,
            exclude_paths=exclude_paths,
        )
    return middleware_factory


def create_security_headers_middleware(
    include_csp: bool = True,
    include_hsts: bool = False,
    custom_headers: Optional[dict[str, str]] = None,
) -> Callable[[ASGIApp], SecurityHeadersMiddleware]:
    """Create security headers middleware factory."""
    def middleware_factory(app: ASGIApp) -> SecurityHeadersMiddleware:
        return SecurityHeadersMiddleware(
            app=app,
            include_csp=include_csp,
            include_hsts=include_hsts,
            custom_headers=custom_headers,
        )
    return middleware_factory


def create_rate_limiting_middleware(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    enabled: bool = True,
) -> Callable[[ASGIApp], RateLimitingMiddleware]:
    """Create rate limiting middleware factory."""
    def middleware_factory(app: ASGIApp) -> RateLimitingMiddleware:
        return RateLimitingMiddleware(
            app=app,
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            enabled=enabled,
        )
    return middleware_factory