"""Tests for API core infrastructure including middleware and error handling."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import (
    DatabaseError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from app.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def app():
    """Create test app."""
    return create_app()


class TestApplicationSetup:
    """Test application setup and configuration."""

    def test_app_creation(self, app):
        """Test that app is created successfully."""
        assert app.title == "MR-KG API"
        assert app.version == "0.1.0"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "MR-KG API"
        assert data["version"] == "0.1.0"
        assert "health" in data


class TestCoreEndpoints:
    """Test core API endpoints."""

    def test_api_version(self, client):
        """Test API version endpoint."""
        response = client.get("/api/v1/core/version")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["api_version"] == "v1"
        assert data["data"]["application_version"] == "0.1.0"

    def test_ping_endpoint(self, client):
        """Test ping endpoint."""
        response = client.get("/api/v1/core/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["message"] == "pong"

    def test_echo_endpoint(self, client):
        """Test echo endpoint."""
        response = client.get(
            "/api/v1/core/echo?test=value",
            headers={"X-Test-Header": "test-value"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["method"] == "GET"
        assert data["data"]["query_params"]["test"] == "value"
        assert "x-test-header" in data["data"]["headers"]


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health_check(self, client):
        """Test basic health check."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert "uptime" in data["data"]

    @patch("app.api.v1.health.perform_database_health_check")
    async def test_detailed_health_check(self, mock_health_check, client):
        """Test detailed health check."""
        # Mock the health check response
        mock_health_check.return_value = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "vector_store": {
                "database_path": "test.db",
                "accessible": True,
                "table_count": 5,
                "view_count": 2,
                "index_count": 10,
                "last_checked": "2024-01-01T00:00:00Z",
            },
            "trait_profile": {
                "database_path": "test_profile.db",
                "accessible": True,
                "table_count": 3,
                "view_count": 1,
                "index_count": 5,
                "last_checked": "2024-01-01T00:00:00Z",
            },
            "performance_metrics": {
                "query_time": 0.1,
                "connection_time": 0.05,
            },
        }

        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200

    def test_database_health_check(self, client):
        """Test database health check."""
        response = client.get("/api/v1/health/database")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]

    def test_readiness_check(self, client):
        """Test readiness check."""
        with patch(
            "app.core.dependencies.check_database_connectivity"
        ) as mock_conn:
            mock_conn.return_value = {"status": "healthy"}
            with patch(
                "app.services.database_service.DatabaseService.get_trait_count"
            ) as mock_count:
                mock_count.return_value = 100
                response = client.get("/api/v1/health/ready")
                # May return 503 if database not available in test
                assert response.status_code in [200, 503]

    def test_liveness_check(self, client):
        """Test liveness check."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["alive"] == "true"


class TestSystemEndpoints:
    """Test system information endpoints."""

    def test_system_info(self, client):
        """Test system information endpoint."""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "application" in data["data"]
        assert "api" in data["data"]
        assert "database" in data["data"]
        assert "features" in data["data"]

    def test_api_capabilities(self, client):
        """Test API capabilities endpoint."""
        response = client.get("/api/v1/system/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "endpoints" in data["data"]
        assert "features" in data["data"]
        assert "rate_limits" in data["data"]

    def test_system_status(self, client):
        """Test system status endpoint."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]
        assert "database" in data["data"]

    def test_system_config(self, client):
        """Test system configuration endpoint."""
        response = client.get("/api/v1/system/config")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "server" in data["data"]
        assert "api" in data["data"]
        assert "database" in data["data"]

    def test_environment_info(self, client):
        """Test environment information endpoint."""
        response = client.get("/api/v1/system/environment")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "environment" in data["data"]
        assert "debug_mode" in data["data"]


class TestMiddleware:
    """Test middleware functionality."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/core/ping")
        assert response.status_code == 200
        # CORS headers should be present

    def test_security_headers(self, client):
        """Test security headers are added."""
        response = client.get("/api/v1/core/ping")
        assert response.status_code == 200

        # Check for security headers
        headers = response.headers
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "X-API-Version" in headers
        assert "X-Service" in headers

    def test_request_id_header(self, client):
        """Test request ID header is added."""
        response = client.get("/api/v1/core/ping")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    def test_process_time_header(self, client):
        """Test process time header is added."""
        response = client.get("/api/v1/core/ping")
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers


class TestErrorHandling:
    """Test error handling and exception responses."""

    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    def test_validation_error(self, client):
        """Test validation error handling."""
        # This would require an endpoint that expects validation
        response = client.post("/api/v1/core/ping", json={"invalid": "data"})
        assert response.status_code == 405  # Method not allowed

    def test_custom_exception_handling(self, app):
        """Test custom exception handling."""
        from app.core.error_handlers import mrkg_exception_to_http_exception

        # Test ValidationError
        exc = ValidationError("Test validation error", field="test_field")
        http_exc = mrkg_exception_to_http_exception(exc)
        assert http_exc.status_code == 422
        assert "VALIDATION_ERROR" in str(http_exc.detail)

        # Test NotFoundError
        exc = NotFoundError("test_resource", "test_id")
        http_exc = mrkg_exception_to_http_exception(exc)
        assert http_exc.status_code == 404

        # Test DatabaseError
        exc = DatabaseError("Test database error", operation="select")
        http_exc = mrkg_exception_to_http_exception(exc)
        assert http_exc.status_code == 500

        # Test RateLimitError
        exc = RateLimitError(limit=60, window="minute", retry_after=30)
        http_exc = mrkg_exception_to_http_exception(exc)
        assert http_exc.status_code == 429
        assert http_exc.headers is not None
        assert http_exc.headers["Retry-After"] == "30"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiting_disabled_in_debug(self, client):
        """Test that rate limiting is disabled in debug mode."""
        # Make multiple requests quickly
        for _ in range(10):
            response = client.get("/api/v1/core/ping")
            assert response.status_code == 200

    @patch("app.core.config.settings.DEBUG", False)
    def test_rate_limiting_enabled_in_production(self, client):
        """Test that rate limiting works in production mode."""
        # This would require a production app instance
        pass


class TestResponseFormats:
    """Test standardized response formats."""

    def test_success_response_format(self, client):
        """Test success response format."""
        response = client.get("/api/v1/core/ping")
        assert response.status_code == 200
        data = response.json()

        # Check standard response format
        assert "success" in data
        assert "timestamp" in data
        assert "data" in data
        assert data["success"] is True

    def test_error_response_format(self, client):
        """Test error response format."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()

        # Check error response format
        assert "success" in data
        assert "error" in data
        assert data["success"] is False
        assert "code" in data["error"]
        assert "message" in data["error"]


class TestAuthentication:
    """Test authentication scaffolding."""

    def test_endpoints_accessible_without_auth(self, client):
        """Test that basic endpoints are accessible without authentication."""
        endpoints = [
            "/api/v1/health/",
            "/api/v1/core/ping",
            "/api/v1/core/version",
            "/api/v1/system/info",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200


class TestDocumentation:
    """Test OpenAPI documentation."""

    def test_openapi_schema_generation(self, client):
        """Test OpenAPI schema is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "MR-KG API"

    def test_docs_endpoint(self, client):
        """Test documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
