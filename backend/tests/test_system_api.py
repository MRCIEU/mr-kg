"""Tests for the system API endpoints."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestSystemAPI:
    """Test cases for system API endpoints."""

    def test_system_info(self):
        """Test system info endpoint."""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        info = data["data"]
        assert "application" in info
        assert "api" in info
        assert "database" in info
        assert "features" in info

    def test_system_status(self):
        """Test system status endpoint."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        status = data["data"]
        assert "status" in status
        assert "uptime" in status
        assert "database" in status
        assert "services" in status

    def test_system_capabilities(self):
        """Test system capabilities endpoint."""
        response = client.get("/api/v1/system/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        capabilities = data["data"]
        assert "endpoints" in capabilities
        assert "features" in capabilities
        assert "rate_limits" in capabilities
        assert isinstance(capabilities["endpoints"], list)
        assert len(capabilities["endpoints"]) > 0

    def test_system_config(self):
        """Test system config endpoint."""
        response = client.get("/api/v1/system/config")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        config = data["data"]
        assert "api_version" in config
        assert "debug_mode" in config
        assert "cors_origins" in config

    def test_system_environment(self):
        """Test system environment endpoint."""
        response = client.get("/api/v1/system/environment")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        env = data["data"]
        assert "environment" in env
        assert "database_url_configured" in env
        assert "api_prefix" in env

    @patch("app.core.dependencies.get_database_service")
    def test_system_validate_success(self, mock_get_service):
        """Test system validation endpoint with successful validation."""
        mock_service = Mock()
        mock_service.validate_system.return_value = {
            "database": True,
            "models": True,
            "dependencies": True,
        }
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/system/validate")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        validation = data["data"]
        assert "validation_results" in validation
        assert "overall_status" in validation

    @patch("app.core.dependencies.get_database_service")
    def test_system_validate_failure(self, mock_get_service):
        """Test system validation endpoint with validation failures."""
        mock_service = Mock()
        mock_service.validate_system.return_value = {
            "database": False,
            "models": True,
            "dependencies": False,
        }
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/system/validate")
        assert response.status_code == 200
        data = response.json()
        validation = data["data"]
        assert validation["overall_status"] is False

    def test_system_validate_without_database(self):
        """Test system validation when database is unavailable."""
        response = client.get("/api/v1/system/validate")
        # Should still return a response even if database validation fails
        assert response.status_code in [200, 500]


class TestSystemAPIPermissions:
    """Test system API permission handling."""

    def test_system_endpoints_require_no_auth(self):
        """Test that system endpoints don't require authentication."""
        endpoints = [
            "/api/v1/system/info",
            "/api/v1/system/status",
            "/api/v1/system/capabilities",
            "/api/v1/system/config",
            "/api/v1/system/environment",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should not return 401/403 (authentication/authorization errors)
            assert response.status_code not in [401, 403]

    def test_system_validate_endpoint_access(self):
        """Test system validate endpoint access."""
        response = client.get("/api/v1/system/validate")
        # Should not return 401/403
        assert response.status_code not in [401, 403]


class TestSystemAPIPerformance:
    """Test system API performance."""

    def test_system_info_response_time(self):
        """Test system info response time."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/system/info")
        end_time = time.time()

        assert response.status_code == 200
        # Should respond within 1 second
        assert (end_time - start_time) < 1.0

    def test_system_capabilities_response_time(self):
        """Test system capabilities response time."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/system/capabilities")
        end_time = time.time()

        assert response.status_code == 200
        # Should respond within 1 second
        assert (end_time - start_time) < 1.0


class TestSystemAPIEdgeCases:
    """Test edge cases for system API."""

    def test_system_status_consistency(self):
        """Test that system status is consistent across calls."""
        response1 = client.get("/api/v1/system/status")
        response2 = client.get("/api/v1/system/status")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Both should have similar structure
        data1 = response1.json()["data"]
        data2 = response2.json()["data"]

        assert set(data1.keys()) == set(data2.keys())

    def test_system_config_no_sensitive_data(self):
        """Test that system config doesn't expose sensitive data."""
        response = client.get("/api/v1/system/config")
        assert response.status_code == 200

        config_str = response.text.lower()

        # Check that no sensitive data is exposed
        sensitive_terms = [
            "password",
            "secret",
            "key",
            "token",
            "credential",
            "private",
            "auth",
        ]

        for term in sensitive_terms:
            # Allow the word "key" in context like "api_key_configured"
            if term == "key":
                assert "database_key" not in config_str
                assert "secret_key" not in config_str
            else:
                assert term not in config_str
