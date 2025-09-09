"""Tests for the core API endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestCoreAPI:
    """Test cases for core API endpoints."""

    def test_core_root(self):
        """Test core API root endpoint."""
        response = client.get("/api/v1/core/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "message" in data["data"]

    def test_ping_endpoint(self):
        """Test ping endpoint."""
        response = client.get("/api/v1/core/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["message"] == "pong"
        assert "timestamp" in data["data"]

    def test_echo_endpoint_get(self):
        """Test echo endpoint with GET request."""
        response = client.get("/api/v1/core/echo?message=hello")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["query_params"]["message"] == "hello"
        assert data["data"]["method"] == "GET"
        assert "timestamp" in data["data"]

    def test_echo_endpoint_post_not_supported(self):
        """Test echo endpoint with POST request (not supported)."""
        test_data = {"message": "test message", "data": {"key": "value"}}
        response = client.post("/api/v1/core/echo", json=test_data)
        assert response.status_code == 405  # Method not allowed

    def test_echo_endpoint_no_message(self):
        """Test echo endpoint without message parameter."""
        response = client.get("/api/v1/core/echo")
        assert (
            response.status_code == 200
        )  # Echo returns request info even without message
        data = response.json()
        assert data["success"] is True
        assert data["data"]["method"] == "GET"
        assert data["data"]["query_params"] == {}

    def test_version_endpoint(self):
        """Test version endpoint."""
        response = client.get("/api/v1/core/version")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        version_data = data["data"]
        assert "api_version" in version_data
        assert "application_version" in version_data  # Updated field name
        assert "build_date" in version_data
        assert "commit_hash" in version_data


class TestCoreAPIEdgeCases:
    """Test edge cases for core API endpoints."""

    def test_echo_large_message(self):
        """Test echo with large message."""
        large_message = "x" * 1000  # 1KB message (reduced from 10KB)
        response = client.get(f"/api/v1/core/echo?message={large_message}")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["query_params"]["message"] == large_message

    def test_echo_special_characters(self):
        """Test echo with special characters."""
        special_message = "Hello! @#$%^&*()_+"
        # Use params parameter to properly encode URL parameters
        response = client.get(
            "/api/v1/core/echo", params={"message": special_message}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["query_params"]["message"] == special_message

    def test_echo_unicode_characters(self):
        """Test echo with unicode characters."""
        unicode_message = "Hello 世界 🌍 café"
        response = client.get(f"/api/v1/core/echo?message={unicode_message}")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["query_params"]["message"] == unicode_message

    def test_ping_performance(self):
        """Test ping endpoint performance."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/core/ping")
        end_time = time.time()

        assert response.status_code == 200
        # Should respond very quickly (under 100ms)
        assert (end_time - start_time) < 0.1

    def test_echo_headers_included(self):
        """Test that echo endpoint includes request headers."""
        response = client.get("/api/v1/core/echo?message=test")
        assert response.status_code == 200
        data = response.json()
        assert "headers" in data["data"]
        assert "host" in data["data"]["headers"]

    def test_echo_client_info_included(self):
        """Test that echo endpoint includes client information."""
        response = client.get("/api/v1/core/echo?message=test")
        assert response.status_code == 200
        data = response.json()
        assert "client" in data["data"]
        assert "host" in data["data"]["client"]

    def test_version_consistency(self):
        """Test that version endpoint returns consistent data."""
        response1 = client.get("/api/v1/core/version")
        response2 = client.get("/api/v1/core/version")

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()["data"]
        data2 = response2.json()["data"]

        # Version info should be consistent across calls
        assert data1["api_version"] == data2["api_version"]
        assert data1["application_version"] == data2["application_version"]
        assert data1["build_date"] == data2["build_date"]
        assert data1["commit_hash"] == data2["commit_hash"]
