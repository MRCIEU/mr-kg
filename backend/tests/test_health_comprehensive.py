"""Comprehensive tests for the health API endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthAPI:
    """Test cases for health API endpoints."""

    def test_health_root(self):
        """Test basic health endpoint."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"

    def test_health_live(self):
        """Test liveness probe endpoint."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "alive" in data["data"]
        assert "timestamp" in data["data"]

    def test_health_ready(self):
        """Test readiness probe endpoint."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "ready" in data["data"]
        assert "checks" in data["data"]

    @patch("app.core.dependencies.check_database_connectivity")
    def test_health_database_success(self, mock_check_db):
        """Test database health check with successful connection."""
        mock_check_db.return_value = {
            "status": "healthy",
            "connection_pool": {"active": 2, "idle": 8, "total": 10},
            "response_time_ms": 15.5,
        }

        response = client.get("/api/v1/health/database")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        db_health = data["data"]
        assert db_health["status"] == "healthy"
        assert "pool_status" in db_health

    @patch("app.api.v1.health.check_database_connectivity")
    def test_health_database_failure(self, mock_check_db):
        """Test database health check with connection failure."""
        mock_check_db.return_value = {
            "status": "unhealthy",
            "error": "Connection refused",
            "response_time_ms": None,
        }

        response = client.get("/api/v1/health/database")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        db_health = data["data"]
        assert db_health["status"] == "unhealthy"
        assert "error" in db_health

    def test_health_detailed(self):
        """Test detailed health check endpoint."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data  # Direct response, not wrapped
        assert "timestamp" in data
        assert "vector_store" in data
        assert "trait_profile" in data

    def test_health_pool(self):
        """Test connection pool health endpoint."""
        response = client.get("/api/v1/health/pool")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        pool_info = data["data"]
        assert "initialized" in pool_info

    def test_health_system(self):
        """Test system health endpoint."""
        response = client.get("/api/v1/health/system")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        system_health = data["data"]
        assert "application" in system_health
        assert "version" in system_health
        assert "disk_usage" in system_health
        assert "load_average" in system_health

    def test_health_metrics(self):
        """Test health metrics endpoint."""
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        metrics = data["data"]
        assert "requests_total" in metrics
        assert "response_time_avg" in metrics
        assert "active_connections" in metrics


class TestHealthAPIKubernetes:
    """Test health API endpoints for Kubernetes compatibility."""

    def test_liveness_probe_format(self):
        """Test liveness probe returns Kubernetes-compatible format."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200

        # Should be quick response for liveness
        data = response.json()
        assert data["success"] is True
        assert "timestamp" in data["data"]

    def test_readiness_probe_format(self):
        """Test readiness probe returns detailed checks."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200

        data = response.json()
        checks = data["data"]["checks"]
        assert isinstance(checks, dict)

        # Should include database check
        assert "database" in checks

    @patch("app.api.v1.health.check_database_connectivity")
    def test_readiness_probe_database_failure(self, mock_check_db):
        """Test readiness probe fails when database is down."""
        mock_check_db.return_value = {
            "status": "unhealthy",
            "error": "Database connection failed",
        }

        response = client.get("/api/v1/health/ready")
        # Should return 503 when database is down
        assert response.status_code == 503
        data = response.json()
        # The response is wrapped as an error response
        assert "error" in data


class TestHealthAPIPerformance:
    """Test health API performance requirements."""

    def test_liveness_probe_performance(self):
        """Test liveness probe responds quickly."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/health/live")
        end_time = time.time()

        assert response.status_code == 200
        # Liveness should be very fast (under 50ms)
        assert (end_time - start_time) < 0.05

    def test_basic_health_performance(self):
        """Test basic health endpoint performance."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/health/")
        end_time = time.time()

        assert response.status_code == 200
        # Basic health should be fast (under 100ms)
        assert (end_time - start_time) < 0.1

    def test_detailed_health_performance(self):
        """Test detailed health endpoint reasonable performance."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/health/detailed")
        end_time = time.time()

        assert response.status_code == 200
        # Detailed health can be slower but should be under 2 seconds
        assert (end_time - start_time) < 2.0


class TestHealthAPIEdgeCases:
    """Test edge cases for health API."""

    def test_health_endpoints_consistent_format(self):
        """Test all health endpoints return consistent format."""
        endpoints = [
            "/api/v1/health/",
            "/api/v1/health/live",
            "/api/v1/health/ready",
            "/api/v1/health/system",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            data = response.json()
            assert "success" in data
            assert "data" in data
            assert data["success"] is True

    def test_health_multiple_requests(self):
        """Test health endpoints handle multiple concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get("/api/v1/health/")

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        # All should succeed
        for response in responses:
            assert response.status_code == 200

    @patch("app.core.dependencies.get_database_pool")
    def test_health_database_pool_unavailable(self, mock_get_pool):
        """Test health when database pool is unavailable."""
        mock_get_pool.side_effect = Exception("Pool not available")

        response = client.get("/api/v1/health/pool")
        # Should handle gracefully
        assert response.status_code in [200, 500]

    def test_health_system_resource_monitoring(self):
        """Test system health provides useful resource information."""
        response = client.get("/api/v1/health/system")
        assert response.status_code == 200

        data = response.json()["data"]

        # CPU usage should be a reasonable value
        if "cpu_usage" in data and data["cpu_usage"] is not None:
            assert 0 <= data["cpu_usage"] <= 100

        # Memory usage should be structured
        if "memory_usage" in data:
            memory = data["memory_usage"]
            if isinstance(memory, dict):
                assert "percent" in memory or "used" in memory
