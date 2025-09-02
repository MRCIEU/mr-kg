"""Basic health check tests."""

import pytest
from fastapi.testclient import TestClient


def test_health_check():
    """Test health check endpoint returns healthy status."""
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
