"""Pytest fixtures and configuration for integration tests.

Integration tests require the API service to be running and databases present.
Tests are skipped when the service is unavailable.
"""

import os

import httpx
import pytest


# ==== Configuration ====


def get_api_base_url() -> str:
    """Get the API base URL from environment or default.

    Returns:
        API base URL string
    """
    res = os.environ.get("API_URL", "http://localhost:8000")
    return res


# ==== Markers ====


def is_api_available() -> bool:
    """Check if the API service is available.

    Returns:
        True if API is reachable and healthy
    """
    base_url = get_api_base_url()
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                return data.get("status") in ("healthy", "degraded")
    except Exception:
        pass
    return False


# ---- Skip marker for when API is unavailable ----
requires_api = pytest.mark.skipif(
    not is_api_available(),
    reason="API service is not available",
)


# ==== Fixtures ====


@pytest.fixture
def api_base_url() -> str:
    """Provide API base URL for tests.

    Returns:
        API base URL string
    """
    return get_api_base_url()


@pytest.fixture
def api_client(api_base_url: str):
    """Create an HTTP client for API requests.

    Args:
        api_base_url: Base URL for the API

    Yields:
        httpx.Client instance
    """
    with httpx.Client(base_url=api_base_url, timeout=30.0) as client:
        yield client


# ==== Helper functions ====


def make_api_request(
    client: httpx.Client,
    endpoint: str,
    params: dict | None = None,
) -> httpx.Response:
    """Make a GET request to an API endpoint.

    Args:
        client: HTTP client instance
        endpoint: API endpoint path
        params: Query parameters

    Returns:
        HTTP response object
    """
    res = client.get(endpoint, params=params)
    return res
