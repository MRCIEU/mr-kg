"""API client service for communicating with the MR-KG API backend."""

import logging
from dataclasses import dataclass
from typing import Any

import httpx
import streamlit as st

from config import get_settings

logger = logging.getLogger(__name__)


# ==== Result types ====


@dataclass
class AutocompleteResult:
    """Result of an autocomplete request.

    Distinguishes between successful results (possibly empty), API errors,
    and connection errors.
    """

    items: list
    success: bool
    error_message: str | None = None

    @classmethod
    def success_result(cls, items: list) -> "AutocompleteResult":
        """Create a successful result."""
        return cls(items=items, success=True, error_message=None)

    @classmethod
    def error_result(cls, message: str) -> "AutocompleteResult":
        """Create an error result."""
        return cls(items=[], success=False, error_message=message)


# ==== HTTP client utilities ====


def _get_base_url() -> str:
    """Get the API base URL from settings."""
    settings = get_settings()
    return settings.api_url


def _make_request(
    method: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict | None:
    """Make an HTTP request to the API.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path (e.g., /api/studies)
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        JSON response as dict, or None on error
    """
    base_url = _get_base_url()
    url = f"{base_url}{endpoint}"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e}")
        if e.response.status_code == 404:
            return None
        raise
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise


# ==== Studies endpoints ====


@st.cache_data(ttl=300)
def search_studies(
    q: str | None = None,
    trait: str | None = None,
    model: str = "gpt-5",
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Search for studies with optional filtering.

    Args:
        q: Search query for title or PMID
        trait: Filter by trait label
        model: Extraction model filter
        limit: Maximum results to return
        offset: Pagination offset

    Returns:
        Dict with total count and list of studies
    """
    params: dict[str, Any] = {
        "model": model,
        "limit": limit,
        "offset": offset,
    }
    if q:
        params["q"] = q
    if trait:
        params["trait"] = trait

    result = _make_request("GET", "/api/studies", params=params)
    if result is None:
        return {"total": 0, "limit": limit, "offset": offset, "studies": []}
    return result


def get_extraction(pmid: str, model: str = "gpt-5") -> dict | None:
    """Get extraction results for a specific study.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model

    Returns:
        Extraction data or None if not found
    """
    params = {"model": model}
    return _make_request(
        "GET", f"/api/studies/{pmid}/extraction", params=params
    )


# ==== Similarity endpoints ====


def get_similar_by_trait(
    pmid: str,
    model: str = "gpt-5",
    limit: int = 10,
) -> dict | None:
    """Get similar studies by trait profile similarity.

    Args:
        pmid: PubMed ID of the query study
        model: Extraction model
        limit: Maximum similar studies to return

    Returns:
        Similarity data or None if not found
    """
    params = {"model": model, "limit": limit}
    return _make_request(
        "GET", f"/api/studies/{pmid}/similar/trait", params=params
    )


def get_similar_by_evidence(
    pmid: str,
    model: str = "gpt-5",
    limit: int = 10,
) -> dict | None:
    """Get similar studies by evidence profile similarity.

    Args:
        pmid: PubMed ID of the query study
        model: Extraction model
        limit: Maximum similar studies to return

    Returns:
        Similarity data or None if not found
    """
    params = {"model": model, "limit": limit}
    return _make_request(
        "GET", f"/api/studies/{pmid}/similar/evidence", params=params
    )


# ==== Autocomplete endpoints ====


@st.cache_data(ttl=300)
def autocomplete_traits(
    q: str, model: str = "gpt-5", limit: int = 20
) -> list[str]:
    """Get trait autocomplete suggestions.

    Args:
        q: Search term (prefix match)
        model: Extraction model filter
        limit: Maximum suggestions to return

    Returns:
        List of trait labels matching the search term that have
        extraction results for the specified model
    """
    if len(q) < 2:
        return []

    params = {"q": q, "model": model, "limit": limit}
    result = _make_request("GET", "/api/traits/autocomplete", params=params)
    if result is None:
        return []
    return result.get("traits", [])


@st.cache_data(ttl=300)
def autocomplete_studies(
    q: str, model: str = "gpt-5", limit: int = 20
) -> list[dict]:
    """Get study autocomplete suggestions.

    Args:
        q: Search term (substring match in title)
        model: Extraction model filter
        limit: Maximum suggestions to return

    Returns:
        List of dicts with pmid and title for studies that have
        extraction results for the specified model
    """
    if len(q) < 2:
        return []

    params = {"q": q, "model": model, "limit": limit}
    result = _make_request("GET", "/api/studies/autocomplete", params=params)
    if result is None:
        return []
    return result.get("studies", [])


def autocomplete_traits_with_status(
    q: str, model: str = "gpt-5", limit: int = 20
) -> AutocompleteResult:
    """Get trait autocomplete suggestions with error status.

    Unlike autocomplete_traits, this function distinguishes between
    "no results found" and "API error".

    Args:
        q: Search term (prefix match)
        model: Extraction model filter
        limit: Maximum suggestions to return

    Returns:
        AutocompleteResult with items and success/error status
    """
    if len(q) < 2:
        return AutocompleteResult.success_result([])

    params = {"q": q, "model": model, "limit": limit}
    try:
        result = _make_request(
            "GET", "/api/traits/autocomplete", params=params
        )
        if result is None:
            return AutocompleteResult.error_result("API returned no response")
        return AutocompleteResult.success_result(result.get("traits", []))
    except httpx.RequestError as e:
        return AutocompleteResult.error_result(f"Connection error: {e}")
    except httpx.HTTPStatusError as e:
        return AutocompleteResult.error_result(
            f"API error: {e.response.status_code}"
        )


def autocomplete_studies_with_status(
    q: str, model: str = "gpt-5", limit: int = 20
) -> AutocompleteResult:
    """Get study autocomplete suggestions with error status.

    Unlike autocomplete_studies, this function distinguishes between
    "no results found" and "API error".

    Args:
        q: Search term (substring match in title)
        model: Extraction model filter
        limit: Maximum suggestions to return

    Returns:
        AutocompleteResult with items and success/error status
    """
    if len(q) < 2:
        return AutocompleteResult.success_result([])

    params = {"q": q, "model": model, "limit": limit}
    try:
        result = _make_request(
            "GET", "/api/studies/autocomplete", params=params
        )
        if result is None:
            return AutocompleteResult.error_result("API returned no response")
        return AutocompleteResult.success_result(result.get("studies", []))
    except httpx.RequestError as e:
        return AutocompleteResult.error_result(f"Connection error: {e}")
    except httpx.HTTPStatusError as e:
        return AutocompleteResult.error_result(
            f"API error: {e.response.status_code}"
        )


# ==== Statistics endpoint ====


@st.cache_data(ttl=3600)
def get_statistics() -> dict | None:
    """Get resource-wide statistics.

    Returns:
        Statistics data or None on error
    """
    return _make_request("GET", "/api/statistics")


# ==== Health check ====


def check_health() -> dict:
    """Check API health status.

    Returns:
        Health status dict with status and database connectivity info
    """
    try:
        result = _make_request("GET", "/api/health", timeout=5.0)
        if result is None:
            return {"status": "unhealthy", "databases": {}}
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "error": str(e), "databases": {}}
