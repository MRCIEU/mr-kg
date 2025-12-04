"""Pytest fixtures for webapp tests."""

from typing import Any
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_api_response() -> dict[str, Any]:
    """Sample API response for studies endpoint."""
    return {
        "total": 2,
        "limit": 20,
        "offset": 0,
        "studies": [
            {
                "pmid": "12345678",
                "title": "Test study on body mass index",
                "pub_date": "2023-01-15",
                "journal": "Nature",
                "model": "gpt-5",
            },
            {
                "pmid": "23456789",
                "title": "Mendelian randomization study",
                "pub_date": "2022-06-20",
                "journal": "BMJ",
                "model": "gpt-5",
            },
        ],
    }


@pytest.fixture
def mock_extraction_response() -> dict[str, Any]:
    """Sample API response for extraction endpoint."""
    return {
        "pmid": "12345678",
        "model": "gpt-5",
        "title": "Test study on body mass index",
        "pub_date": "2023-01-15",
        "journal": "Nature",
        "abstract": "This study investigates the causal relationship...",
        "traits": [
            {
                "trait_index": 1,
                "trait_label": "body mass index",
                "trait_id_in_result": "E1",
            },
            {
                "trait_index": 2,
                "trait_label": "type 2 diabetes",
                "trait_id_in_result": "O1",
            },
        ],
        "results": [
            {
                "exposure": "body mass index",
                "outcome": "type 2 diabetes",
                "beta": 0.45,
                "odds_ratio": None,
                "hazard_ratio": None,
                "ci_lower": 0.32,
                "ci_upper": 0.58,
                "p_value": 1.2e-8,
                "direction": "increases",
            },
        ],
        "metadata": {
            "exposures": [{"id": "E1", "label": "body mass index"}],
            "outcomes": [{"id": "O1", "label": "type 2 diabetes"}],
        },
    }


@pytest.fixture
def mock_trait_similarity_response() -> dict[str, Any]:
    """Sample API response for trait similarity endpoint."""
    return {
        "query_pmid": "12345678",
        "query_model": "gpt-5",
        "query_title": "Test study on body mass index",
        "query_trait_count": 5,
        "similar_studies": [
            {
                "pmid": "23456789",
                "title": "Related MR study",
                "trait_profile_similarity": 0.85,
                "trait_jaccard_similarity": 0.42,
                "trait_count": 6,
            },
            {
                "pmid": "34567890",
                "title": "Another study",
                "trait_profile_similarity": 0.72,
                "trait_jaccard_similarity": 0.31,
                "trait_count": 4,
            },
        ],
    }


@pytest.fixture
def mock_evidence_similarity_response() -> dict[str, Any]:
    """Sample API response for evidence similarity endpoint."""
    return {
        "query_pmid": "12345678",
        "query_model": "gpt-5",
        "query_title": "Test study on body mass index",
        "query_result_count": 3,
        "similar_studies": [
            {
                "pmid": "23456789",
                "title": "Consistent study",
                "direction_concordance": 0.92,
                "matched_pairs": 5,
                "match_type_exact": True,
                "match_type_fuzzy": False,
                "match_type_efo": False,
            },
            {
                "pmid": "34567890",
                "title": "Another study",
                "direction_concordance": 0.65,
                "matched_pairs": 2,
                "match_type_exact": False,
                "match_type_fuzzy": True,
                "match_type_efo": False,
            },
        ],
    }


@pytest.fixture
def mock_statistics_response() -> dict[str, Any]:
    """Sample API response for statistics endpoint."""
    return {
        "overall": {
            "total_papers": 15635,
            "total_traits": 75121,
            "total_models": 7,
            "total_extractions": 50402,
        },
        "model_similarity_stats": [
            {
                "model": "gpt-5",
                "extractions": 8400,
                "avg_traits": 5.2,
                "similarities": 84000,
            },
            {
                "model": "gpt-4-1",
                "extractions": 8400,
                "avg_traits": 4.8,
                "similarities": 84000,
            },
        ],
        "model_evidence_stats": [
            {
                "model": "gpt-5",
                "extractions": 8400,
                "avg_results": 3.1,
                "similarities": 72000,
            },
            {
                "model": "gpt-4-1",
                "extractions": 8400,
                "avg_results": 2.9,
                "similarities": 72000,
            },
        ],
    }


@pytest.fixture
def mock_health_response() -> dict[str, Any]:
    """Sample API response for health endpoint."""
    return {
        "status": "healthy",
        "databases": {
            "vector_store": True,
            "trait_profile": True,
            "evidence_profile": True,
        },
    }


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing API calls."""
    with patch("services.api_client.httpx.Client") as mock_client:
        yield mock_client
