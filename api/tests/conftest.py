"""Pytest fixtures for API tests."""

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


# ==== Mock data ====


MOCK_STUDY = {
    "pmid": "12345678",
    "title": "Test study on body mass index and type 2 diabetes",
    "pub_date": "2023-01-15",
    "journal": "Nature",
    "model": "gpt-5",
}

MOCK_EXTRACTION = {
    "pmid": "12345678",
    "model": "gpt-5",
    "title": "Test study on body mass index and type 2 diabetes",
    "pub_date": "2023-01-15",
    "journal": "Nature",
    "abstract": "Background: This study investigates...",
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
        "exposures": ["body mass index"],
        "outcomes": ["type 2 diabetes"],
    },
}

MOCK_TRAIT_SIMILARITY = {
    "query_pmid": "12345678",
    "query_model": "gpt-5",
    "query_title": "Test study on body mass index and type 2 diabetes",
    "query_trait_count": 2,
    "similar_studies": [
        {
            "pmid": "23456789",
            "title": "Related study on BMI",
            "trait_profile_similarity": 0.85,
            "trait_jaccard_similarity": 0.42,
            "trait_count": 3,
        },
        {
            "pmid": "34567890",
            "title": "Another MR study",
            "trait_profile_similarity": 0.72,
            "trait_jaccard_similarity": 0.33,
            "trait_count": 4,
        },
    ],
}

MOCK_EVIDENCE_SIMILARITY = {
    "query_pmid": "12345678",
    "query_model": "gpt-5",
    "query_title": "Test study on body mass index and type 2 diabetes",
    "query_result_count": 5,
    "similar_studies": [
        {
            "pmid": "23456789",
            "title": "Consistent study",
            "direction_concordance": 0.92,
            "matched_pairs": 5,
            "match_type_exact": True,
            "match_type_fuzzy": False,
            "match_type_efo": False,
            "matched_evidence_pairs": [
                {
                    "query_exposure": "body mass index",
                    "query_outcome": "type 2 diabetes",
                    "query_direction": "increases",
                    "similar_exposure": "body mass index",
                    "similar_outcome": "type 2 diabetes",
                    "similar_direction": "increases",
                    "match_type": "exact",
                },
            ],
        },
        {
            "pmid": "34567890",
            "title": "Another consistent study",
            "direction_concordance": 0.75,
            "matched_pairs": 3,
            "match_type_exact": False,
            "match_type_fuzzy": True,
            "match_type_efo": False,
            "matched_evidence_pairs": [
                {
                    "query_exposure": "BMI",
                    "query_outcome": "diabetes",
                    "query_direction": "increases",
                    "similar_exposure": "body mass index",
                    "similar_outcome": "type 2 diabetes",
                    "similar_direction": "increases",
                    "match_type": "fuzzy",
                },
            ],
        },
    ],
}

MOCK_STATISTICS = {
    "overall": {
        "total_papers": 15635,
        "total_traits": 75121,
        "total_models": 7,
        "total_extractions": 50402,
    },
    "model_similarity_stats": [
        {
            "model": "gpt-5",
            "total_combinations": 8400,
            "avg_trait_count": 5.2,
            "min_trait_count": 1,
            "max_trait_count": 25,
            "total_similarity_pairs": 84000,
        },
    ],
    "model_evidence_stats": [
        {
            "model": "gpt-5",
            "total_combinations": 8400,
            "avg_result_count": 3.5,
            "avg_completeness": 0.85,
            "min_result_count": 1,
            "max_result_count": 50,
            "total_similarity_pairs": 84000,
        },
    ],
}


# ==== Fixtures ====


@pytest.fixture
def mock_vector_store():
    """Mock vector store repository functions."""
    with patch("app.routers.studies.vs_repo") as mock:
        mock.get_studies.return_value = (1, [MOCK_STUDY])
        mock.get_study_extraction.return_value = MOCK_EXTRACTION
        mock.search_traits.return_value = [
            "body mass index",
            "blood pressure",
        ]
        mock.search_studies.return_value = [
            {"pmid": "12345678", "title": "Test study"},
        ]
        yield mock


@pytest.fixture
def mock_trait_profile():
    """Mock trait profile repository functions."""
    with patch("app.routers.similar.tp_repo") as mock:
        mock.get_similar_by_trait.return_value = MOCK_TRAIT_SIMILARITY
        yield mock


@pytest.fixture
def mock_evidence_profile():
    """Mock evidence profile repository functions."""
    with patch("app.routers.similar.ep_repo") as mock:
        mock.get_similar_by_evidence.return_value = MOCK_EVIDENCE_SIMILARITY
        yield mock


@pytest.fixture
def mock_statistics():
    """Mock statistics repository functions."""
    with patch("app.routers.studies.stats_repo") as mock:
        mock.get_overall_statistics.return_value = MOCK_STATISTICS["overall"]
        mock.get_model_similarity_stats.return_value = MOCK_STATISTICS[
            "model_similarity_stats"
        ]
        mock.get_model_evidence_stats.return_value = MOCK_STATISTICS[
            "model_evidence_stats"
        ]
        yield mock


@pytest.fixture
def mock_database_connections():
    """Mock database connection functions for health checks."""
    with (
        patch("app.main.get_vector_store_connection") as mock_vs,
        patch("app.main.get_trait_profile_connection") as mock_tp,
        patch("app.main.get_evidence_profile_connection") as mock_ep,
        patch("app.main.Path") as mock_path,
    ):
        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock connection execute
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (1,)
        mock_vs.return_value = mock_conn
        mock_tp.return_value = mock_conn
        mock_ep.return_value = mock_conn

        yield {
            "vector_store": mock_vs,
            "trait_profile": mock_tp,
            "evidence_profile": mock_ep,
        }


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for anyio."""
    return "asyncio"
