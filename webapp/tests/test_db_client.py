"""Tests for the database client service."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# Mock streamlit before importing db_client
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit module for all tests."""
    mock_st = MagicMock()
    # Make cache_data a passthrough decorator
    mock_st.cache_data = lambda **kwargs: lambda f: f
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        yield mock_st


class TestSearchStudies:
    """Tests for search_studies function."""

    def test_search_studies_with_query(
        self,
        mock_api_response: dict[str, Any],
    ) -> None:
        """Test searching studies with a query string."""
        with patch("services.db_client.get_studies") as mock_get:
            mock_get.return_value = (
                mock_api_response["total"],
                mock_api_response["studies"],
            )

            from services.db_client import search_studies

            result = search_studies(q="body mass index", model="gpt-5")

            assert result["total"] == 2
            assert len(result["studies"]) == 2
            assert result["studies"][0]["pmid"] == "12345678"

            mock_get.assert_called_once()

    def test_search_studies_with_trait_filter(
        self,
        mock_api_response: dict[str, Any],
    ) -> None:
        """Test searching studies filtered by trait."""
        with patch("services.db_client.get_studies") as mock_get:
            mock_get.return_value = (
                mock_api_response["total"],
                mock_api_response["studies"],
            )

            from services.db_client import search_studies

            result = search_studies(
                trait="body mass index",
                model="gpt-5",
                limit=10,
            )

            assert result["total"] == 2
            mock_get.assert_called_once_with(
                q=None,
                trait="body mass index",
                model="gpt-5",
                limit=10,
                offset=0,
            )

    def test_search_studies_returns_empty_on_error(self) -> None:
        """Test that search_studies returns empty on error."""
        with patch("services.db_client.get_studies") as mock_get:
            mock_get.side_effect = Exception("Database error")

            from services.db_client import search_studies

            result = search_studies(q="test")

            assert result["total"] == 0
            assert result["studies"] == []


class TestGetExtraction:
    """Tests for get_extraction function."""

    def test_get_extraction_success(
        self,
        mock_extraction_response: dict[str, Any],
    ) -> None:
        """Test successful extraction retrieval."""
        with patch("services.db_client.get_study_extraction") as mock_get:
            mock_get.return_value = mock_extraction_response

            from services.db_client import get_extraction

            result = get_extraction("12345678", "gpt-5")

            assert result is not None
            assert result["pmid"] == "12345678"
            assert len(result["traits"]) == 2
            assert len(result["results"]) == 1

    def test_get_extraction_not_found(self) -> None:
        """Test extraction returns None when study not found."""
        with patch("services.db_client.get_study_extraction") as mock_get:
            mock_get.return_value = None

            from services.db_client import get_extraction

            result = get_extraction("nonexistent", "gpt-5")

            assert result is None


class TestSimilarityEndpoints:
    """Tests for similarity endpoint functions."""

    def test_get_similar_by_trait(
        self,
        mock_trait_similarity_response: dict[str, Any],
    ) -> None:
        """Test trait similarity retrieval."""
        with patch("services.db_client._get_trait") as mock_get:
            mock_get.return_value = mock_trait_similarity_response

            from services.db_client import get_similar_by_trait

            result = get_similar_by_trait("12345678", "gpt-5", limit=10)

            assert result is not None
            assert result["query_pmid"] == "12345678"
            assert len(result["similar_studies"]) == 2

    def test_get_similar_by_evidence(
        self,
        mock_evidence_similarity_response: dict[str, Any],
    ) -> None:
        """Test evidence similarity retrieval."""
        with patch("services.db_client._get_evidence") as mock_get:
            mock_get.return_value = mock_evidence_similarity_response

            from services.db_client import get_similar_by_evidence

            result = get_similar_by_evidence("12345678", "gpt-5", limit=10)

            assert result is not None
            assert result["query_pmid"] == "12345678"
            assert len(result["similar_studies"]) == 2


class TestAutocomplete:
    """Tests for autocomplete functions."""

    def test_autocomplete_traits(self) -> None:
        """Test trait autocomplete suggestions."""
        with patch("services.db_client._search_traits") as mock_search:
            mock_search.return_value = ["body mass index", "blood pressure"]

            from services.db_client import autocomplete_traits

            result = autocomplete_traits("body", limit=20)

            assert len(result) == 2
            assert "body mass index" in result

    def test_autocomplete_traits_short_query(self) -> None:
        """Test that short queries return empty list."""
        from services.db_client import autocomplete_traits

        result = autocomplete_traits("b")

        assert result == []

    def test_autocomplete_studies(self) -> None:
        """Test study autocomplete suggestions."""
        mock_response = [
            {"pmid": "12345678", "title": "Test study"},
            {"pmid": "23456789", "title": "Another study"},
        ]

        with patch("services.db_client._search_studies") as mock_search:
            mock_search.return_value = mock_response

            from services.db_client import autocomplete_studies

            result = autocomplete_studies("test", limit=20)

            assert len(result) == 2
            assert result[0]["pmid"] == "12345678"


class TestStatistics:
    """Tests for statistics function."""

    def test_get_statistics(
        self,
        mock_statistics_response: dict[str, Any],
    ) -> None:
        """Test statistics retrieval."""
        with (
            patch("services.db_client.get_overall_statistics") as mock_overall,
            patch("services.db_client.get_model_similarity_stats") as mock_sim,
            patch("services.db_client.get_model_evidence_stats") as mock_ev,
            patch("services.db_client.get_metric_availability") as mock_metric,
        ):
            mock_overall.return_value = mock_statistics_response["overall"]
            mock_sim.return_value = mock_statistics_response[
                "model_similarity_stats"
            ]
            mock_ev.return_value = mock_statistics_response[
                "model_evidence_stats"
            ]
            mock_metric.return_value = {}

            from services.db_client import get_statistics

            result = get_statistics()

            assert result is not None
            assert result["overall"]["total_papers"] == 15635
            assert len(result["model_similarity_stats"]) == 2


class TestDatabaseHealth:
    """Tests for database health check function."""

    def test_check_database_health_success(self) -> None:
        """Test successful database health check."""
        with patch("services.db_client.Path") as mock_path:
            mock_path.return_value.exists.return_value = True

            from services.db_client import check_database_health

            result = check_database_health()

            assert result["status"] == "healthy"
            assert result["databases"]["vector_store"] is True

    def test_check_database_health_unhealthy(self) -> None:
        """Test health check when database missing."""
        with patch("services.db_client.Path") as mock_path:
            # First database exists, others don't
            mock_path.return_value.exists.side_effect = [True, False, True]

            from services.db_client import check_database_health

            result = check_database_health()

            assert result["status"] == "unhealthy"
