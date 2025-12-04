"""Tests for the API client service."""

from typing import Any
from unittest.mock import patch


class TestSearchStudies:
    """Tests for search_studies function."""

    def test_search_studies_with_query(
        self,
        mock_api_response: dict[str, Any],
    ) -> None:
        """Test searching studies with a query string."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_api_response

            from services.api_client import search_studies

            # Clear cache to ensure fresh call
            search_studies.clear()

            result = search_studies(q="body mass index", model="gpt-5")

            assert result["total"] == 2
            assert len(result["studies"]) == 2
            assert result["studies"][0]["pmid"] == "12345678"

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert call_args[0][1] == "/api/studies"

    def test_search_studies_with_trait_filter(
        self,
        mock_api_response: dict[str, Any],
    ) -> None:
        """Test searching studies filtered by trait."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_api_response

            from services.api_client import search_studies

            search_studies.clear()

            result = search_studies(
                trait="body mass index",
                model="gpt-5",
                limit=10,
            )

            assert result["total"] == 2
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["trait"] == "body mass index"

    def test_search_studies_returns_empty_on_none(self) -> None:
        """Test that search_studies returns empty dict on None response."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = None

            from services.api_client import search_studies

            search_studies.clear()

            result = search_studies(q="nonexistent")

            assert result["total"] == 0
            assert result["studies"] == []


class TestGetExtraction:
    """Tests for get_extraction function."""

    def test_get_extraction_success(
        self,
        mock_extraction_response: dict[str, Any],
    ) -> None:
        """Test successful extraction retrieval."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_extraction_response

            from services.api_client import get_extraction

            result = get_extraction("12345678", "gpt-5")

            assert result is not None
            assert result["pmid"] == "12345678"
            assert len(result["traits"]) == 2
            assert len(result["results"]) == 1

    def test_get_extraction_not_found(self) -> None:
        """Test extraction returns None when study not found."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = None

            from services.api_client import get_extraction

            result = get_extraction("nonexistent", "gpt-5")

            assert result is None


class TestSimilarityEndpoints:
    """Tests for similarity endpoint functions."""

    def test_get_similar_by_trait(
        self,
        mock_trait_similarity_response: dict[str, Any],
    ) -> None:
        """Test trait similarity retrieval."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_trait_similarity_response

            from services.api_client import get_similar_by_trait

            result = get_similar_by_trait("12345678", "gpt-5", limit=10)

            assert result is not None
            assert result["query_pmid"] == "12345678"
            assert len(result["similar_studies"]) == 2

    def test_get_similar_by_evidence(
        self,
        mock_evidence_similarity_response: dict[str, Any],
    ) -> None:
        """Test evidence similarity retrieval."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_evidence_similarity_response

            from services.api_client import get_similar_by_evidence

            result = get_similar_by_evidence("12345678", "gpt-5", limit=10)

            assert result is not None
            assert result["query_pmid"] == "12345678"
            assert len(result["similar_studies"]) == 2


class TestAutocomplete:
    """Tests for autocomplete functions."""

    def test_autocomplete_traits(self) -> None:
        """Test trait autocomplete suggestions."""
        mock_response = {"traits": ["body mass index", "blood pressure"]}

        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_response

            from services.api_client import autocomplete_traits

            autocomplete_traits.clear()

            result = autocomplete_traits("body", limit=20)

            assert len(result) == 2
            assert "body mass index" in result

    def test_autocomplete_traits_short_query(self) -> None:
        """Test that short queries return empty list."""
        from services.api_client import autocomplete_traits

        autocomplete_traits.clear()

        result = autocomplete_traits("b")

        assert result == []

    def test_autocomplete_studies(self) -> None:
        """Test study autocomplete suggestions."""
        mock_response = {
            "studies": [
                {"pmid": "12345678", "title": "Test study"},
                {"pmid": "23456789", "title": "Another study"},
            ]
        }

        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_response

            from services.api_client import autocomplete_studies

            autocomplete_studies.clear()

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
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_statistics_response

            from services.api_client import get_statistics

            get_statistics.clear()

            result = get_statistics()

            assert result is not None
            assert result["overall"]["total_papers"] == 15635
            assert len(result["model_similarity_stats"]) == 2


class TestHealthCheck:
    """Tests for health check function."""

    def test_check_health_success(
        self,
        mock_health_response: dict[str, Any],
    ) -> None:
        """Test successful health check."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.return_value = mock_health_response

            from services.api_client import check_health

            result = check_health()

            assert result["status"] == "healthy"
            assert result["databases"]["vector_store"] is True

    def test_check_health_failure(self) -> None:
        """Test health check on API failure."""
        with patch("services.api_client._make_request") as mock_request:
            mock_request.side_effect = Exception("Connection refused")

            from services.api_client import check_health

            result = check_health()

            assert result["status"] == "error"
            assert "error" in result
