"""Integration tests for search functionality.

Tests cover the complete search flows including autocomplete suggestions
and study search operations. These tests require the API service to be
running with databases available.
"""

import httpx
import pytest

from .conftest import make_api_request, requires_api


# ==== Trait autocomplete tests ====


@requires_api
class TestTraitAutocomplete:
    """Tests for trait autocomplete endpoint."""

    def test_autocomplete_returns_suggestions(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that trait autocomplete returns suggestions."""
        response = make_api_request(
            api_client,
            "/traits/autocomplete",
            params={"q": "body", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()

        assert "traits" in data
        assert isinstance(data["traits"], list)
        # May be empty if no matching traits, but structure should be correct

    def test_autocomplete_respects_limit(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that autocomplete respects the limit parameter."""
        response = make_api_request(
            api_client,
            "/traits/autocomplete",
            params={"q": "blood", "limit": 5},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["traits"]) <= 5

    def test_autocomplete_requires_min_length(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that autocomplete requires minimum query length."""
        response = make_api_request(
            api_client,
            "/traits/autocomplete",
            params={"q": "b"},
        )

        # Should return validation error for query too short
        assert response.status_code == 422


# ==== Study autocomplete tests ====


@requires_api
class TestStudyAutocomplete:
    """Tests for study autocomplete endpoint."""

    def test_autocomplete_returns_suggestions(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that study autocomplete returns suggestions."""
        response = make_api_request(
            api_client,
            "/studies/autocomplete",
            params={"q": "mendelian", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()

        assert "studies" in data
        assert isinstance(data["studies"], list)

    def test_autocomplete_returns_pmid_and_title(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that autocomplete results include pmid and title."""
        response = make_api_request(
            api_client,
            "/studies/autocomplete",
            params={"q": "randomization", "limit": 5},
        )

        assert response.status_code == 200
        data = response.json()

        for study in data["studies"]:
            assert "pmid" in study
            assert "title" in study


# ==== Study search tests ====


@requires_api
class TestStudySearch:
    """Tests for study search endpoint."""

    def test_search_with_trait_filter_returns_results(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that search with trait filter returns results."""
        # First get a valid trait from autocomplete
        autocomplete_response = make_api_request(
            api_client,
            "/traits/autocomplete",
            params={"q": "body", "limit": 1},
        )

        if autocomplete_response.status_code != 200:
            pytest.skip("Could not get trait for testing")

        traits = autocomplete_response.json().get("traits", [])
        if not traits:
            pytest.skip("No traits found for testing")

        trait = traits[0]

        # Search with the trait
        response = make_api_request(
            api_client,
            "/studies",
            params={"trait": trait, "model": "gpt-5", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "studies" in data
        assert "limit" in data
        assert "offset" in data

    def test_search_with_text_query_returns_results(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that search with text query returns results."""
        response = make_api_request(
            api_client,
            "/studies",
            params={"q": "diabetes", "model": "gpt-5", "limit": 10},
        )

        assert response.status_code == 200
        data = response.json()

        assert "total" in data
        assert "studies" in data
        assert isinstance(data["studies"], list)

    def test_search_pagination_works(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that search pagination works correctly."""
        # Get first page
        response1 = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 5, "offset": 0},
        )

        assert response1.status_code == 200
        data1 = response1.json()

        # Get second page
        response2 = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 5, "offset": 5},
        )

        assert response2.status_code == 200
        data2 = response2.json()

        # Results should differ if there are enough studies
        if data1["total"] > 5:
            first_pmids = {s["pmid"] for s in data1["studies"]}
            second_pmids = {s["pmid"] for s in data2["studies"]}
            assert first_pmids != second_pmids

    def test_search_validates_limit(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that search validates limit parameter."""
        response = make_api_request(
            api_client,
            "/studies",
            params={"limit": 200},  # Exceeds max of 100
        )

        assert response.status_code == 422


# ==== Complete search flow tests ====


@requires_api
class TestCompleteSearchFlow:
    """Tests for complete search workflows."""

    def test_trait_search_to_study_flow(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test complete flow from trait search to study retrieval."""
        # Step 1: Get trait suggestions
        autocomplete_response = make_api_request(
            api_client,
            "/traits/autocomplete",
            params={"q": "body", "limit": 5},
        )

        if autocomplete_response.status_code != 200:
            pytest.skip("Autocomplete endpoint unavailable")

        traits = autocomplete_response.json().get("traits", [])
        if not traits:
            pytest.skip("No traits found for testing flow")

        # Step 2: Search studies by trait
        search_response = make_api_request(
            api_client,
            "/studies",
            params={"trait": traits[0], "model": "gpt-5", "limit": 5},
        )

        assert search_response.status_code == 200
        studies = search_response.json().get("studies", [])

        if not studies:
            pytest.skip("No studies found for the trait")

        # Step 3: Get extraction for first study
        pmid = studies[0]["pmid"]
        extraction_response = make_api_request(
            api_client,
            f"/studies/{pmid}/extraction",
            params={"model": "gpt-5"},
        )

        assert extraction_response.status_code == 200
        extraction = extraction_response.json()

        assert extraction["pmid"] == pmid
        assert "traits" in extraction
        assert "results" in extraction

    def test_study_title_search_flow(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test complete flow from study title search to details."""
        # Step 1: Search for studies by title
        autocomplete_response = make_api_request(
            api_client,
            "/studies/autocomplete",
            params={"q": "mendelian", "limit": 5},
        )

        if autocomplete_response.status_code != 200:
            pytest.skip("Study autocomplete endpoint unavailable")

        studies = autocomplete_response.json().get("studies", [])
        if not studies:
            pytest.skip("No studies found for testing flow")

        # Step 2: Get extraction for selected study
        pmid = studies[0]["pmid"]
        extraction_response = make_api_request(
            api_client,
            f"/studies/{pmid}/extraction",
            params={"model": "gpt-5"},
        )

        assert extraction_response.status_code == 200
        extraction = extraction_response.json()

        assert extraction["pmid"] == pmid
        assert "title" in extraction
