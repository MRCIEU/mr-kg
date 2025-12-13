"""Integration tests for study detail and similarity functionality.

Tests cover study extraction retrieval and similarity queries.
These tests require the API service to be running with databases available.
"""

import httpx
import pytest

from .conftest import make_api_request, requires_api


# ==== Extraction tests ====


@requires_api
class TestGetExtraction:
    """Tests for study extraction endpoint."""

    def test_get_extraction_for_valid_pmid(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test extraction retrieval for a valid PMID."""
        # First, get a valid PMID from the studies list
        list_response = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 1},
        )

        if list_response.status_code != 200:
            pytest.skip("Could not get study list")

        studies = list_response.json().get("studies", [])
        if not studies:
            pytest.skip("No studies available for testing")

        pmid = studies[0]["pmid"]

        # Get extraction
        response = make_api_request(
            api_client,
            f"/studies/{pmid}/extraction",
            params={"model": "gpt-5"},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["pmid"] == pmid
        assert "model" in data
        assert "title" in data
        assert "traits" in data
        assert "results" in data
        assert "metadata" in data
        assert isinstance(data["traits"], list)
        assert isinstance(data["results"], list)

    def test_get_extraction_for_invalid_pmid(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test extraction returns 404 for invalid PMID."""
        response = make_api_request(
            api_client,
            "/studies/00000000/extraction",
            params={"model": "gpt-5"},
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_get_extraction_with_different_models(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test extraction retrieval with different model parameters."""
        # Get a valid PMID first
        list_response = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 1},
        )

        if list_response.status_code != 200:
            pytest.skip("Could not get study list")

        studies = list_response.json().get("studies", [])
        if not studies:
            pytest.skip("No studies available")

        pmid = studies[0]["pmid"]

        # Test with default model
        response = make_api_request(
            api_client,
            f"/studies/{pmid}/extraction",
        )

        assert response.status_code == 200


# ==== Trait similarity tests ====


@requires_api
class TestTraitSimilarity:
    """Tests for trait similarity endpoint."""

    def test_get_trait_similarity_for_valid_pmid(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test trait similarity returns data for valid PMID."""
        # Get a valid PMID
        list_response = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 1},
        )

        if list_response.status_code != 200:
            pytest.skip("Could not get study list")

        studies = list_response.json().get("studies", [])
        if not studies:
            pytest.skip("No studies available")

        pmid = studies[0]["pmid"]

        # Get trait similarity
        response = make_api_request(
            api_client,
            f"/studies/{pmid}/similar/trait",
            params={"model": "gpt-5", "limit": 10},
        )

        # May return 404 if study not in trait profile db, which is valid
        if response.status_code == 404:
            pytest.skip(
                "Study not found in trait profile database - expected for some studies"
            )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["query_pmid"] == pmid
        assert "query_model" in data
        assert "query_title" in data
        assert "query_trait_count" in data
        assert "similar_studies" in data
        assert isinstance(data["similar_studies"], list)

        # Verify similar study structure if any exist
        for similar in data["similar_studies"]:
            assert "pmid" in similar
            assert "title" in similar
            assert "trait_profile_similarity" in similar
            assert "trait_jaccard_similarity" in similar
            assert "trait_count" in similar

    def test_trait_similarity_respects_limit(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that trait similarity respects limit parameter."""
        list_response = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 1},
        )

        if list_response.status_code != 200:
            pytest.skip("Could not get study list")

        studies = list_response.json().get("studies", [])
        if not studies:
            pytest.skip("No studies available")

        pmid = studies[0]["pmid"]

        response = make_api_request(
            api_client,
            f"/studies/{pmid}/similar/trait",
            params={"model": "gpt-5", "limit": 3},
        )

        if response.status_code == 404:
            pytest.skip("Study not in trait profile database")

        assert response.status_code == 200
        data = response.json()
        assert len(data["similar_studies"]) <= 3


# ==== Evidence similarity tests ====


@requires_api
class TestEvidenceSimilarity:
    """Tests for evidence similarity endpoint."""

    def test_get_evidence_similarity_for_valid_pmid(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test evidence similarity returns data for valid PMID."""
        # Get a valid PMID
        list_response = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 1},
        )

        if list_response.status_code != 200:
            pytest.skip("Could not get study list")

        studies = list_response.json().get("studies", [])
        if not studies:
            pytest.skip("No studies available")

        pmid = studies[0]["pmid"]

        # Get evidence similarity
        response = make_api_request(
            api_client,
            f"/studies/{pmid}/similar/evidence",
            params={"model": "gpt-5", "limit": 10},
        )

        # May return 404 if study not in evidence profile db
        if response.status_code == 404:
            pytest.skip(
                "Study not found in evidence profile database - expected for "
                "some studies"
            )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["query_pmid"] == pmid
        assert "query_model" in data
        assert "query_title" in data
        assert "query_result_count" in data
        assert "similar_studies" in data
        assert isinstance(data["similar_studies"], list)

        # Verify similar study structure if any exist
        for similar in data["similar_studies"]:
            assert "pmid" in similar
            assert "title" in similar
            assert "direction_concordance" in similar
            assert "matched_pairs" in similar
            assert "match_type_exact" in similar
            assert "match_type_fuzzy" in similar
            assert "match_type_efo" in similar


# ==== Complete study detail flow tests ====


@requires_api
class TestCompleteStudyDetailFlow:
    """Tests for complete study detail workflows."""

    def test_extraction_to_similarities_flow(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test flow from extraction to both similarity endpoints."""
        # Step 1: Get a study
        list_response = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 5},
        )

        if list_response.status_code != 200:
            pytest.skip("Could not get study list")

        studies = list_response.json().get("studies", [])
        if not studies:
            pytest.skip("No studies available")

        pmid = studies[0]["pmid"]

        # Step 2: Get extraction
        extraction_response = make_api_request(
            api_client,
            f"/studies/{pmid}/extraction",
            params={"model": "gpt-5"},
        )

        assert extraction_response.status_code == 200
        extraction = extraction_response.json()
        assert extraction["pmid"] == pmid

        # Step 3: Get trait similarity
        trait_sim_response = make_api_request(
            api_client,
            f"/studies/{pmid}/similar/trait",
            params={"model": "gpt-5"},
        )

        # Either 200 or 404 is acceptable
        assert trait_sim_response.status_code in (200, 404)

        # Step 4: Get evidence similarity
        evidence_sim_response = make_api_request(
            api_client,
            f"/studies/{pmid}/similar/evidence",
            params={"model": "gpt-5"},
        )

        # Either 200 or 404 is acceptable
        assert evidence_sim_response.status_code in (200, 404)

    def test_navigate_to_similar_study(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test navigating from similar study to its extraction."""
        # Get a study with trait similarities
        list_response = make_api_request(
            api_client,
            "/studies",
            params={"model": "gpt-5", "limit": 10},
        )

        if list_response.status_code != 200:
            pytest.skip("Could not get study list")

        studies = list_response.json().get("studies", [])
        similar_pmid = None

        # Find a study that has similar studies
        for study in studies:
            sim_response = make_api_request(
                api_client,
                f"/studies/{study['pmid']}/similar/trait",
                params={"model": "gpt-5", "limit": 1},
            )

            if sim_response.status_code == 200:
                sim_data = sim_response.json()
                if sim_data.get("similar_studies"):
                    similar_pmid = sim_data["similar_studies"][0]["pmid"]
                    break

        if not similar_pmid:
            pytest.skip("No similar studies found for testing navigation")

        # Navigate to the similar study
        extraction_response = make_api_request(
            api_client,
            f"/studies/{similar_pmid}/extraction",
            params={"model": "gpt-5"},
        )

        assert extraction_response.status_code == 200
        extraction = extraction_response.json()
        assert extraction["pmid"] == similar_pmid


# ==== Statistics endpoint tests ====


@requires_api
class TestStatistics:
    """Tests for statistics endpoint."""

    def test_get_statistics_returns_data(
        self,
        api_client: httpx.Client,
    ) -> None:
        """Test that statistics endpoint returns data."""
        response = make_api_request(
            api_client,
            "/statistics",
        )

        assert response.status_code == 200
        data = response.json()

        assert "overall" in data
        assert "model_similarity_stats" in data
        assert "model_evidence_stats" in data

        # Verify overall structure
        overall = data["overall"]
        assert "total_papers" in overall or overall is not None
