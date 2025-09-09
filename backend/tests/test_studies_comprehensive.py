"""Comprehensive tests for the studies API endpoints."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_analytics_service, get_study_service
from app.main import app
from app.models.database import ModelResult
from app.services.database_service import AnalyticsService, StudyService

client = TestClient(app)


# ---- Test Helpers ----


def override_study_service_dependency(mock_service):
    """Helper to override study service dependency."""

    async def mock_get_study_service():
        yield mock_service

    app.dependency_overrides[get_study_service] = mock_get_study_service


def override_analytics_service_dependency(mock_service):
    """Helper to override analytics service dependency."""

    async def mock_get_analytics_service():
        yield mock_service

    app.dependency_overrides[get_analytics_service] = mock_get_analytics_service


def clear_dependency_overrides():
    """Helper to clear all dependency overrides."""
    app.dependency_overrides.clear()


# ---- Test Fixtures ----


@pytest.fixture
def mock_study_service():
    """Mock StudyService for testing."""
    service = Mock(spec=StudyService)
    service.study_repo = Mock()
    service.trait_repo = Mock()
    service.similarity_repo = Mock()  # Add similarity_repo mock
    return service


@pytest.fixture
def mock_analytics_service():
    """Mock AnalyticsService for testing."""
    service = Mock(spec=AnalyticsService)
    service.study_repo = Mock()
    return service


@pytest.fixture
def sample_study():
    """Sample study for testing."""
    return ModelResult(
        id=1,
        model="gpt-4-1",
        pmid="12345678",
        metadata={
            "title": "Sample Study on Body Mass Index",
            "journal": "Nature Medicine",
        },
        results={"traits": ["Body mass index", "Type 2 diabetes"]},
    )


@pytest.fixture
def sample_studies_list():
    """Sample studies list for testing."""
    return [
        (
            1,  # id
            "gpt-4-1",  # model
            "12345678",  # pmid
            "Study 1",  # title
            "Nature",  # journal
            "2023-01-15",  # pub_date
            2,  # trait_count
        ),
        (
            2,  # id
            "gpt-3.5-turbo",  # model
            "12345679",  # pmid
            "Study 2",  # title
            "Science",  # journal
            "2023-02-15",  # pub_date
            1,  # trait_count
        ),
    ]


# ---- Tests for Studies Listing Endpoint ----


def test_list_studies_success(mock_study_service, sample_studies_list):
    """Test successful studies listing."""

    # Mock repository responses
    mock_study_service.study_repo.execute_query.side_effect = [
        [(2,)],  # Total count query
        sample_studies_list,  # Results query
    ]

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 2
        assert data["pagination"]["total_items"] == 2
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 50

        first_study = data["data"][0]
        assert first_study["id"] == 1
        assert first_study["model"] == "gpt-4-1"
        assert first_study["pmid"] == "12345678"
    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_list_studies_with_filters(mock_study_service):
    """Test studies listing with filters."""

    # Mock repository responses
    mock_study_service.study_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [
            (
                1,  # id
                "gpt-4-1",  # model
                "12345678",  # pmid
                "Study",  # title
                "Nature",  # journal
                "2023-01-15",  # pub_date
                1,  # trait_count
            )
        ],  # Results query
    ]

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get(
            "/api/v1/studies/?model=gpt-4-1&journal=Nature&date_from=2023-01-01&date_to=2023-12-31"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1

        # Verify the study data structure
        study = data["data"][0]
        assert study["id"] == 1
        assert study["model"] == "gpt-4-1"
        assert study["pmid"] == "12345678"
    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_list_studies_pagination(mock_study_service):
    """Test studies listing with pagination."""

    # Mock repository responses
    mock_study_service.study_repo.execute_query.side_effect = [
        [(100,)],  # Total count query
        [
            (
                1,  # id
                "gpt-4-1",  # model
                "12345678",  # pmid
                "Study",  # title
                "Nature",  # journal
                "2023-01-15",  # pub_date
                1,  # trait_count
            )
        ],  # Results query
    ]

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/?page=2&page_size=20")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 20
        assert data["pagination"]["total_pages"] == 5
        assert data["pagination"]["has_next"] is True
        assert data["pagination"]["has_previous"] is True
    finally:
        # Clean up the override
        clear_dependency_overrides()


# ---- Tests for Study Search Endpoint ----


def test_search_studies_success(mock_study_service):
    """Test successful study search."""

    # Mock repository responses
    mock_study_service.study_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [
            (
                1,  # id
                "gpt-4-1",  # model
                "12345678",  # pmid
                "Diabetes Study",  # title
                "Nature",  # journal
                "2023-01-15",  # pub_date
                1,  # trait_count
            )
        ],  # Results query
    ]

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/search?q=diabetes")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 1

        # Check that query was properly formatted
        call_args = mock_study_service.study_repo.execute_query.call_args_list
        assert "%diabetes%" in call_args[0][0][1]  # First call is count query
    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_search_studies_no_query():
    """Test study search without query parameter."""
    response = client.get("/api/v1/studies/search")
    assert response.status_code == 422  # Validation error


def test_search_studies_empty_results(mock_study_service):
    """Test study search with no results."""

    # Mock repository responses
    mock_study_service.study_repo.execute_query.side_effect = [
        [(0,)],  # Total count query
        [],  # Results query
    ]

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/search?q=nonexistent")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 0
        assert data["pagination"]["total_items"] == 0
    finally:
        # Clean up the override
        clear_dependency_overrides()


# ---- Tests for Study Details Endpoint ----


def test_get_study_details_success(
    mock_study_service, mock_analytics_service, sample_study
):
    """Test successful study details retrieval."""
    # Import the models we need for the mock response
    from app.models.database import (
        ModelResultTrait,
        MRPubmedData,
        StudyDetailResponse,
        TraitSimilarity,
    )

    # Create mock PubMed data (with required parameters)
    mock_pubmed_data = MRPubmedData(
        pmid="12345678",
        title="Sample Study on Body Mass Index",
        abstract="A comprehensive study investigating the relationship between genetic variants and BMI.",
        journal="Nature Medicine",
        pub_date="2023-01-15",
        journal_issn="1546-170X",
        author_affil="University Research Center",
    )

    # Create mock traits (with required parameters)
    mock_traits = [
        ModelResultTrait(
            id=1,
            model_result_id=1,
            trait_index=0,
            trait_label="Body mass index",
            trait_id_in_result="trait_0",
        ),
        ModelResultTrait(
            id=2,
            model_result_id=1,
            trait_index=1,
            trait_label="Type 2 diabetes",
            trait_id_in_result="trait_1",
        ),
    ]

    # Create mock similar studies (with required parameters)
    mock_similar_studies = [
        TraitSimilarity(
            id=1,
            query_combination_id=1,
            similar_pmid="87654321",
            similar_model="gpt-4-1",
            similar_title="Related BMI Study",
            trait_profile_similarity=0.85,
            trait_jaccard_similarity=0.75,
            query_trait_count=2,
            similar_trait_count=3,
        )
    ]

    # Create the study detail response
    study_detail_response = StudyDetailResponse(
        study=sample_study,
        pubmed_data=mock_pubmed_data,
        traits=mock_traits,
        similar_studies=mock_similar_studies,
    )

    # Mock the get_study_details method to return the response
    mock_study_service.get_study_details.return_value = study_detail_response

    # Mock the execute_query method for statistics query (called in the endpoint)
    mock_study_service.study_repo.execute_query.return_value = [(2, 2)]

    # Override both dependencies
    override_study_service_dependency(mock_study_service)
    override_analytics_service_dependency(mock_analytics_service)

    try:
        response = client.get("/api/v1/studies/1")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        study_data = data["data"]

        # Check the main study data
        assert study_data["study"]["id"] == 1
        assert study_data["study"]["model"] == "gpt-4-1"
        assert study_data["study"]["pmid"] == "12345678"
        assert (
            study_data["study"]["metadata"]["title"]
            == "Sample Study on Body Mass Index"
        )

        # Check PubMed data
        assert (
            study_data["pubmed_data"]["title"]
            == "Sample Study on Body Mass Index"
        )
        assert study_data["pubmed_data"]["journal"] == "Nature Medicine"

        # Check traits
        assert len(study_data["traits"]) == 2
        assert study_data["traits"][0]["trait_label"] == "Body mass index"

        # Check similar studies
        assert len(study_data["similar_studies"]) == 1
        assert study_data["similar_studies"][0]["similarity"] == 0.85

        # Check statistics
        assert "statistics" in study_data
        assert study_data["statistics"]["trait_count"] == 2
        assert study_data["statistics"]["unique_trait_count"] == 2

        # Verify that the service method was called
        mock_study_service.get_study_details.assert_called_once_with(1)

    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_get_study_details_not_found(
    mock_study_service, mock_analytics_service
):
    """Test study details for non-existent study."""
    # Mock the get_study_details method to return None (study not found)
    mock_study_service.get_study_details.return_value = None

    # Override both dependencies
    override_study_service_dependency(mock_study_service)
    override_analytics_service_dependency(mock_analytics_service)

    try:
        response = client.get("/api/v1/studies/99999")

        assert response.status_code == 404
        data = response.json()

        # Check the API's standard error response format
        assert data["success"] is False
        assert "not found" in data["error"]["message"].lower()

        # Verify that the service method was called
        mock_study_service.get_study_details.assert_called_once_with(99999)
    finally:
        # Clean up the override
        clear_dependency_overrides()


# ---- Tests for Study by PMID Endpoint ----


def test_get_studies_by_pmid_success(mock_study_service):
    """Test successful studies by PMID retrieval."""
    from app.models.database import ModelResult

    # Create mock study data for PMID "12345678"
    mock_studies = [
        ModelResult(
            id=1,
            model="gpt-4-1",
            pmid="12345678",
            metadata={"title": "Study 1", "journal": "Nature"},
            results={"traits": ["trait1"]},
        ),
        ModelResult(
            id=2,
            model="gpt-3.5-turbo",
            pmid="12345678",
            metadata={"title": "Study 2", "journal": "Science"},
            results={"traits": ["trait2"]},
        ),
    ]

    # Mock the get_studies_by_pmid method to return the mock studies
    mock_study_service.get_studies_by_pmid.return_value = mock_studies

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/pmid/12345678")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 2

        # Verify study data structure
        for study in data["data"]:
            assert study["pmid"] == "12345678"
            assert study["id"] in [1, 2]
            assert study["model"] in ["gpt-4-1", "gpt-3.5-turbo"]

        # Verify that the service method was called with correct PMID
        mock_study_service.get_studies_by_pmid.assert_called_once_with(
            "12345678"
        )
    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_get_studies_by_pmid_not_found(mock_study_service):
    """Test studies by PMID for non-existent PMID."""
    # Mock the get_studies_by_pmid method to return empty list (no studies found)
    mock_study_service.get_studies_by_pmid.return_value = []

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/pmid/99999999")

        assert response.status_code == 404
        data = response.json()

        # Check the API's standard error response format
        assert data["success"] is False
        assert "no studies found" in data["error"]["message"].lower()

        # Verify that the service method was called with the correct PMID
        mock_study_service.get_studies_by_pmid.assert_called_once_with(
            "99999999"
        )
    finally:
        # Clean up the override
        clear_dependency_overrides()


# ---- Tests for Similar Studies Endpoint ----


def test_get_similar_studies_success(mock_study_service, sample_study):
    """Test successful similar studies retrieval."""
    from app.models.database import QueryCombination, TraitSimilarity

    # Mock study retrieval
    mock_study_service.study_repo.get_study_by_id.return_value = sample_study

    # Mock combination finding
    mock_combination = QueryCombination(
        id=1,
        pmid="12345678",
        model="gpt-4-1",
        title="Sample Study",
        trait_count=2,
    )
    mock_study_service.similarity_repo.find_combination.return_value = (
        mock_combination
    )

    # Mock similarities retrieval
    mock_similarities = [
        TraitSimilarity(
            id=1,
            query_combination_id=1,
            similar_pmid="87654321",
            similar_model="gpt-4-1",
            similar_title="Similar Study",
            trait_profile_similarity=0.85,
            trait_jaccard_similarity=0.75,
            query_trait_count=2,
            similar_trait_count=3,
        )
    ]
    mock_study_service.similarity_repo.get_similarities.return_value = (
        mock_similarities
    )

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/1/similar")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 1
        assert data["data"][0]["similarity"] == 0.85
    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_get_similar_studies_with_parameters(mock_study_service, sample_study):
    """Test similar studies with custom parameters."""
    from app.models.database import QueryCombination

    # Mock study retrieval
    mock_study_service.study_repo.get_study_by_id.return_value = sample_study

    # Mock combination finding
    mock_combination = QueryCombination(
        id=1,
        pmid="12345678",
        model="gpt-4-1",
        title="Sample Study",
        trait_count=2,
    )
    mock_study_service.similarity_repo.find_combination.return_value = (
        mock_combination
    )

    # Mock empty similarities (testing custom parameters)
    mock_study_service.similarity_repo.get_similarities.return_value = []

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get(
            "/api/v1/studies/1/similar?max_results=5&similarity_threshold=0.8"
        )

        assert response.status_code == 200

        # Check that similarity_repo methods were called with correct parameters
        mock_study_service.similarity_repo.find_combination.assert_called_once_with(
            "12345678", "gpt-4-1"
        )
        mock_study_service.similarity_repo.get_similarities.assert_called_once_with(
            1, top_k=5, min_similarity=0.8, similarity_type="trait_profile"
        )
    finally:
        # Clean up the override
        clear_dependency_overrides()


# ---- Tests for Metadata Endpoints ----


def test_get_available_models(mock_study_service):
    """Test get available models endpoint."""

    # Mock the query to return just model names (not counts)
    mock_study_service.study_repo.execute_query.return_value = [
        ("gpt-4-1",),
        ("gpt-3.5-turbo",),
        ("claude-3",),
    ]

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/models")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 3

        # The endpoint returns a simple list of model names
        assert data["data"][0] == "gpt-4-1"
        assert data["data"][1] == "gpt-3.5-turbo"
        assert data["data"][2] == "claude-3"
    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_get_available_journals(mock_study_service):
    """Test get available journals endpoint."""

    # Mock the query to return just journal names (not counts)
    mock_study_service.study_repo.execute_query.return_value = [
        ("Nature",),
        ("Science",),
        ("Cell",),
    ]

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/journals")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 3

        # The endpoint returns a simple list of journal names
        assert data["data"][0] == "Nature"
        assert data["data"][1] == "Science"
        assert data["data"][2] == "Cell"
    finally:
        # Clean up the override
        clear_dependency_overrides()


# ---- Tests for Studies Overview Endpoint ----


def test_get_studies_overview(mock_analytics_service):
    """Test get studies overview endpoint."""
    override_analytics_service_dependency(mock_analytics_service)

    mock_analytics_service.study_repo.get_count.return_value = 500
    mock_analytics_service.study_repo.execute_query.side_effect = [
        [(450,)],  # Unique PMIDs
        [
            ("gpt-4-1", 200),
            ("gpt-3.5-turbo", 180),
            ("claude-3", 120),
        ],  # Model distribution
        [("Nature", 50), ("Science", 45), ("Cell", 30)],  # Journal distribution
        [("2023", 300), ("2022", 150), ("2021", 50)],  # Year distribution
        [
            ("1-5", 200),
            ("6-10", 150),
            ("11-20", 100),
        ],  # Trait count distribution
    ]

    try:
        response = client.get("/api/v1/studies/stats/overview")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        overview = data["data"]

        assert overview["total_studies"] == 500
        assert overview["total_pmids"] == 450
        assert len(overview["model_distribution"]) == 3
        assert len(overview["journal_distribution"]) == 3
        assert len(overview["year_distribution"]) == 3
    finally:
        clear_dependency_overrides()


# ---- Tests for Error Handling ----


def test_list_studies_database_error(mock_study_service):
    """Test studies listing with database error."""
    mock_study_service.study_repo.execute_query.side_effect = Exception(
        "Database connection failed"
    )

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/")

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "Failed to list studies" in data["error"]["message"]
    finally:
        # Clean up the override
        clear_dependency_overrides()


def test_search_studies_database_error(mock_study_service):
    """Test study search with database error."""
    mock_study_service.study_repo.execute_query.side_effect = Exception(
        "Database connection failed"
    )

    # Override the dependency
    override_study_service_dependency(mock_study_service)

    try:
        response = client.get("/api/v1/studies/search?q=test")

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "Failed to search studies" in data["error"]["message"]
    finally:
        # Clean up the override
        clear_dependency_overrides()


# ---- Tests for Parameter Validation ----


def test_list_studies_invalid_pagination():
    """Test studies listing with invalid pagination parameters."""
    # Invalid page number (0)
    response = client.get("/api/v1/studies/?page=0")
    assert response.status_code == 422

    # Invalid page size (too large)
    response = client.get("/api/v1/studies/?page_size=2000")
    assert response.status_code == 422

    # Invalid page size (negative)
    response = client.get("/api/v1/studies/?page_size=-1")
    assert response.status_code == 422


def test_search_studies_invalid_pagination():
    """Test study search with invalid pagination parameters."""
    # Invalid page size (too large)
    response = client.get("/api/v1/studies/search?q=test&page_size=600")
    assert response.status_code == 422


def test_get_similar_studies_invalid_parameters(mock_study_service):
    """Test similar studies with invalid parameters."""
    # Setup basic mock to prevent dependency injection errors
    # Mock the study_repo to return None for invalid study IDs
    mock_study_service.study_repo.get_study_by_id.return_value = None
    override_study_service_dependency(mock_study_service)

    try:
        # Invalid similarity threshold (too high)
        response = client.get(
            "/api/v1/studies/1/similar?similarity_threshold=1.5"
        )
        assert response.status_code == 422

        # Invalid max results (too high)
        response = client.get("/api/v1/studies/1/similar?max_results=200")
        assert response.status_code == 422

        # Invalid study ID (negative) - this should return 404 since the study doesn't exist
        response = client.get("/api/v1/studies/-1/similar")
        assert (
            response.status_code == 404
        )  # Changed from 422 to 404 since the endpoint checks if study exists
    finally:
        clear_dependency_overrides()
