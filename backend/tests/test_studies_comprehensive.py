"""Comprehensive tests for the studies API endpoints."""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.database import ModelResult, SimilaritySearchResult
from app.services.database_service import AnalyticsService, StudyService

client = TestClient(app)


# ---- Test Fixtures ----


@pytest.fixture
def mock_study_service():
    """Mock StudyService for testing."""
    service = Mock(spec=StudyService)
    service.study_repo = Mock()
    service.trait_repo = Mock()
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
            1,
            "gpt-4-1",
            "12345678",
            {"test": "data"},
            {"traits": ["BMI"]},
            "Study 1",
            "Abstract 1",
            "Nature",
            "2023-01-15",
            "Uni 1",
        ),
        (
            2,
            "gpt-3.5-turbo",
            "12345679",
            {"test": "data2"},
            {"traits": ["T2D"]},
            "Study 2",
            "Abstract 2",
            "Science",
            "2023-02-15",
            "Uni 2",
        ),
    ]


# ---- Tests for Studies Listing Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_list_studies_success(
    mock_get_service, mock_study_service, sample_studies_list
):
    """Test successful studies listing."""
    mock_get_service.return_value = mock_study_service

    # Mock repository responses
    mock_study_service.study_repo.execute_query.side_effect = [
        [(2,)],  # Total count query
        sample_studies_list,  # Results query
    ]

    response = client.get("/api/v1/studies/")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 2
    assert data["total_count"] == 2
    assert data["page"] == 1
    assert data["page_size"] == 50

    first_study = data["data"][0]
    assert first_study["id"] == 1
    assert first_study["model"] == "gpt-4-1"
    assert first_study["pmid"] == "12345678"


@patch("app.core.dependencies.get_database_service")
def test_list_studies_with_filters(mock_get_service, mock_study_service):
    """Test studies listing with filters."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [
            (
                1,
                "gpt-4-1",
                "12345678",
                {},
                {},
                "Study",
                "Abstract",
                "Nature",
                "2023-01-15",
                "Uni",
            )
        ],
    ]

    response = client.get(
        "/api/v1/studies/?model=gpt-4-1&journal=Nature&date_from=2023-01-01&date_to=2023-12-31"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 1


@patch("app.core.dependencies.get_database_service")
def test_list_studies_pagination(mock_get_service, mock_study_service):
    """Test studies listing with pagination."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.execute_query.side_effect = [
        [(100,)],  # Total count query
        [
            (
                1,
                "gpt-4-1",
                "12345678",
                {},
                {},
                "Study",
                "Abstract",
                "Nature",
                "2023-01-15",
                "Uni",
            )
        ],
    ]

    response = client.get("/api/v1/studies/?page=2&page_size=20")

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 2
    assert data["page_size"] == 20
    assert data["total_pages"] == 5
    assert data["has_next"] is True
    assert data["has_previous"] is True


# ---- Tests for Study Search Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_search_studies_success(mock_get_service, mock_study_service):
    """Test successful study search."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [
            (
                1,
                "gpt-4-1",
                "12345678",
                {},
                {},
                "Diabetes Study",
                "Abstract",
                "Nature",
                "2023-01-15",
                "Uni",
            )
        ],
    ]

    response = client.get("/api/v1/studies/search?q=diabetes")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 1

    # Check that query was properly formatted
    call_args = mock_study_service.study_repo.execute_query.call_args_list
    assert "%diabetes%" in call_args[0][0][1]  # First call is count query


@patch("app.core.dependencies.get_database_service")
def test_search_studies_no_query(mock_get_service, mock_study_service):
    """Test study search without query parameter."""
    response = client.get("/api/v1/studies/search")
    assert response.status_code == 422  # Validation error


@patch("app.core.dependencies.get_database_service")
def test_search_studies_empty_results(mock_get_service, mock_study_service):
    """Test study search with no results."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.execute_query.side_effect = [
        [(0,)],  # Total count query
        [],  # Results query
    ]

    response = client.get("/api/v1/studies/search?q=nonexistent")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 0
    assert data["total_count"] == 0


# ---- Tests for Study Details Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_get_study_details_success(
    mock_get_service, mock_study_service, sample_study
):
    """Test successful study details retrieval."""
    mock_get_service.return_value = mock_study_service
    mock_study_service.study_repo.get_study_by_id.return_value = sample_study

    response = client.get("/api/v1/studies/1")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    study = data["data"]
    assert study["id"] == 1
    assert study["model"] == "gpt-4-1"
    assert study["pmid"] == "12345678"


@patch("app.core.dependencies.get_database_service")
def test_get_study_details_not_found(mock_get_service, mock_study_service):
    """Test study details for non-existent study."""
    mock_get_service.return_value = mock_study_service
    mock_study_service.study_repo.get_study_by_id.return_value = None

    response = client.get("/api/v1/studies/99999")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


# ---- Tests for Study by PMID Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_get_studies_by_pmid_success(mock_get_service, mock_study_service):
    """Test successful studies by PMID retrieval."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.execute_query.return_value = [
        (
            1,
            "gpt-4-1",
            "12345678",
            {},
            {},
            "Study",
            "Abstract",
            "Nature",
            "2023-01-15",
            "Uni",
        ),
        (
            2,
            "gpt-3.5-turbo",
            "12345678",
            {},
            {},
            "Study 2",
            "Abstract 2",
            "Science",
            "2023-01-15",
            "Uni 2",
        ),
    ]

    response = client.get("/api/v1/studies/pmid/12345678")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 2

    for study in data["data"]:
        assert study["pmid"] == "12345678"


@patch("app.core.dependencies.get_database_service")
def test_get_studies_by_pmid_not_found(mock_get_service, mock_study_service):
    """Test studies by PMID for non-existent PMID."""
    mock_get_service.return_value = mock_study_service
    mock_study_service.study_repo.execute_query.return_value = []

    response = client.get("/api/v1/studies/pmid/99999999")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


# ---- Tests for Similar Studies Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_get_similar_studies_success(
    mock_get_service, mock_study_service, sample_study
):
    """Test successful similar studies retrieval."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.get_study_by_id.return_value = sample_study
    mock_study_service.find_similar_studies.return_value = [
        SimilaritySearchResult(
            query_id="1",
            query_label="Sample Study",
            result_id="2",
            result_label="Similar Study",
            similarity=0.85,
        )
    ]

    response = client.get("/api/v1/studies/1/similar")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 1
    assert data["data"][0]["similarity"] == 0.85


@patch("app.core.dependencies.get_database_service")
def test_get_similar_studies_with_parameters(
    mock_get_service, mock_study_service, sample_study
):
    """Test similar studies with custom parameters."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.get_study_by_id.return_value = sample_study
    mock_study_service.find_similar_studies.return_value = []

    response = client.get(
        "/api/v1/studies/1/similar?max_results=5&similarity_threshold=0.8"
    )

    assert response.status_code == 200

    # Check that service was called with correct parameters
    mock_study_service.find_similar_studies.assert_called_once()
    call_args = mock_study_service.find_similar_studies.call_args
    assert call_args[0][0] == 1  # study_id
    assert call_args[0][1].max_results == 5
    assert call_args[0][1].min_similarity == 0.8


# ---- Tests for Metadata Endpoints ----


@patch("app.core.dependencies.get_database_service")
def test_get_available_models(mock_get_service, mock_study_service):
    """Test get available models endpoint."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.execute_query.return_value = [
        ("gpt-4-1", 150),
        ("gpt-3.5-turbo", 120),
        ("claude-3", 80),
    ]

    response = client.get("/api/v1/studies/models")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 3

    first_model = data["data"][0]
    assert first_model["model"] == "gpt-4-1"
    assert first_model["count"] == 150


@patch("app.core.dependencies.get_database_service")
def test_get_available_journals(mock_get_service, mock_study_service):
    """Test get available journals endpoint."""
    mock_get_service.return_value = mock_study_service

    mock_study_service.study_repo.execute_query.return_value = [
        ("Nature", 45),
        ("Science", 38),
        ("Cell", 25),
    ]

    response = client.get("/api/v1/studies/journals")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 3

    first_journal = data["data"][0]
    assert first_journal["journal"] == "Nature"
    assert first_journal["count"] == 45


# ---- Tests for Studies Overview Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_get_studies_overview(mock_get_service, mock_analytics_service):
    """Test get studies overview endpoint."""
    mock_get_service.return_value = mock_analytics_service

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
    ]

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


# ---- Tests for Error Handling ----


@patch("app.core.dependencies.get_database_service")
def test_list_studies_database_error(mock_get_service, mock_study_service):
    """Test studies listing with database error."""
    mock_get_service.return_value = mock_study_service
    mock_study_service.study_repo.execute_query.side_effect = Exception(
        "Database connection failed"
    )

    response = client.get("/api/v1/studies/")

    assert response.status_code == 500
    data = response.json()
    assert "Failed to list studies" in data["detail"]


@patch("app.core.dependencies.get_database_service")
def test_search_studies_database_error(mock_get_service, mock_study_service):
    """Test study search with database error."""
    mock_get_service.return_value = mock_study_service
    mock_study_service.study_repo.execute_query.side_effect = Exception(
        "Database connection failed"
    )

    response = client.get("/api/v1/studies/search?q=test")

    assert response.status_code == 500
    data = response.json()
    assert "Failed to search studies" in data["detail"]


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


def test_get_similar_studies_invalid_parameters():
    """Test similar studies with invalid parameters."""
    # Invalid similarity threshold (too high)
    response = client.get("/api/v1/studies/1/similar?similarity_threshold=1.5")
    assert response.status_code == 422

    # Invalid max results (too high)
    response = client.get("/api/v1/studies/1/similar?max_results=200")
    assert response.status_code == 422

    # Invalid study ID (negative)
    response = client.get("/api/v1/studies/-1/similar")
    assert response.status_code == 422
