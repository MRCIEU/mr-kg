"""Comprehensive tests for the similarities API endpoints."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.database import (
    QueryCombination,
    SimilaritySearchResult,
)
from app.services.database_service import AnalyticsService, SimilarityService

client = TestClient(app)


# ---- Test Fixtures ----


@pytest.fixture
def mock_similarity_service():
    """Mock SimilarityService for testing."""
    service = Mock(spec=SimilarityService)
    service.similarity_repo = Mock()
    service.trait_repo = Mock()
    service.efo_repo = Mock()
    return service


@pytest.fixture
def mock_analytics_service():
    """Mock AnalyticsService for testing."""
    service = Mock(spec=AnalyticsService)
    service.similarity_repo = Mock()
    return service


@pytest.fixture
def sample_combination():
    """Sample query combination for testing."""
    return QueryCombination(
        id=1,
        pmid="12345678",
        model="gpt-4-1",
        title="Sample Study on BMI",
        trait_count=5,
    )


@pytest.fixture
def sample_similarity_results():
    """Sample similarity search results for testing."""
    return [
        SimilaritySearchResult(
            query_id="1",
            query_label="Body mass index",
            result_id="2",
            result_label="Obesity",
            similarity=0.85,
        ),
        SimilaritySearchResult(
            query_id="1",
            query_label="Body mass index",
            result_id="3",
            result_label="Weight",
            similarity=0.78,
        ),
    ]


@pytest.fixture
def sample_combinations_list():
    """Sample combinations list for testing."""
    return [
        (1, "12345678", "gpt-4-1", "Study 1", 5),
        (2, "12345679", "gpt-3.5-turbo", "Study 2", 3),
        (3, "12345680", "claude-3", "Study 3", 7),
    ]


# ---- Tests for Analyze Similarity Endpoint ----


@patch("app.api.v1.similarities.get_database_service")
def test_analyze_similarity_success(
    mock_get_service, mock_similarity_service, sample_combination
):
    """Test successful similarity analysis."""
    # Create a proper mock service with similarity_repo attribute
    mock_service = AsyncMock()
    mock_service.similarity_repo = Mock()
    mock_service.similarity_repo.find_combination.return_value = (
        sample_combination
    )

    # Mock get_similarities
    mock_similarities = [
        type(
            "Similarity",
            (),
            {
                "trait_profile_similarity": 0.92,
                "trait_jaccard_similarity": 0.88,
            },
        )(),
        type(
            "Similarity",
            (),
            {
                "trait_profile_similarity": 0.87,
                "trait_jaccard_similarity": 0.85,
            },
        )(),
    ]
    mock_service.similarity_repo.get_similarities.return_value = (
        mock_similarities
    )
    mock_get_service.return_value = mock_service

    response = client.get(
        "/api/v1/similarities/analyze?pmid=12345678&model=gpt-4-1"
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    analysis = data["data"]
    assert analysis["combination"]["pmid"] == "12345678"
    assert len(analysis["similarities"]) == 2
    assert "summary" in analysis


@patch("app.core.dependencies.get_database_service")
def test_analyze_similarity_not_found(
    mock_get_service, mock_similarity_service
):
    """Test similarity analysis for non-existent combination."""
    mock_get_service.return_value = mock_similarity_service
    mock_similarity_service.similarity_repo.find_combination.return_value = None

    response = client.get(
        "/api/v1/similarities/analyze?pmid=99999999&model=nonexistent"
    )

    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_analyze_similarity_missing_params():
    """Test similarity analysis with missing parameters."""
    # Missing model parameter
    response = client.get("/api/v1/similarities/analyze?pmid=12345678")
    assert response.status_code == 422

    # Missing pmid parameter
    response = client.get("/api/v1/similarities/analyze?model=gpt-4-1")
    assert response.status_code == 422


@patch("app.core.dependencies.get_database_service")
def test_analyze_similarity_with_filters(
    mock_get_service, mock_similarity_service
):
    """Test similarity analysis with filters."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.similarity_repo.execute_query.return_value = [
        (1, "12345678", "gpt-4-1", "Study Title", 5)
    ]
    mock_similarity_service.analyze_similarity.return_value = {
        "combination": sample_combination,
        "similar_studies": [],
        "statistics": {"total_similarities": 0},
    }

    response = client.get(
        "/api/v1/similarities/analyze?pmid=12345678&model=gpt-4-1"
        "&min_similarity=0.8&max_results=10&similarity_type=jaccard"
    )

    assert response.status_code == 200


# ---- Tests for Search Combinations Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_search_combinations_success(
    mock_get_service, mock_similarity_service, sample_combinations_list
):
    """Test successful combinations search."""
    mock_get_service.return_value = mock_similarity_service

    # Mock repository responses
    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(3,)],  # Total count query
        sample_combinations_list,  # Results query
    ]

    response = client.get("/api/v1/similarities/combinations")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 3
    assert data["total_count"] == 3

    first_combination = data["data"][0]
    assert first_combination["id"] == 1
    assert first_combination["pmid"] == "12345678"
    assert first_combination["model"] == "gpt-4-1"


@patch("app.core.dependencies.get_database_service")
def test_search_combinations_with_filters(
    mock_get_service, mock_similarity_service
):
    """Test combinations search with filters."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [(1, "12345678", "gpt-4-1", "Study 1", 5)],  # Results query
    ]

    response = client.get(
        "/api/v1/similarities/combinations?model=gpt-4-1&min_trait_count=5&max_trait_count=10"
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 1


@patch("app.core.dependencies.get_database_service")
def test_search_combinations_pagination(
    mock_get_service, mock_similarity_service
):
    """Test combinations search with pagination."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(50,)],  # Total count query
        [(1, "12345678", "gpt-4-1", "Study 1", 5)],  # Results query
    ]

    response = client.get(
        "/api/v1/similarities/combinations?page=2&page_size=20"
    )

    assert response.status_code == 200
    data = response.json()

    assert data["page"] == 2
    assert data["page_size"] == 20
    assert data["total_pages"] == 3


# ---- Tests for Combination Details Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_get_combination_details_success(
    mock_get_service, mock_similarity_service, sample_combination
):
    """Test successful combination details retrieval."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.similarity_repo.get_combination_by_id.return_value = sample_combination
    mock_similarity_service.similarity_repo.execute_query.return_value = [
        (5, 0.85),  # Similar combinations count and avg similarity
    ]

    response = client.get("/api/v1/similarities/combinations/1")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    combination = data["data"]
    assert combination["combination"]["id"] == 1
    assert "statistics" in combination


@patch("app.core.dependencies.get_database_service")
def test_get_combination_details_not_found(
    mock_get_service, mock_similarity_service
):
    """Test combination details for non-existent combination."""
    mock_get_service.return_value = mock_similarity_service
    mock_similarity_service.similarity_repo.get_combination_by_id.return_value = None

    response = client.get("/api/v1/similarities/combinations/99999")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


# ---- Tests for Combination Similarities Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_get_combination_similarities_success(
    mock_get_service, mock_similarity_service, sample_combination
):
    """Test successful combination similarities retrieval."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.similarity_repo.get_combination_by_id.return_value = sample_combination
    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(2,)],  # Total count query
        [
            (1, 1, "87654321", "gpt-4-1", "Similar Study 1", 0.92, 0.85),
            (2, 1, "11111111", "gpt-3.5-turbo", "Similar Study 2", 0.87, 0.78),
        ],  # Results query
    ]

    response = client.get("/api/v1/similarities/combinations/1/similarities")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 2
    assert data["total_count"] == 2

    first_similarity = data["data"][0]
    assert first_similarity["similar_pmid"] == "87654321"
    assert first_similarity["trait_profile_similarity"] == 0.92


@patch("app.core.dependencies.get_database_service")
def test_get_combination_similarities_with_filters(
    mock_get_service, mock_similarity_service, sample_combination
):
    """Test combination similarities with filters."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.similarity_repo.get_combination_by_id.return_value = sample_combination
    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [
            (1, 1, "87654321", "gpt-4-1", "Similar Study 1", 0.92, 0.85)
        ],  # Results query
    ]

    response = client.get(
        "/api/v1/similarities/combinations/1/similarities"
        "?min_similarity=0.8&similarity_type=jaccard&page_size=20"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


# ---- Tests for Vector Similarity Search Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_vector_similarity_search_success(
    mock_get_service, mock_similarity_service, sample_similarity_results
):
    """Test successful vector similarity search."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.perform_vector_search.return_value = (
        sample_similarity_results
    )

    request_data = {
        "query_vector": [0.1] * 200,  # 200-dimensional vector
        "top_k": 10,
        "threshold": 0.3,
    }

    response = client.post("/api/v1/similarities/vector", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 2

    first_result = data["data"][0]
    assert first_result["similarity"] == 0.85
    assert first_result["result_label"] == "Obesity"


def test_vector_similarity_search_invalid_vector():
    """Test vector similarity search with invalid vector dimension."""
    request_data = {
        "query_vector": [0.1, 0.2],  # Wrong dimension (should be 200)
        "top_k": 10,
        "threshold": 0.3,
    }

    response = client.post("/api/v1/similarities/vector", json=request_data)
    assert response.status_code == 400


@patch("app.core.dependencies.get_database_service")
def test_vector_similarity_search_different_types(
    mock_get_service, mock_similarity_service
):
    """Test vector similarity search with different search types."""
    mock_get_service.return_value = mock_similarity_service
    mock_similarity_service.perform_vector_search.return_value = []

    request_data = {
        "query_vector": [0.1] * 200,
        "top_k": 10,
        "threshold": 0.3,
    }

    # Test traits search
    response = client.post(
        "/api/v1/similarities/vector?search_type=traits", json=request_data
    )
    assert response.status_code == 200

    # Test efo search
    response = client.post(
        "/api/v1/similarities/vector?search_type=efo", json=request_data
    )
    assert response.status_code == 200


def test_vector_similarity_search_invalid_type():
    """Test vector similarity search with invalid search type."""
    request_data = {
        "query_vector": [0.1] * 200,
        "top_k": 10,
        "threshold": 0.3,
    }

    response = client.post(
        "/api/v1/similarities/vector?search_type=invalid", json=request_data
    )
    assert response.status_code == 400


# ---- Tests for Bulk Trait to EFO Mapping Endpoint ----


@patch("app.core.dependencies.get_database_service")
def test_bulk_trait_to_efo_mapping_success(
    mock_get_service, mock_similarity_service
):
    """Test successful bulk trait-to-EFO mapping."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.bulk_trait_to_efo_mapping.return_value = {
        "1": [
            SimilaritySearchResult(
                query_id="1",
                query_label="Body mass index",
                result_id="EFO_0004340",
                result_label="body mass index",
                similarity=0.95,
            )
        ],
        "2": [
            SimilaritySearchResult(
                query_id="2",
                query_label="Type 2 diabetes",
                result_id="EFO_0001360",
                result_label="type 2 diabetes mellitus",
                similarity=0.93,
            )
        ],
    }

    request_data = {
        "trait_indices": [1, 2],
        "top_k": 5,
        "threshold": 0.3,
    }

    response = client.post(
        "/api/v1/similarities/trait-to-efo", json=request_data
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "1" in data["data"]
    assert "2" in data["data"]
    assert len(data["data"]["1"]) == 1
    assert data["data"]["1"][0]["similarity"] == 0.95


def test_bulk_trait_to_efo_mapping_too_many():
    """Test bulk trait-to-EFO mapping with too many traits."""
    request_data = {
        "trait_indices": list(range(101)),  # Too many traits (limit is 100)
        "top_k": 5,
        "threshold": 0.3,
    }

    response = client.post(
        "/api/v1/similarities/trait-to-efo", json=request_data
    )
    assert response.status_code == 400


def test_bulk_trait_to_efo_mapping_empty():
    """Test bulk trait-to-EFO mapping with empty list."""
    request_data = {
        "trait_indices": [],
        "top_k": 5,
        "threshold": 0.3,
    }

    response = client.post(
        "/api/v1/similarities/trait-to-efo", json=request_data
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"] == []


# ---- Tests for Metadata Endpoints ----


@patch("app.core.dependencies.get_database_service")
def test_get_available_similarity_models(
    mock_get_service, mock_similarity_service
):
    """Test get available similarity models endpoint."""
    mock_get_service.return_value = mock_similarity_service

    mock_similarity_service.similarity_repo.execute_query.return_value = [
        ("gpt-4-1", 150),
        ("gpt-3.5-turbo", 120),
        ("claude-3", 80),
    ]

    response = client.get("/api/v1/similarities/models")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert len(data["data"]) == 3

    first_model = data["data"][0]
    assert first_model["model"] == "gpt-4-1"
    assert first_model["count"] == 150


@patch("app.core.dependencies.get_database_service")
def test_get_similarity_statistics(mock_get_service, mock_analytics_service):
    """Test get similarity statistics endpoint."""
    mock_get_service.return_value = mock_analytics_service

    mock_analytics_service.similarity_repo.execute_query.side_effect = [
        [(500,)],  # Total combinations
        [(2500,)],  # Total similarities
        [(0.72,)],  # Average similarity
        [("jaccard", 1500), ("cosine", 1000)],  # Similarity type distribution
        [("gpt-4-1", 200), ("gpt-3.5-turbo", 180)],  # Model distribution
    ]

    response = client.get("/api/v1/similarities/stats")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    stats = data["data"]

    assert stats["total_combinations"] == 500
    assert stats["total_similarities"] == 2500
    assert stats["average_similarity"] == 0.72
    assert len(stats["similarity_type_distribution"]) == 2
    assert len(stats["model_distribution"]) == 2


# ---- Tests for Error Handling ----


@patch("app.core.dependencies.get_database_service")
def test_analyze_similarity_database_error(
    mock_get_service, mock_similarity_service
):
    """Test similarity analysis with database error."""
    mock_get_service.return_value = mock_similarity_service
    mock_similarity_service.similarity_repo.execute_query.side_effect = (
        Exception("Database connection failed")
    )

    response = client.get(
        "/api/v1/similarities/analyze?pmid=12345678&model=gpt-4-1"
    )

    assert response.status_code == 500
    data = response.json()
    assert "Failed to analyze similarity" in data["detail"]


@patch("app.core.dependencies.get_database_service")
def test_search_combinations_database_error(
    mock_get_service, mock_similarity_service
):
    """Test combinations search with database error."""
    mock_get_service.return_value = mock_similarity_service
    mock_similarity_service.similarity_repo.execute_query.side_effect = (
        Exception("Database connection failed")
    )

    response = client.get("/api/v1/similarities/combinations")

    assert response.status_code == 500
    data = response.json()
    assert "Failed to search combinations" in data["detail"]


# ---- Tests for Parameter Validation ----


def test_analyze_similarity_invalid_params():
    """Test similarity analysis with invalid parameters."""
    # Invalid similarity threshold (too high)
    response = client.get(
        "/api/v1/similarities/analyze?pmid=12345&model=test&min_similarity=2.0"
    )
    assert response.status_code == 422

    # Invalid max_results (too high)
    response = client.get(
        "/api/v1/similarities/analyze?pmid=12345&model=test&max_results=1000"
    )
    assert response.status_code == 422


def test_search_combinations_invalid_pagination():
    """Test combinations search with invalid pagination."""
    # Invalid page number (0)
    response = client.get("/api/v1/similarities/combinations?page=0")
    assert response.status_code == 422

    # Invalid page size (too large)
    response = client.get("/api/v1/similarities/combinations?page_size=2000")
    assert response.status_code == 422


def test_vector_search_invalid_params():
    """Test vector search with invalid parameters."""
    # Invalid top_k (too high)
    request_data = {
        "query_vector": [0.1] * 200,
        "top_k": 1000,  # Too high
        "threshold": 0.3,
    }

    response = client.post("/api/v1/similarities/vector", json=request_data)
    assert response.status_code == 422

    # Invalid threshold (negative)
    request_data = {
        "query_vector": [0.1] * 200,
        "top_k": 10,
        "threshold": -0.1,  # Negative
    }

    response = client.post("/api/v1/similarities/vector", json=request_data)
    assert response.status_code == 422
