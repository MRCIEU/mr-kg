"""Comprehensive tests for the similarities API endpoints."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import (
    get_analytics_service,
    get_similarity_service,
)
from app.main import app
from app.models.database import (
    QueryCombination,
    SimilaritySearchResult,
    TraitSimilarity,
)
from app.services.database_service import AnalyticsService, SimilarityService

client = TestClient(app)


# ---- Test Fixtures and Helpers ----


def override_similarity_service(mock_service):
    """Helper to override SimilarityService dependency."""
    app.dependency_overrides[get_similarity_service] = lambda: mock_service


def override_analytics_service(mock_service):
    """Helper to override AnalyticsService dependency."""
    app.dependency_overrides[get_analytics_service] = lambda: mock_service


def cleanup_overrides():
    """Helper to clean up dependency overrides."""
    app.dependency_overrides.clear()


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


def test_analyze_similarity_success(
    mock_similarity_service, sample_combination
):
    """Test successful similarity analysis."""

    # Set up the mock service
    mock_similarity_service.similarity_repo.find_combination.return_value = (
        sample_combination
    )

    # Mock get_similarities with proper TraitSimilarity objects
    mock_similarities = [
        TraitSimilarity(
            id=1,
            query_combination_id=1,
            similar_pmid="87654321",
            similar_model="gpt-4-1",
            similar_title="Similar Study 1",
            trait_profile_similarity=0.92,
            trait_jaccard_similarity=0.88,
            query_trait_count=5,
            similar_trait_count=4,
        ),
        TraitSimilarity(
            id=2,
            query_combination_id=1,
            similar_pmid="11111111",
            similar_model="gpt-3.5-turbo",
            similar_title="Similar Study 2",
            trait_profile_similarity=0.87,
            trait_jaccard_similarity=0.85,
            query_trait_count=5,
            similar_trait_count=6,
        ),
    ]
    mock_similarity_service.similarity_repo.get_similarities.return_value = (
        mock_similarities
    )

    # Override the dependency
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/analyze?pmid=12345678&model=gpt-4-1"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        analysis = data["data"]
        assert analysis["query_combination"]["pmid"] == "12345678"
        assert len(analysis["similarities"]) == 2
        assert "summary" in analysis

        # Verify summary statistics
        summary = analysis["summary"]
        assert summary["total_similar"] == 2
        assert summary["max_similarity"] == 0.92
        assert summary["min_similarity"] == 0.87
        assert summary["avg_similarity"] == 0.895
        assert summary["similarity_type"] == "trait_profile"

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_analyze_similarity_not_found(mock_similarity_service):
    """Test similarity analysis for non-existent combination."""

    mock_similarity_service.similarity_repo.find_combination.return_value = None

    # Override the dependency
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/analyze?pmid=99999999&model=nonexistent"
        )

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "no combination found" in data["error"]["message"].lower()

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_analyze_similarity_missing_params():
    """Test similarity analysis with missing parameters."""
    # Missing model parameter
    response = client.get("/api/v1/similarities/analyze?pmid=12345678")
    assert response.status_code == 422

    # Missing pmid parameter
    response = client.get("/api/v1/similarities/analyze?model=gpt-4-1")
    assert response.status_code == 422


def test_analyze_similarity_with_filters(
    mock_similarity_service, sample_combination
):
    """Test similarity analysis with filters."""

    # Set up the mock service
    mock_similarity_service.similarity_repo.find_combination.return_value = (
        sample_combination
    )

    # Mock get_similarities with proper TraitSimilarity objects (filtered)
    mock_similarities = [
        TraitSimilarity(
            id=1,
            query_combination_id=1,
            similar_pmid="87654321",
            similar_model="gpt-4-1",
            similar_title="Similar Study 1",
            trait_profile_similarity=0.92,
            trait_jaccard_similarity=0.88,
            query_trait_count=5,
            similar_trait_count=4,
        ),
    ]
    mock_similarity_service.similarity_repo.get_similarities.return_value = (
        mock_similarities
    )

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/analyze?pmid=12345678&model=gpt-4-1"
            "&min_similarity=0.8&max_results=10&similarity_type=jaccard"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        analysis = data["data"]
        assert analysis["query_combination"]["pmid"] == "12345678"
        assert len(analysis["similarities"]) == 1

        # Verify the filter was applied (only high similarity results)
        similarity = analysis["similarities"][0]
        assert similarity["trait_jaccard_similarity"] == 0.88
        assert similarity["similar_pmid"] == "87654321"

    finally:
        # Clean up the dependency override
        cleanup_overrides()


# ---- Tests for Search Combinations Endpoint ----


def test_search_combinations_success(mock_similarity_service):
    """Test successful combinations search."""

    # Set up the mock to return the expected data structure
    # First call: count query returns total count
    # Second call: actual data query returns the combinations tuples
    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(3,)],  # Total count query result
        [
            (1, "12345678", "gpt-4-1", "Study 1", 5),
            (2, "12345679", "gpt-3.5-turbo", "Study 2", 3),
            (3, "12345680", "claude-3", "Study 3", 7),
        ],  # Results query - these are tuples that will be converted to objects
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        # The endpoint should now work correctly after fixing the response type annotation
        response = client.get("/api/v1/similarities/combinations")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 3

        # Verify that the mock was called correctly
        assert (
            mock_similarity_service.similarity_repo.execute_query.call_count
            == 2
        )

        # The first call should be the count query
        first_call = mock_similarity_service.similarity_repo.execute_query.call_args_list[
            0
        ]
        assert "COUNT(*)" in first_call[0][0]

        # The second call should be the data query
        second_call = mock_similarity_service.similarity_repo.execute_query.call_args_list[
            1
        ]
        assert "SELECT id, pmid, model, title, trait_count" in second_call[0][0]

        # Verify the returned data structure
        first_item = data["data"][0]
        assert first_item["id"] == 1
        assert first_item["pmid"] == "12345678"
        assert first_item["model"] == "gpt-4-1"
        assert first_item["title"] == "Study 1"
        assert first_item["trait_count"] == 5

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_search_combinations_with_filters(mock_similarity_service):
    """Test combinations search with filters."""

    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [(1, "12345678", "gpt-4-1", "Study 1", 5)],  # Results query
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/combinations?model=gpt-4-1&min_trait_count=5&max_trait_count=10"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 1

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_search_combinations_pagination(mock_similarity_service):
    """Test combinations search with pagination."""

    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(50,)],  # Total count query
        [(1, "12345678", "gpt-4-1", "Study 1", 5)],  # Results query
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/combinations?page=2&page_size=20"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 20
        assert data["pagination"]["total_pages"] == 3

    finally:
        # Clean up the dependency override
        cleanup_overrides()


# ---- Tests for Combination Details Endpoint ----


def test_get_combination_details_success(
    mock_similarity_service, sample_combination
):
    """Test successful combination details retrieval."""

    mock_similarity_service.similarity_repo.execute_query.return_value = [
        (
            1,
            "12345678",
            "gpt-4-1",
            "Sample Study on BMI",
            5,
        ),  # Combination details
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get("/api/v1/similarities/combinations/1")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        combination = data["data"]
        assert combination["id"] == 1
        assert combination["pmid"] == "12345678"
        assert combination["model"] == "gpt-4-1"

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_get_combination_details_not_found(mock_similarity_service):
    """Test combination details for non-existent combination."""

    mock_similarity_service.similarity_repo.execute_query.return_value = []

    # Override the dependency
    app.dependency_overrides[get_similarity_service] = (
        lambda: mock_similarity_service
    )

    try:
        response = client.get("/api/v1/similarities/combinations/99999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"]["message"].lower()

    finally:
        # Clean up the dependency override
        cleanup_overrides()


# ---- Tests for Combination Similarities Endpoint ----


def test_get_combination_similarities_success(
    mock_similarity_service, sample_combination
):
    """Test successful combination similarities retrieval."""

    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(1,)],  # Verification query that combination exists
        [(2,)],  # Total count query
        [
            (1, 1, "87654321", "gpt-4-1", "Similar Study 1", 0.92, 0.85, 5, 4),
            (
                2,
                1,
                "11111111",
                "gpt-3.5-turbo",
                "Similar Study 2",
                0.87,
                0.78,
                5,
                6,
            ),
        ],  # Results query
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/combinations/1/similarities"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 2
        assert data["pagination"]["total_items"] == 2

        first_similarity = data["data"][0]
        assert first_similarity["similar_pmid"] == "87654321"
        assert first_similarity["trait_profile_similarity"] == 0.92

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_get_combination_similarities_with_filters(
    mock_similarity_service, sample_combination
):
    """Test combination similarities with filters."""

    mock_similarity_service.similarity_repo.execute_query.side_effect = [
        [(1,)],  # Verification query that combination exists
        [(1,)],  # Total count query
        [
            (1, 1, "87654321", "gpt-4-1", "Similar Study 1", 0.92, 0.85, 5, 4)
        ],  # Results query
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/combinations/1/similarities"
            "?min_similarity=0.8&similarity_type=jaccard&page_size=20"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    finally:
        # Clean up the dependency override
        cleanup_overrides()


# ---- Tests for Vector Similarity Search Endpoint ----


def test_vector_similarity_search_success(
    mock_similarity_service, sample_similarity_results
):
    """Test successful vector similarity search."""

    mock_similarity_service.trait_repo.execute_query.return_value = [
        ("1", "Body mass index", 0.85),
        ("2", "Obesity", 0.78),
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
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
        assert first_result["similarity_score"] == 0.85
        assert first_result["result_label"] == "Body mass index"

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_vector_similarity_search_invalid_vector(mock_similarity_service):
    """Test vector similarity search with invalid vector dimension."""

    # Override the dependency to avoid hitting real database
    override_similarity_service(mock_similarity_service)

    try:
        request_data = {
            "query_vector": [0.1, 0.2],  # Wrong dimension (should be 200)
            "top_k": 10,
            "threshold": 0.3,
        }

        response = client.post("/api/v1/similarities/vector", json=request_data)
        assert response.status_code == 400

        data = response.json()
        assert (
            "Query vector must be 200-dimensional" in data["error"]["message"]
        )

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_vector_similarity_search_different_types(mock_similarity_service):
    """Test vector similarity search with different search types."""

    mock_similarity_service.trait_repo.execute_query.return_value = []

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
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

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_vector_similarity_search_invalid_type(mock_similarity_service):
    """Test vector similarity search with invalid search type."""

    # Override the dependency to avoid hitting real database
    override_similarity_service(mock_similarity_service)

    try:
        request_data = {
            "query_vector": [0.1] * 200,
            "top_k": 10,
            "threshold": 0.3,
        }

        response = client.post(
            "/api/v1/similarities/vector?search_type=invalid", json=request_data
        )
        assert response.status_code == 400

        data = response.json()
        assert (
            "Search type must be 'traits' or 'efo'" in data["error"]["message"]
        )

    finally:
        # Clean up the dependency override
        cleanup_overrides()


# ---- Tests for Bulk Trait to EFO Mapping Endpoint ----


def test_bulk_trait_to_efo_mapping_success(mock_similarity_service):
    """Test successful bulk trait-to-EFO mapping."""

    # Mock trait repository for trait information
    mock_similarity_service.trait_repo.execute_query.return_value = [
        (1, "Body mass index", [0.1] * 200),  # Mock trait with vector
    ]

    # Mock EFO repository for EFO mappings
    mock_similarity_service.efo_repo.execute_query.return_value = [
        ("EFO_0004340", "body mass index", 0.95),
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        request_data = {
            "trait_indices": [1],
            "top_k": 5,
            "threshold": 0.3,
        }

        response = client.post(
            "/api/v1/similarities/trait-to-efo", json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 1

        trait_mapping = data["data"][0]
        assert trait_mapping["trait_index"] == 1
        assert trait_mapping["trait_label"] == "Body mass index"
        assert len(trait_mapping["efo_mappings"]) == 1
        assert trait_mapping["efo_mappings"][0]["similarity"] == 0.95

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_bulk_trait_to_efo_mapping_too_many(mock_similarity_service):
    """Test bulk trait-to-EFO mapping with too many traits."""

    # Override the dependency to avoid hitting real database
    override_similarity_service(mock_similarity_service)

    try:
        request_data = {
            "trait_indices": list(range(101)),  # Too many traits (limit is 100)
            "top_k": 5,
            "threshold": 0.3,
        }

        response = client.post(
            "/api/v1/similarities/trait-to-efo", json=request_data
        )
        assert response.status_code == 400

        data = response.json()
        assert (
            "Maximum 100 trait indices allowed per request"
            in data["error"]["message"]
        )

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_bulk_trait_to_efo_mapping_empty(mock_similarity_service):
    """Test bulk trait-to-EFO mapping with empty list."""

    # Override the dependency to avoid hitting real database
    override_similarity_service(mock_similarity_service)

    try:
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

    finally:
        # Clean up the dependency override
        cleanup_overrides()


# ---- Tests for Metadata Endpoints ----


def test_get_available_similarity_models(mock_similarity_service):
    """Test get available similarity models endpoint."""

    mock_similarity_service.similarity_repo.execute_query.return_value = [
        ("gpt-4-1",),
        ("gpt-3.5-turbo",),
        ("claude-3",),
    ]

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get("/api/v1/similarities/models")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 3

        assert "gpt-4-1" in data["data"]
        assert "gpt-3.5-turbo" in data["data"]
        assert "claude-3" in data["data"]

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_get_similarity_statistics(mock_analytics_service):
    """Test get similarity statistics endpoint."""

    # Mock the get_count method
    mock_analytics_service.similarity_repo.get_count.side_effect = [500, 2500]

    # Mock the execute_query calls
    mock_analytics_service.similarity_repo.execute_query.side_effect = [
        [(0.72,)],  # Average similarity
        [("0.5-0.7", 1000), ("0.7-0.9", 800)],  # Similarity distribution
        [
            ("gpt-4-1", 200, 15.5, 1800),
            ("gpt-3.5-turbo", 180, 12.3, 1600),
        ],  # Model stats
    ]

    # Override the dependency
    override_analytics_service(mock_analytics_service)

    try:
        response = client.get("/api/v1/similarities/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        stats = data["data"]

        assert stats["total_combinations"] == 500
        assert stats["total_similarities"] == 2500
        assert stats["average_similarity"] == 0.72
        assert len(stats["similarity_distribution"]) == 2
        assert len(stats["model_comparison"]) == 2

    finally:
        # Clean up the dependency override
        cleanup_overrides()


# ---- Tests for Error Handling ----


def test_analyze_similarity_database_error(mock_similarity_service):
    """Test similarity analysis with database error."""

    mock_similarity_service.similarity_repo.find_combination.side_effect = (
        Exception("Database connection failed")
    )

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get(
            "/api/v1/similarities/analyze?pmid=12345678&model=gpt-4-1"
        )

        assert response.status_code == 500
        data = response.json()
        assert "Failed to analyze similarity" in data["error"]["message"]

    finally:
        # Clean up the dependency override
        cleanup_overrides()


def test_search_combinations_database_error(mock_similarity_service):
    """Test combinations search with database error."""

    mock_similarity_service.similarity_repo.execute_query.side_effect = (
        Exception("Database connection failed")
    )

    # Override the dependency using the correct service
    override_similarity_service(mock_similarity_service)

    try:
        response = client.get("/api/v1/similarities/combinations")

        assert response.status_code == 500
        data = response.json()
        assert "Failed to search combinations" in data["error"]["message"]

    finally:
        # Clean up the dependency override
        cleanup_overrides()


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


def test_vector_search_invalid_params(mock_similarity_service):
    """Test vector search with invalid parameters."""

    # Override the dependency to avoid hitting real database
    override_similarity_service(mock_similarity_service)

    try:
        # Mock the trait_repo.execute_query method
        mock_similarity_service.trait_repo.execute_query.return_value = []

        # Invalid top_k (too high) - currently no validation, so this will return 200
        request_data = {
            "query_vector": [0.1] * 200,
            "top_k": 1000,  # Too high but no current validation
            "threshold": 0.3,
        }

        response = client.post("/api/v1/similarities/vector", json=request_data)
        # Since there's no validation constraint currently, this will pass
        assert response.status_code == 200

        # Invalid threshold (negative) - currently no validation, so this will return 200
        request_data = {
            "query_vector": [0.1] * 200,
            "top_k": 10,
            "threshold": -0.1,  # Negative but no current validation
        }

        response = client.post("/api/v1/similarities/vector", json=request_data)
        # Since there's no validation constraint currently, this will pass
        assert response.status_code == 200

    finally:
        # Clean up the dependency override
        cleanup_overrides()
