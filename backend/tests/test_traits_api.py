"""Tests for the traits API endpoints."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_analytics_service, get_trait_service
from app.main import app
from app.models.database import SimilaritySearchResult, TraitEmbedding
from app.services.database_service import AnalyticsService, TraitService

client = TestClient(app)


# ---- Test Helpers ----


def override_trait_service_dependency(mock_service):
    """Helper to override trait service dependency."""

    async def mock_get_trait_service():
        yield mock_service

    app.dependency_overrides[get_trait_service] = mock_get_trait_service


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
def mock_trait_service():
    """Mock TraitService for testing."""
    service = Mock(spec=TraitService)

    # Mock trait repository
    service.trait_repo = Mock()
    service.study_repo = Mock()
    service.efo_repo = Mock()

    return service


@pytest.fixture
def mock_analytics_service():
    """Mock AnalyticsService for testing."""
    service = Mock(spec=AnalyticsService)
    service.trait_repo = Mock()
    return service


@pytest.fixture
def sample_trait_embedding():
    """Sample trait embedding for testing."""
    return TraitEmbedding(
        trait_index=1, trait_label="Body mass index", vector=[0.1] * 200
    )


@pytest.fixture
def sample_trait_list_items():
    """Sample trait list items for testing."""
    return [
        {
            "trait_index": 1,
            "trait_label": "Body mass index",
            "appearance_count": 150,
        },
        {
            "trait_index": 2,
            "trait_label": "Type 2 diabetes",
            "appearance_count": 120,
        },
        {
            "trait_index": 3,
            "trait_label": "Coronary artery disease",
            "appearance_count": 90,
        },
    ]


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


# ---- Tests for Trait Listing Endpoint ----


def test_list_traits_success(mock_trait_service, sample_trait_list_items):
    """Test successful trait listing."""

    # Mock repository responses
    mock_trait_service.trait_repo.execute_query.side_effect = [
        [(3,)],  # Total count query
        [
            (1, "Body mass index", 150),
            (2, "Type 2 diabetes", 120),
            (3, "Coronary artery disease", 90),
        ],  # Results query
    ]

    # Override the dependency
    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 3
        assert data["pagination"]["total_items"] == 3
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 50
        assert data["pagination"]["has_next"] is False
        assert data["pagination"]["has_previous"] is False

        first_trait = data["data"][0]
        assert first_trait["trait_index"] == 1
        assert first_trait["trait_label"] == "Body mass index"
        assert first_trait["appearance_count"] == 150
    finally:
        # Clean up the dependency override
        clear_dependency_overrides()


def test_list_traits_with_pagination(mock_trait_service):
    """Test trait listing with pagination parameters."""

    # Mock repository responses
    mock_trait_service.trait_repo.execute_query.side_effect = [
        [(100,)],  # Total count query
        [(1, "Body mass index", 150)],  # Results query
    ]

    # Override the dependency
    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/?page=2&page_size=20")

        assert response.status_code == 200
        data = response.json()

        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 20
        assert data["pagination"]["total_pages"] == 5
        assert data["pagination"]["has_next"] is True
        assert data["pagination"]["has_previous"] is True
    finally:
        # Clean up the dependency override
        clear_dependency_overrides()


def test_list_traits_with_filters(mock_trait_service):
    """Test trait listing with minimum appearances filter."""

    # Mock repository responses
    mock_trait_service.trait_repo.execute_query.side_effect = [
        [(2,)],  # Total count query
        [
            (1, "Body mass index", 150),
            (2, "Type 2 diabetes", 120),
        ],  # Results query
    ]

    # Override the dependency
    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/?min_appearances=100")

        assert response.status_code == 200
        data = response.json()

        assert data["pagination"]["total_items"] == 2
        assert len(data["data"]) == 2
    finally:
        # Clean up the dependency override
        clear_dependency_overrides()


# ---- Tests for Trait Search Endpoint ----


def test_search_traits_success(mock_trait_service):
    """Test successful trait search."""

    # Mock repository responses
    mock_trait_service.trait_repo.execute_query.side_effect = [
        [(2,)],  # Total count query
        [(1, "Body mass index", 150), (2, "Body weight", 80)],  # Results query
    ]

    # Override the dependency
    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/search?q=body")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 2
        assert data["pagination"]["total_items"] == 2

        # Check that query was properly formatted with ILIKE pattern
        call_args = mock_trait_service.trait_repo.execute_query.call_args_list
        assert "%body%" in call_args[0][0][1]  # First call is count query
        assert "%body%" in call_args[1][0][1]  # Second call is results query
    finally:
        # Clean up the dependency override
        clear_dependency_overrides()


def test_search_traits_with_pagination(mock_trait_service):
    """Test trait search with pagination."""

    # Mock repository responses
    mock_trait_service.trait_repo.execute_query.side_effect = [
        [(50,)],  # Total count query
        [(1, "Body mass index", 150)],  # Results query
    ]

    # Override the dependency
    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get(
            "/api/v1/traits/search?q=body&page=2&page_size=10"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 10
        assert data["pagination"]["total_pages"] == 5
    finally:
        # Clean up the dependency override
        clear_dependency_overrides()


def test_search_traits_no_query(mock_trait_service):
    """Test trait search without query parameter."""
    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/search")

        assert (
            response.status_code == 422
        )  # Validation error for missing required parameter
    finally:
        clear_dependency_overrides()


# ---- Tests for Trait Details Endpoint ----


def test_get_trait_details_success(
    mock_trait_service,
    mock_analytics_service,
    sample_trait_embedding,
):
    """Test successful trait details retrieval."""
    # Mock trait repository
    mock_trait_service.trait_repo.get_trait_by_index.return_value = (
        sample_trait_embedding
    )
    mock_trait_service.trait_repo.find_similar_traits.return_value = []
    mock_trait_service.study_repo.execute_query.return_value = [
        (
            1,
            "gpt-4-1",
            "12345",
            {"test": "metadata"},
            {"test": "results"},
            "Test Title",
            "Nature",
            "2023-01-01",
        )
    ]
    mock_trait_service.efo_repo.find_trait_efo_mappings.return_value = []

    # Mock analytics service
    mock_analytics_service.get_trait_statistics.return_value = {
        "study_count": 10
    }

    # Override dependencies
    override_trait_service_dependency(mock_trait_service)
    override_analytics_service_dependency(mock_analytics_service)

    try:
        response = client.get("/api/v1/traits/1")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        trait_detail = data["data"]

        assert trait_detail["trait"]["trait_index"] == 1
        assert trait_detail["trait"]["trait_label"] == "Body mass index"
        assert "statistics" in trait_detail
        assert "studies" in trait_detail
        assert "similar_traits" in trait_detail
        assert "efo_mappings" in trait_detail
    finally:
        clear_dependency_overrides()


def test_get_trait_details_not_found(
    mock_trait_service, mock_analytics_service
):
    """Test trait details for non-existent trait."""
    mock_trait_service.trait_repo.get_trait_by_index.return_value = None

    override_trait_service_dependency(mock_trait_service)
    override_analytics_service_dependency(mock_analytics_service)

    try:
        response = client.get("/api/v1/traits/99999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"]["message"].lower()
    finally:
        clear_dependency_overrides()


def test_get_trait_details_with_options(
    mock_trait_service,
    mock_analytics_service,
    sample_trait_embedding,
):
    """Test trait details with include options."""
    # Mock trait repository
    mock_trait_service.trait_repo.get_trait_by_index.return_value = (
        sample_trait_embedding
    )
    mock_trait_service.trait_repo.find_similar_traits.return_value = []
    mock_trait_service.study_repo.execute_query.return_value = []
    mock_trait_service.efo_repo.find_trait_efo_mappings.return_value = []

    # Mock analytics service
    mock_analytics_service.get_trait_statistics.return_value = {
        "study_count": 0
    }

    # Override dependencies
    override_trait_service_dependency(mock_trait_service)
    override_analytics_service_dependency(mock_analytics_service)

    try:
        response = client.get(
            "/api/v1/traits/1?include_studies=false&include_similar=false&include_efo=false"
        )

        assert response.status_code == 200
        data = response.json()

        trait_detail = data["data"]
        assert len(trait_detail["studies"]) == 0
        assert len(trait_detail["similar_traits"]) == 0
        assert len(trait_detail["efo_mappings"]) == 0
    finally:
        clear_dependency_overrides()


# ---- Tests for Trait Studies Endpoint ----


def test_get_trait_studies_success(mock_trait_service, sample_trait_embedding):
    """Test successful trait studies retrieval."""
    # Mock trait repository
    mock_trait_service.trait_repo.get_trait_by_index.return_value = (
        sample_trait_embedding
    )
    mock_trait_service.study_repo.execute_query.side_effect = [
        [(2,)],  # Total count query
        [
            (
                1,
                "gpt-4-1",
                "12345",
                {"test": "metadata"},
                {"test": "results"},
                "Test Title",
                "Abstract",
                "Nature",
                "2023-01-01",
                "Institution",
            ),
            (
                2,
                "gpt-3.5-turbo",
                "12346",
                {"test": "metadata2"},
                {"test": "results2"},
                "Test Title 2",
                "Abstract 2",
                "Science",
                "2023-02-01",
                "Institution 2",
            ),
        ],  # Results query
    ]

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/1/studies")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 2
        assert data["pagination"]["total_items"] == 2

        first_study = data["data"][0]
        assert first_study["id"] == 1
        assert first_study["model"] == "gpt-4-1"
        assert first_study["pmid"] == "12345"
        assert first_study["title"] == "Test Title"
    finally:
        clear_dependency_overrides()


def test_get_trait_studies_with_filters(
    mock_trait_service, sample_trait_embedding
):
    """Test trait studies with filters."""
    # Mock trait repository
    mock_trait_service.trait_repo.get_trait_by_index.return_value = (
        sample_trait_embedding
    )
    mock_trait_service.study_repo.execute_query.side_effect = [
        [(1,)],  # Total count query
        [
            (
                1,
                "gpt-4-1",
                "12345",
                {"test": "metadata"},
                {"test": "results"},
                "Test Title",
                "Abstract",
                "Nature",
                "2023-01-01",
                "Institution",
            ),
        ],  # Results query
    ]

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get(
            "/api/v1/traits/1/studies?model=gpt-4-1&journal=Nature&date_from=2023-01-01&date_to=2023-12-31"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 1
    finally:
        clear_dependency_overrides()


def test_get_trait_studies_trait_not_found(
    mock_trait_service, mock_analytics_service
):
    """Test trait studies for non-existent trait."""
    mock_trait_service.trait_repo.get_trait_by_index.return_value = None

    override_trait_service_dependency(mock_trait_service)
    override_analytics_service_dependency(mock_analytics_service)

    try:
        response = client.get("/api/v1/traits/99999/studies")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"]["message"].lower()
    finally:
        clear_dependency_overrides()


# ---- Tests for Similar Traits Endpoint ----


def test_get_similar_traits_success(
    mock_trait_service,
    sample_trait_embedding,
    sample_similarity_results,
):
    """Test successful similar traits retrieval."""
    # Mock trait repository
    mock_trait_service.trait_repo.get_trait_by_index.return_value = (
        sample_trait_embedding
    )
    mock_trait_service.find_similar_traits.return_value = (
        sample_similarity_results
    )

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/1/similar")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 2

        first_similar = data["data"][0]
        assert first_similar["query_id"] == "1"
        assert first_similar["result_label"] == "Obesity"
        assert first_similar["similarity"] == 0.85
    finally:
        clear_dependency_overrides()


def test_get_similar_traits_with_parameters(
    mock_trait_service, sample_trait_embedding
):
    """Test similar traits with custom parameters."""
    # Mock trait repository
    mock_trait_service.trait_repo.get_trait_by_index.return_value = (
        sample_trait_embedding
    )
    mock_trait_service.find_similar_traits.return_value = []

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get(
            "/api/v1/traits/1/similar?max_results=5&similarity_threshold=0.8"
        )

        assert response.status_code == 200

        # Check that the service was called with correct parameters
        mock_trait_service.find_similar_traits.assert_called_once()
        call_args = mock_trait_service.find_similar_traits.call_args
        assert call_args[0][0] == 1  # trait_index
        assert call_args[0][1].max_results == 5
        assert call_args[0][1].min_similarity == 0.8
    finally:
        clear_dependency_overrides()


# ---- Tests for EFO Mappings Endpoint ----


def test_get_trait_efo_mappings_success(
    mock_trait_service, sample_trait_embedding
):
    """Test successful EFO mappings retrieval."""
    # Mock trait repository
    mock_trait_service.trait_repo.get_trait_by_index.return_value = (
        sample_trait_embedding
    )
    mock_trait_service.efo_repo.find_trait_efo_mappings.return_value = [
        SimilaritySearchResult(
            query_id="1",
            query_label="Body mass index",
            result_id="EFO_0004340",
            result_label="body mass index",
            similarity=0.95,
        )
    ]

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/1/efo-mappings")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 1

        first_mapping = data["data"][0]
        assert first_mapping["result_id"] == "EFO_0004340"
        assert first_mapping["result_label"] == "body mass index"
        assert first_mapping["similarity"] == 0.95
    finally:
        clear_dependency_overrides()


# ---- Tests for Traits Overview Endpoint ----


def test_get_traits_overview_success(mock_analytics_service):
    """Test successful traits overview retrieval."""
    # Mock analytics service
    mock_analytics_service.trait_repo.get_count.return_value = 1000
    mock_analytics_service.trait_repo.execute_query.side_effect = [
        [(50000,)],  # Total appearances
        [
            (1, "Body mass index", 150),
            (2, "Type 2 diabetes", 120),
        ],  # Top traits
        [("1", 100), ("2-5", 200), ("6-10", 150)],  # Distribution
        [
            ("gpt-4-1", 800, 5000),
            ("gpt-3.5-turbo", 750, 4500),
        ],  # Model coverage
    ]

    override_analytics_service_dependency(mock_analytics_service)

    try:
        response = client.get("/api/v1/traits/stats/overview")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        overview = data["data"]

        assert overview["total_traits"] == 1000
        assert overview["total_appearances"] == 50000
        assert overview["average_appearances"] == 50.0
        assert len(overview["top_traits"]) == 2
        assert len(overview["appearance_distribution"]) == 3
        assert len(overview["model_coverage"]) == 2
    finally:
        clear_dependency_overrides()


# ---- Tests for Bulk Traits Endpoint ----


def test_get_traits_bulk_success(mock_trait_service):
    """Test successful bulk traits retrieval."""
    # Mock trait repository
    mock_trait_service.trait_repo.execute_query.return_value = [
        (1, "Body mass index", [0.1] * 200),
        (2, "Type 2 diabetes", [0.2] * 200),
        (3, "Coronary artery disease", [0.3] * 200),
    ]

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.post("/api/v1/traits/bulk", json=[1, 2, 3])

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 3

        first_trait = data["data"][0]
        assert first_trait["trait_index"] == 1
        assert first_trait["trait_label"] == "Body mass index"
        assert first_trait["vector"] is not None
        assert len(first_trait["vector"]) == 200
    finally:
        clear_dependency_overrides()


def test_get_traits_bulk_empty_list(mock_trait_service):
    """Test bulk traits with empty list."""
    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.post("/api/v1/traits/bulk", json=[])

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["data"]) == 0
    finally:
        clear_dependency_overrides()


def test_get_traits_bulk_too_many_indices(mock_trait_service):
    """Test bulk traits with too many indices."""
    override_trait_service_dependency(mock_trait_service)

    try:
        # Create a list with 1001 indices (exceeds the 1000 limit)
        large_list = list(range(1, 1002))

        response = client.post("/api/v1/traits/bulk", json=large_list)

        assert response.status_code == 400
        data = response.json()
        assert "Maximum 1000" in data["error"]["message"]

    finally:
        clear_dependency_overrides()


# ---- Tests for Error Handling ----


def test_list_traits_database_error(mock_trait_service):
    """Test trait listing with database error."""
    mock_trait_service.trait_repo.execute_query.side_effect = Exception(
        "Database connection failed"
    )

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/")

        assert response.status_code == 500
        data = response.json()
        assert "Failed to list traits" in data["error"]["message"]

    finally:
        clear_dependency_overrides()


def test_search_traits_database_error(mock_trait_service):
    """Test trait search with database error."""
    mock_trait_service.trait_repo.execute_query.side_effect = Exception(
        "Database connection failed"
    )

    override_trait_service_dependency(mock_trait_service)

    try:
        response = client.get("/api/v1/traits/search?q=body")

        assert response.status_code == 500
        data = response.json()
        assert "Failed to search traits" in data["error"]["message"]

    finally:
        clear_dependency_overrides()


# ---- Tests for Parameter Validation ----


def test_list_traits_invalid_pagination():
    """Test trait listing with invalid pagination parameters."""
    # Invalid page number (0)
    response = client.get("/api/v1/traits/?page=0")
    assert response.status_code == 422

    # Invalid page size (too large)
    response = client.get("/api/v1/traits/?page_size=2000")
    assert response.status_code == 422

    # Invalid page size (negative)
    response = client.get("/api/v1/traits/?page_size=-1")
    assert response.status_code == 422


def test_search_traits_invalid_pagination():
    """Test trait search with invalid pagination parameters."""
    # Invalid page size (too large)
    response = client.get("/api/v1/traits/search?q=body&page_size=600")
    assert response.status_code == 422


def test_get_similar_traits_invalid_parameters():
    """Test similar traits with invalid parameters."""
    # Invalid similarity threshold (too high)
    response = client.get("/api/v1/traits/1/similar?similarity_threshold=1.5")
    assert response.status_code == 422

    # Invalid max results (too high)
    response = client.get("/api/v1/traits/1/similar?max_results=200")
    assert response.status_code == 422


def test_get_efo_mappings_invalid_parameters():
    """Test EFO mappings with invalid parameters."""
    # Invalid similarity threshold (negative)
    response = client.get(
        "/api/v1/traits/1/efo-mappings?similarity_threshold=-0.1"
    )
    assert response.status_code == 422

    # Invalid max results (too high)
    response = client.get("/api/v1/traits/1/efo-mappings?max_results=100")
    assert response.status_code == 422
