"""Tests for studies router endpoints."""

import pytest
from httpx import AsyncClient


# ==== GET /api/studies tests ====


@pytest.mark.asyncio
async def test_list_studies_success(
    client: AsyncClient,
    mock_vector_store,
):
    """Test successful study list retrieval."""
    response = await client.get("/api/studies")

    assert response.status_code == 200
    data = response.json()

    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert "studies" in data
    assert data["limit"] == 20
    assert data["offset"] == 0

    mock_vector_store.get_studies.assert_called_once()


@pytest.mark.asyncio
async def test_list_studies_with_query(
    client: AsyncClient,
    mock_vector_store,
):
    """Test study list with search query."""
    response = await client.get("/api/studies?q=diabetes")

    assert response.status_code == 200

    mock_vector_store.get_studies.assert_called_once()
    call_args = mock_vector_store.get_studies.call_args
    assert call_args.kwargs["q"] == "diabetes"


@pytest.mark.asyncio
async def test_list_studies_with_trait_filter(
    client: AsyncClient,
    mock_vector_store,
):
    """Test study list filtered by trait."""
    response = await client.get("/api/studies?trait=body%20mass%20index")

    assert response.status_code == 200

    mock_vector_store.get_studies.assert_called_once()
    call_args = mock_vector_store.get_studies.call_args
    assert call_args.kwargs["trait"] == "body mass index"


@pytest.mark.asyncio
async def test_list_studies_with_model_filter(
    client: AsyncClient,
    mock_vector_store,
):
    """Test study list filtered by model."""
    response = await client.get("/api/studies?model=gpt-4o")

    assert response.status_code == 200

    mock_vector_store.get_studies.assert_called_once()
    call_args = mock_vector_store.get_studies.call_args
    assert call_args.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_list_studies_pagination(
    client: AsyncClient,
    mock_vector_store,
):
    """Test study list pagination."""
    response = await client.get("/api/studies?limit=50&offset=100")

    assert response.status_code == 200
    data = response.json()

    assert data["limit"] == 50
    assert data["offset"] == 100

    mock_vector_store.get_studies.assert_called_once()
    call_args = mock_vector_store.get_studies.call_args
    assert call_args.kwargs["limit"] == 50
    assert call_args.kwargs["offset"] == 100


@pytest.mark.asyncio
async def test_list_studies_limit_validation(
    client: AsyncClient,
    mock_vector_store,
):
    """Test study list limit validation (max 100)."""
    response = await client.get("/api/studies?limit=150")

    assert response.status_code == 422  # Validation error


# ==== GET /api/studies/{pmid}/extraction tests ====


@pytest.mark.asyncio
async def test_get_extraction_success(
    client: AsyncClient,
    mock_vector_store,
):
    """Test successful extraction retrieval."""
    response = await client.get("/api/studies/12345678/extraction")

    assert response.status_code == 200
    data = response.json()

    assert data["pmid"] == "12345678"
    assert data["model"] == "gpt-5"
    assert "traits" in data
    assert "results" in data
    assert "metadata" in data

    mock_vector_store.get_study_extraction.assert_called_once_with(
        pmid="12345678",
        model="gpt-5",
    )


@pytest.mark.asyncio
async def test_get_extraction_with_model(
    client: AsyncClient,
    mock_vector_store,
):
    """Test extraction retrieval with specific model."""
    response = await client.get(
        "/api/studies/12345678/extraction?model=gpt-4o"
    )

    assert response.status_code == 200

    mock_vector_store.get_study_extraction.assert_called_once_with(
        pmid="12345678",
        model="gpt-4o",
    )


@pytest.mark.asyncio
async def test_get_extraction_not_found(
    client: AsyncClient,
    mock_vector_store,
):
    """Test extraction retrieval for non-existent study."""
    mock_vector_store.get_study_extraction.return_value = None

    response = await client.get("/api/studies/99999999/extraction")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


# ==== GET /api/traits/autocomplete tests ====


@pytest.mark.asyncio
async def test_autocomplete_traits_success(
    client: AsyncClient,
    mock_vector_store,
):
    """Test successful trait autocomplete."""
    response = await client.get("/api/traits/autocomplete?q=body")

    assert response.status_code == 200
    data = response.json()

    assert "traits" in data
    assert len(data["traits"]) > 0

    mock_vector_store.search_traits.assert_called_once()


@pytest.mark.asyncio
async def test_autocomplete_traits_min_length(
    client: AsyncClient,
    mock_vector_store,
):
    """Test trait autocomplete minimum length validation."""
    response = await client.get("/api/traits/autocomplete?q=b")

    assert response.status_code == 422  # Validation error - min_length=2


@pytest.mark.asyncio
async def test_autocomplete_traits_custom_limit(
    client: AsyncClient,
    mock_vector_store,
):
    """Test trait autocomplete with custom limit."""
    response = await client.get("/api/traits/autocomplete?q=body&limit=5")

    assert response.status_code == 200

    mock_vector_store.search_traits.assert_called_once()
    call_args = mock_vector_store.search_traits.call_args
    assert call_args.kwargs["limit"] == 5


# ==== GET /api/studies/autocomplete tests ====


@pytest.mark.asyncio
async def test_autocomplete_studies_success(
    client: AsyncClient,
    mock_vector_store,
):
    """Test successful study autocomplete."""
    response = await client.get("/api/studies/autocomplete?q=diabetes")

    assert response.status_code == 200
    data = response.json()

    assert "studies" in data
    assert len(data["studies"]) > 0

    mock_vector_store.search_studies.assert_called_once()


@pytest.mark.asyncio
async def test_autocomplete_studies_min_length(
    client: AsyncClient,
    mock_vector_store,
):
    """Test study autocomplete minimum length validation."""
    response = await client.get("/api/studies/autocomplete?q=d")

    assert response.status_code == 422  # Validation error - min_length=2


# ==== GET /api/statistics tests ====


@pytest.mark.asyncio
async def test_get_statistics_success(
    client: AsyncClient,
    mock_statistics,
):
    """Test successful statistics retrieval."""
    response = await client.get("/api/statistics")

    assert response.status_code == 200
    data = response.json()

    assert "overall" in data
    assert "model_similarity_stats" in data
    assert "model_evidence_stats" in data

    mock_statistics.get_overall_statistics.assert_called_once()
    mock_statistics.get_model_similarity_stats.assert_called_once()
    mock_statistics.get_model_evidence_stats.assert_called_once()
