"""Tests for similarity router endpoints."""

import pytest
from httpx import AsyncClient


# ==== GET /api/studies/{pmid}/similar/trait tests ====


@pytest.mark.asyncio
async def test_get_similar_by_trait_success(
    client: AsyncClient,
    mock_trait_profile,
):
    """Test successful trait similarity retrieval."""
    response = await client.get("/api/studies/12345678/similar/trait")

    assert response.status_code == 200
    data = response.json()

    assert data["query_pmid"] == "12345678"
    assert data["query_model"] == "gpt-5"
    assert "query_title" in data
    assert "query_trait_count" in data
    assert "similar_studies" in data
    assert len(data["similar_studies"]) > 0

    mock_trait_profile.get_similar_by_trait.assert_called_once()


@pytest.mark.asyncio
async def test_get_similar_by_trait_with_model(
    client: AsyncClient,
    mock_trait_profile,
):
    """Test trait similarity with specific model."""
    response = await client.get(
        "/api/studies/12345678/similar/trait?model=gpt-4o"
    )

    assert response.status_code == 200

    mock_trait_profile.get_similar_by_trait.assert_called_once()
    call_args = mock_trait_profile.get_similar_by_trait.call_args
    assert call_args.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_get_similar_by_trait_with_limit(
    client: AsyncClient,
    mock_trait_profile,
):
    """Test trait similarity with custom limit."""
    response = await client.get("/api/studies/12345678/similar/trait?limit=5")

    assert response.status_code == 200

    mock_trait_profile.get_similar_by_trait.assert_called_once()
    call_args = mock_trait_profile.get_similar_by_trait.call_args
    assert call_args.kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_get_similar_by_trait_limit_validation(
    client: AsyncClient,
    mock_trait_profile,
):
    """Test trait similarity limit validation (max 50)."""
    response = await client.get(
        "/api/studies/12345678/similar/trait?limit=100"
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_get_similar_by_trait_not_found(
    client: AsyncClient,
    mock_trait_profile,
):
    """Test trait similarity for non-existent study."""
    mock_trait_profile.get_similar_by_trait.return_value = None

    response = await client.get("/api/studies/99999999/similar/trait")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_get_similar_by_trait_empty_results(
    client: AsyncClient,
    mock_trait_profile,
):
    """Test trait similarity with no similar studies."""
    mock_trait_profile.get_similar_by_trait.return_value = {
        "query_pmid": "12345678",
        "query_model": "gpt-5",
        "query_title": "Test study",
        "query_trait_count": 1,
        "similar_studies": [],
    }

    response = await client.get("/api/studies/12345678/similar/trait")

    assert response.status_code == 200
    data = response.json()
    assert data["similar_studies"] == []


@pytest.mark.asyncio
async def test_get_similar_by_trait_response_structure(
    client: AsyncClient,
    mock_trait_profile,
):
    """Test trait similarity response structure."""
    response = await client.get("/api/studies/12345678/similar/trait")

    assert response.status_code == 200
    data = response.json()

    # Check similar study structure
    for study in data["similar_studies"]:
        assert "pmid" in study
        assert "title" in study
        assert "trait_profile_similarity" in study
        assert "trait_jaccard_similarity" in study
        assert "trait_count" in study
        assert isinstance(study["trait_profile_similarity"], float)
        assert isinstance(study["trait_jaccard_similarity"], float)
        assert isinstance(study["trait_count"], int)


# ==== GET /api/studies/{pmid}/similar/evidence tests ====


@pytest.mark.asyncio
async def test_get_similar_by_evidence_success(
    client: AsyncClient,
    mock_evidence_profile,
):
    """Test successful evidence similarity retrieval."""
    response = await client.get("/api/studies/12345678/similar/evidence")

    assert response.status_code == 200
    data = response.json()

    assert data["query_pmid"] == "12345678"
    assert data["query_model"] == "gpt-5"
    assert "query_title" in data
    assert "query_result_count" in data
    assert "similar_studies" in data
    assert len(data["similar_studies"]) > 0

    mock_evidence_profile.get_similar_by_evidence.assert_called_once()


@pytest.mark.asyncio
async def test_get_similar_by_evidence_with_model(
    client: AsyncClient,
    mock_evidence_profile,
):
    """Test evidence similarity with specific model."""
    response = await client.get(
        "/api/studies/12345678/similar/evidence?model=gpt-4o"
    )

    assert response.status_code == 200

    mock_evidence_profile.get_similar_by_evidence.assert_called_once()
    call_args = mock_evidence_profile.get_similar_by_evidence.call_args
    assert call_args.kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_get_similar_by_evidence_with_limit(
    client: AsyncClient,
    mock_evidence_profile,
):
    """Test evidence similarity with custom limit."""
    response = await client.get(
        "/api/studies/12345678/similar/evidence?limit=5"
    )

    assert response.status_code == 200

    mock_evidence_profile.get_similar_by_evidence.assert_called_once()
    call_args = mock_evidence_profile.get_similar_by_evidence.call_args
    assert call_args.kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_get_similar_by_evidence_limit_validation(
    client: AsyncClient,
    mock_evidence_profile,
):
    """Test evidence similarity limit validation (max 50)."""
    response = await client.get(
        "/api/studies/12345678/similar/evidence?limit=100"
    )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_get_similar_by_evidence_not_found(
    client: AsyncClient,
    mock_evidence_profile,
):
    """Test evidence similarity for non-existent study."""
    mock_evidence_profile.get_similar_by_evidence.return_value = None

    response = await client.get("/api/studies/99999999/similar/evidence")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_get_similar_by_evidence_empty_results(
    client: AsyncClient,
    mock_evidence_profile,
):
    """Test evidence similarity with no similar studies."""
    mock_evidence_profile.get_similar_by_evidence.return_value = {
        "query_pmid": "12345678",
        "query_model": "gpt-5",
        "query_title": "Test study",
        "query_result_count": 1,
        "similar_studies": [],
    }

    response = await client.get("/api/studies/12345678/similar/evidence")

    assert response.status_code == 200
    data = response.json()
    assert data["similar_studies"] == []


@pytest.mark.asyncio
async def test_get_similar_by_evidence_response_structure(
    client: AsyncClient,
    mock_evidence_profile,
):
    """Test evidence similarity response structure."""
    response = await client.get("/api/studies/12345678/similar/evidence")

    assert response.status_code == 200
    data = response.json()

    # Check similar study structure
    for study in data["similar_studies"]:
        assert "pmid" in study
        assert "title" in study
        assert "direction_concordance" in study
        assert "matched_pairs" in study
        assert "match_type_exact" in study
        assert "match_type_fuzzy" in study
        assert "match_type_efo" in study
        assert isinstance(study["direction_concordance"], float)
        assert isinstance(study["matched_pairs"], int)
        assert isinstance(study["match_type_exact"], bool)
        assert isinstance(study["match_type_fuzzy"], bool)
        assert isinstance(study["match_type_efo"], bool)


# ==== Health check tests ====


@pytest.mark.asyncio
async def test_health_check_all_healthy(
    client: AsyncClient,
    mock_database_connections,
):
    """Test health check when all databases are healthy."""
    response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["databases"]["vector_store"] is True
    assert data["databases"]["trait_profile"] is True
    assert data["databases"]["evidence_profile"] is True


@pytest.mark.asyncio
async def test_health_check_degraded(
    client: AsyncClient,
    mock_database_connections,
):
    """Test health check when some databases are unhealthy."""
    # Make one connection fail
    mock_database_connections["trait_profile"].side_effect = Exception(
        "Connection failed"
    )

    response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "degraded"
