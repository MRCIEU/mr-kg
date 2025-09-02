"""Tests for the studies and similarities API endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestStudiesAPI:
    """Test cases for studies API endpoints."""

    def test_list_studies(self):
        """Test studies listing endpoint."""
        response = client.get("/api/v1/studies/")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data

    def test_list_studies_with_filters(self):
        """Test studies listing with filters."""
        response = client.get("/api/v1/studies/?model=gpt-4o&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] == 10

    def test_search_studies(self):
        """Test studies search endpoint."""
        response = client.get("/api/v1/studies/search?q=diabetes")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total_count" in data

    def test_search_studies_invalid_query(self):
        """Test studies search with missing query."""
        response = client.get("/api/v1/studies/search")
        assert response.status_code == 422

    def test_get_study_details_not_found(self):
        """Test get study details for non-existent study."""
        response = client.get("/api/v1/studies/999999")
        assert response.status_code == 404

    def test_get_studies_by_pmid_not_found(self):
        """Test get studies by PMID for non-existent PMID."""
        response = client.get("/api/v1/studies/pmid/99999999")
        assert response.status_code == 404

    def test_get_available_models(self):
        """Test get available models endpoint."""
        response = client.get("/api/v1/studies/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_get_available_journals(self):
        """Test get available journals endpoint."""
        response = client.get("/api/v1/studies/journals")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_get_studies_overview(self):
        """Test get studies overview endpoint."""
        response = client.get("/api/v1/studies/stats/overview")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        overview = data["data"]
        assert "total_studies" in overview
        assert "total_pmids" in overview
        assert "model_distribution" in overview


class TestSimilaritiesAPI:
    """Test cases for similarities API endpoints."""

    def test_analyze_similarity_not_found(self):
        """Test analyze similarity for non-existent combination."""
        response = client.get(
            "/api/v1/similarities/analyze?pmid=99999999&model=nonexistent"
        )
        assert response.status_code == 404

    def test_analyze_similarity_missing_params(self):
        """Test analyze similarity with missing parameters."""
        response = client.get("/api/v1/similarities/analyze?pmid=12345")
        assert response.status_code == 422

    def test_search_combinations(self):
        """Test search combinations endpoint."""
        response = client.get("/api/v1/similarities/combinations")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total_count" in data

    def test_search_combinations_with_filters(self):
        """Test search combinations with filters."""
        response = client.get(
            "/api/v1/similarities/combinations?model=gpt-4o&min_trait_count=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_vector_similarity_search_invalid_vector(self):
        """Test vector similarity search with invalid vector."""
        request_data = {
            "query_vector": [0.1, 0.2],  # Wrong dimension
            "top_k": 10,
            "threshold": 0.3,
        }
        response = client.post("/api/v1/similarities/vector", json=request_data)
        assert response.status_code == 400

    def test_vector_similarity_search_invalid_type(self):
        """Test vector similarity search with invalid search type."""
        request_data = {
            "query_vector": [0.1] * 200,  # Correct dimension
            "top_k": 10,
            "threshold": 0.3,
        }
        response = client.post(
            "/api/v1/similarities/vector?search_type=invalid", json=request_data
        )
        assert response.status_code == 400

    def test_bulk_trait_to_efo_mapping_too_many(self):
        """Test bulk trait-to-EFO mapping with too many traits."""
        request_data = {
            "trait_indices": list(range(101)),  # Too many traits
            "top_k": 5,
            "threshold": 0.3,
        }
        response = client.post(
            "/api/v1/similarities/trait-to-efo", json=request_data
        )
        assert response.status_code == 400

    def test_bulk_trait_to_efo_mapping_empty(self):
        """Test bulk trait-to-EFO mapping with empty list."""
        request_data = {"trait_indices": [], "top_k": 5, "threshold": 0.3}
        response = client.post(
            "/api/v1/similarities/trait-to-efo", json=request_data
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []

    def test_get_similarity_statistics(self):
        """Test get similarity statistics endpoint."""
        response = client.get("/api/v1/similarities/stats")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        stats = data["data"]
        assert "total_combinations" in stats
        assert "total_similarities" in stats
        assert "average_similarity" in stats

    def test_get_available_similarity_models(self):
        """Test get available similarity models endpoint."""
        response = client.get("/api/v1/similarities/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_get_combination_details_not_found(self):
        """Test get combination details for non-existent combination."""
        response = client.get("/api/v1/similarities/combinations/999999")
        assert response.status_code == 404

    def test_get_combination_similarities_not_found(self):
        """Test get combination similarities for non-existent combination."""
        response = client.get(
            "/api/v1/similarities/combinations/999999/similarities"
        )
        assert response.status_code == 404

    def test_get_combination_similarities_with_filters(self):
        """Test get combination similarities with filters."""
        # This will fail with 404 for non-existent combination, but tests parameter validation
        response = client.get(
            "/api/v1/similarities/combinations/1/similarities"
            "?min_similarity=0.5&similarity_type=jaccard&page_size=20"
        )
        # Could be 404 (combination not found) or 200 (combination exists)
        assert response.status_code in [200, 404]


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    def test_health_check_before_api_usage(self):
        """Test that health check works before using API."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200

    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        response = client.get("/api/v1/core/")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_system_info_endpoint(self):
        """Test system info endpoint."""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_api_capabilities_endpoint(self):
        """Test API capabilities endpoint."""
        response = client.get("/api/v1/system/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        capabilities = data["data"]
        assert "endpoints" in capabilities
        assert "features" in capabilities


class TestErrorHandling:
    """Test error handling across all endpoints."""

    def test_invalid_pagination_params(self):
        """Test invalid pagination parameters."""
        response = client.get("/api/v1/studies/?page=0&page_size=0")
        assert response.status_code == 422

    def test_invalid_similarity_threshold(self):
        """Test invalid similarity threshold."""
        response = client.get(
            "/api/v1/similarities/analyze?pmid=12345&model=test&min_similarity=2.0"
        )
        assert response.status_code == 422

    def test_invalid_max_results(self):
        """Test invalid max_results parameter."""
        response = client.get("/api/v1/studies/1/similar?max_results=0")
        assert response.status_code == 422

    def test_large_page_size(self):
        """Test very large page size."""
        response = client.get("/api/v1/studies/?page_size=10000")
        assert response.status_code == 422

    def test_negative_study_id(self):
        """Test negative study ID."""
        response = client.get("/api/v1/studies/-1")
        assert response.status_code == 422


class TestPerformance:
    """Basic performance tests."""

    def test_studies_list_response_time(self):
        """Test studies list response time."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/studies/?page_size=10")
        end_time = time.time()

        assert response.status_code == 200
        # Should respond within 2 seconds
        assert (end_time - start_time) < 2.0

    def test_similarities_stats_response_time(self):
        """Test similarities stats response time."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/similarities/stats")
        end_time = time.time()

        assert response.status_code == 200
        # Should respond within 2 seconds
        assert (end_time - start_time) < 2.0
