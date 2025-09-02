"""Pydantic models for API responses based on database schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ==== Base Models ====


class TraitEmbedding(BaseModel):
    """Trait with embedding vector representation."""

    trait_index: int = Field(
        ..., description="Canonical trait index from unique_traits.csv"
    )
    trait_label: str = Field(..., description="Human-readable trait name")
    vector: list[float] | None = Field(
        None, description="200-dimensional embedding vector"
    )


class EFOEmbedding(BaseModel):
    """EFO (Experimental Factor Ontology) term with embedding."""

    id: str = Field(..., description="EFO ontology identifier")
    label: str = Field(..., description="Human-readable EFO term label")
    vector: list[float] | None = Field(
        None, description="200-dimensional embedding vector"
    )


class ModelResult(BaseModel):
    """Model analysis result for a research paper."""

    id: int = Field(..., description="Unique result identifier")
    model: str = Field(..., description="Name of the LLM model")
    pmid: str = Field(..., description="PubMed ID of the research paper")
    metadata: dict[str, Any] = Field(
        ..., description="Structured metadata including exposures/outcomes"
    )
    results: dict[str, Any] = Field(
        ..., description="Complete raw model output"
    )


class ModelResultTrait(BaseModel):
    """Link between model result and trait."""

    id: int = Field(..., description="Unique trait link identifier")
    model_result_id: int = Field(..., description="Foreign key to model result")
    trait_index: int = Field(..., description="Foreign key to trait embedding")
    trait_label: str = Field(..., description="Denormalized trait label")
    trait_id_in_result: str | None = Field(
        None, description="Original trait ID from model output"
    )


class MRPubmedData(BaseModel):
    """PubMed metadata for research papers."""

    pmid: str = Field(..., description="PubMed ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    pub_date: str = Field(..., description="Publication date")
    journal: str = Field(..., description="Journal name")
    journal_issn: str | None = Field(None, description="Journal ISSN")
    author_affil: str | None = Field(None, description="Author affiliation")


# ==== Trait Profile Models ====


class QueryCombination(BaseModel):
    """PMID-model combination with trait profile metadata."""

    id: int = Field(..., description="Unique combination identifier")
    pmid: str = Field(..., description="PubMed ID")
    model: str = Field(..., description="LLM model name")
    title: str = Field(..., description="Paper title")
    trait_count: int = Field(..., description="Number of traits identified")


class TraitSimilarity(BaseModel):
    """Similarity relationship between PMID-model combinations."""

    id: int = Field(..., description="Unique similarity identifier")
    query_combination_id: int = Field(
        ..., description="Foreign key to query combination"
    )
    similar_pmid: str = Field(..., description="PubMed ID of similar paper")
    similar_model: str = Field(
        ..., description="Model name for similar combination"
    )
    similar_title: str = Field(..., description="Title of similar paper")
    trait_profile_similarity: float = Field(
        ..., description="Semantic similarity score (0.0-1.0)"
    )
    trait_jaccard_similarity: float = Field(
        ..., description="Jaccard similarity coefficient (0.0-1.0)"
    )
    query_trait_count: int = Field(
        ..., description="Number of traits in query combination"
    )
    similar_trait_count: int = Field(
        ..., description="Number of traits in similar combination"
    )


# ==== API Response Models ====


class SimilaritySearchResult(BaseModel):
    """Result from similarity search between traits or papers."""

    query_id: str = Field(..., description="Query identifier")
    query_label: str = Field(..., description="Query label")
    result_id: str = Field(..., description="Result identifier")
    result_label: str = Field(..., description="Result label")
    similarity: float = Field(..., description="Similarity score (0.0-1.0)")


class TraitSearchResponse(BaseModel):
    """Response for trait search operations."""

    traits: list[TraitEmbedding] = Field(
        ..., description="List of matching traits"
    )
    total_count: int = Field(..., description="Total number of matching traits")
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Number of items per page")


class StudySearchResponse(BaseModel):
    """Response for study search operations."""

    studies: list[ModelResult] = Field(
        ..., description="List of matching studies"
    )
    total_count: int = Field(
        ..., description="Total number of matching studies"
    )
    page: int = Field(default=1, description="Current page number")
    page_size: int = Field(default=50, description="Number of items per page")


class StudyDetailResponse(BaseModel):
    """Detailed information about a specific study."""

    study: ModelResult = Field(..., description="Study details")
    pubmed_data: MRPubmedData | None = Field(
        None, description="PubMed metadata"
    )
    traits: list[ModelResultTrait] = Field(..., description="Associated traits")
    similar_studies: list[TraitSimilarity] = Field(
        ..., description="Similar studies"
    )


class TraitDetailResponse(BaseModel):
    """Detailed information about a specific trait."""

    trait: TraitEmbedding = Field(..., description="Trait details")
    studies: list[ModelResult] = Field(
        ..., description="Studies mentioning this trait"
    )
    similar_traits: list[SimilaritySearchResult] = Field(
        ..., description="Similar traits"
    )
    efo_mappings: list[SimilaritySearchResult] = Field(
        ..., description="EFO term mappings"
    )


class SimilarityAnalysisResponse(BaseModel):
    """Response for similarity analysis operations."""

    query_combination: QueryCombination = Field(
        ..., description="Query combination details"
    )
    similarities: list[TraitSimilarity] = Field(
        ..., description="Top similarity results"
    )
    total_count: int = Field(
        ..., description="Total number of similarities found"
    )


# ==== Health Check Models ====


class DatabaseStatus(BaseModel):
    """Database health status information."""

    database_path: str = Field(..., description="Path to database file")
    accessible: bool = Field(..., description="Whether database is accessible")
    table_count: int = Field(..., description="Number of tables found")
    view_count: int = Field(..., description="Number of views found")
    index_count: int = Field(..., description="Number of indexes found")
    last_checked: datetime = Field(
        ..., description="Timestamp of last health check"
    )
    error: str | None = Field(None, description="Error message if unhealthy")


class HealthCheckResponse(BaseModel):
    """Overall system health check response."""

    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    vector_store: DatabaseStatus = Field(
        ..., description="Vector store database status"
    )
    trait_profile: DatabaseStatus = Field(
        ..., description="Trait profile database status"
    )
    performance_metrics: dict[str, float] = Field(
        ..., description="Performance metrics"
    )


# ==== Filter and Query Models ====


class TraitSearchFilters(BaseModel):
    """Filters for trait search operations."""

    query: str | None = Field(None, description="Text search query")
    models: list[str] | None = Field(None, description="Filter by model names")
    min_study_count: int | None = Field(
        None, description="Minimum number of studies"
    )
    similarity_threshold: float | None = Field(
        None, description="Minimum similarity threshold"
    )


class StudySearchFilters(BaseModel):
    """Filters for study search operations."""

    query: str | None = Field(None, description="Text search query")
    models: list[str] | None = Field(None, description="Filter by model names")
    journals: list[str] | None = Field(None, description="Filter by journals")
    date_from: str | None = Field(
        None, description="Filter by publication date (from)"
    )
    date_to: str | None = Field(
        None, description="Filter by publication date (to)"
    )
    trait_indices: list[int] | None = Field(
        None, description="Filter by trait indices"
    )


class SimilaritySearchFilters(BaseModel):
    """Filters for similarity search operations."""

    model: str | None = Field(None, description="Filter by model name")
    min_similarity: float | None = Field(
        0.5, description="Minimum similarity threshold"
    )
    max_results: int | None = Field(10, description="Maximum number of results")
    similarity_type: str | None = Field(
        "trait_profile",
        description="Type of similarity (trait_profile or jaccard)",
    )


# ==== Request Models ====


class VectorSimilarityRequest(BaseModel):
    """Request for vector similarity search."""

    query_vector: list[float] = Field(
        ..., description="Query vector for similarity search"
    )
    top_k: int = Field(
        default=10, description="Number of top results to return"
    )
    threshold: float = Field(
        default=0.0, description="Minimum similarity threshold"
    )


class TraitToEFORequest(BaseModel):
    """Request for trait to EFO mapping."""

    trait_indices: list[int] = Field(
        ..., description="List of trait indices to map"
    )
    top_k: int = Field(
        default=5, description="Number of top EFO mappings per trait"
    )
    threshold: float = Field(
        default=0.3, description="Minimum similarity threshold"
    )


# ==== Pagination Models ====


class PaginationParams(BaseModel):
    """Pagination parameters for API responses."""

    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default=50, ge=1, le=1000, description="Number of items per page"
    )

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel):
    """Base class for paginated API responses."""

    total_count: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(
        ..., description="Whether there is a previous page"
    )

    @classmethod
    def create(
        cls, items: list[Any], total_count: int, pagination: PaginationParams
    ) -> "PaginatedResponse":
        """Create paginated response from items and pagination params."""
        total_pages = (
            total_count + pagination.page_size - 1
        ) // pagination.page_size

        return cls(
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=total_pages,
            has_next=pagination.page < total_pages,
            has_previous=pagination.page > 1,
        )
