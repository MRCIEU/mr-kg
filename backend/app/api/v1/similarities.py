"""Similarities API endpoints for version 1."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.dependencies import (
    get_analytics_service,
    get_similarity_service,
)
from app.models.database import (
    PaginationParams,
    QueryCombination,
    SimilaritySearchResult,
    TraitSimilarity,
    TraitToEFORequest,
    VectorSimilarityRequest,
)
from app.models.responses import (
    DataResponse,
    PaginatedDataResponse,
    PaginationInfo,
)
from app.services.database_service import AnalyticsService, SimilarityService

logger = logging.getLogger(__name__)
router = APIRouter()


# ==== Pydantic Models for Similarities API ====


class SimilarityAnalysis(BaseModel):
    """Comprehensive similarity analysis result."""

    query_combination: QueryCombination
    similarities: list[TraitSimilarity]
    summary: dict[str, Any]


class VectorSimilarityResult(BaseModel):
    """Result from vector similarity search."""

    result_id: str
    result_label: str
    similarity_score: float
    result_type: str  # "trait" or "efo"


class TraitEFOMapping(BaseModel):
    """Mapping between trait and EFO terms."""

    trait_index: int
    trait_label: str
    efo_mappings: list[SimilaritySearchResult]


class SimilarityStatistics(BaseModel):
    """Statistics about similarity computations."""

    total_combinations: int
    total_similarities: int
    average_similarity: float
    similarity_distribution: dict[str, int]
    model_comparison: dict[str, dict[str, Any]]


# ==== Endpoints Implementation ====


@router.get("/analyze", response_model=DataResponse[SimilarityAnalysis])
async def analyze_similarity(
    pmid: str = Query(..., description="PubMed ID"),
    model: str = Query(..., description="Model name"),
    max_results: int = Query(
        default=10, ge=1, le=100, description="Maximum results to return"
    ),
    min_similarity: float = Query(
        default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold"
    ),
    similarity_type: str = Query(
        default="trait_profile",
        description="Similarity type: trait_profile or jaccard",
    ),
    service: SimilarityService = Depends(get_similarity_service),
) -> DataResponse[SimilarityAnalysis]:
    """Analyze similarity for a PMID-model combination.

    Returns comprehensive similarity analysis including:
    - Query combination details
    - Top similar combinations
    - Summary statistics
    """
    try:
        # Find the query combination
        combination = service.similarity_repo.find_combination(pmid, model)
        if not combination:
            raise HTTPException(
                status_code=404,
                detail=f"No combination found for PMID {pmid} and model {model}",
            )
        # Get similarities
        similarities = service.similarity_repo.get_similarities(
            combination.id,
            top_k=max_results,
            min_similarity=min_similarity,
            similarity_type=similarity_type,
        )

        # Calculate summary statistics
        if similarities:
            scores = [
                sim.trait_profile_similarity
                if similarity_type == "trait_profile"
                else sim.trait_jaccard_similarity
                for sim in similarities
            ]
            summary = {
                "total_similar": len(similarities),
                "max_similarity": max(scores),
                "min_similarity": min(scores),
                "avg_similarity": sum(scores) / len(scores),
                "similarity_type": similarity_type,
            }
        else:
            summary = {
                "total_similar": 0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "avg_similarity": 0.0,
                "similarity_type": similarity_type,
            }

        analysis = SimilarityAnalysis(
            query_combination=combination,
            similarities=similarities,
            summary=summary,
        )

        return DataResponse(data=analysis)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing similarity for {pmid}, {model}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze similarity: {str(e)}"
        ) from e


@router.get(
    "/combinations",
    response_model=PaginatedDataResponse[QueryCombination],
)
async def search_combinations(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=50, ge=1, le=200, description="Items per page"
    ),
    model: str | None = Query(default=None, description="Filter by model"),
    min_trait_count: int | None = Query(
        default=None, description="Minimum trait count"
    ),
    pmid_query: str | None = Query(default=None, description="Search PMIDs"),
    title_query: str | None = Query(default=None, description="Search titles"),
    order_by: str = Query(
        default="trait_count",
        description="Sort field: trait_count, pmid, model",
    ),
    order_desc: bool = Query(default=True, description="Sort descending"),
    service: SimilarityService = Depends(get_similarity_service),
) -> PaginatedDataResponse[QueryCombination]:
    """Search available PMID-model combinations for similarity analysis.

    Returns combinations with filtering and pagination support.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)

        # Build filter conditions
        where_conditions = []
        params: list[int | str] = []

        if model:
            where_conditions.append("model = ?")
            params.append(model)

        if min_trait_count is not None:
            where_conditions.append("trait_count >= ?")
            params.append(min_trait_count)

        if pmid_query:
            where_conditions.append("pmid ILIKE ?")
            params.append(f"%{pmid_query}%")

        if title_query:
            where_conditions.append("title ILIKE ?")
            params.append(f"%{title_query}%")

        where_clause = (
            " AND ".join(where_conditions) if where_conditions else "1=1"
        )

        # Validate order_by field
        valid_order_fields = ["trait_count", "pmid", "model", "title"]
        if order_by not in valid_order_fields:
            order_by = "trait_count"

        order_direction = "DESC" if order_desc else "ASC"

        # Get total count
        count_query = (
            f"SELECT COUNT(*) FROM query_combinations WHERE {where_clause}"
        )
        total_count = service.similarity_repo.execute_query(
            count_query, tuple(params)
        )[0][0]

        # Get combinations
        query = f"""
        SELECT id, pmid, model, title, trait_count
        FROM query_combinations
        WHERE {where_clause}
        ORDER BY {order_by} {order_direction}
        LIMIT ? OFFSET ?
        """
        params.extend([pagination.page_size, pagination.offset])

        results = service.similarity_repo.execute_query(query, tuple(params))
        combinations = [
            QueryCombination(
                id=row[0],
                pmid=row[1],
                model=row[2],
                title=row[3],
                trait_count=row[4],
            )
            for row in results
        ]

        return PaginatedDataResponse(
            data=combinations,
            pagination=PaginationInfo.create(
                page=pagination.page,
                page_size=pagination.page_size,
                total_items=total_count,
            ),
        )

    except Exception as e:
        logger.error(f"Error searching combinations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search combinations: {str(e)}"
        ) from e


@router.post(
    "/vector", response_model=DataResponse[list[VectorSimilarityResult]]
)
async def vector_similarity_search(
    request: VectorSimilarityRequest,
    search_type: str = Query(
        default="traits", description="Search type: traits or efo"
    ),
    service: SimilarityService = Depends(get_similarity_service),
) -> DataResponse[list[VectorSimilarityResult]]:
    """Perform vector similarity search using custom query vector.

    Computes cosine similarity between the query vector and stored embeddings.
    Returns top-K most similar items with similarity scores.
    """
    try:
        if len(request.query_vector) != 200:
            raise HTTPException(
                status_code=400, detail="Query vector must be 200-dimensional"
            )
        if search_type not in ["traits", "efo"]:
            raise HTTPException(
                status_code=400, detail="Search type must be 'traits' or 'efo'"
            )
        # Determine which table to search
        table_name = (
            "trait_embeddings" if search_type == "traits" else "efo_embeddings"
        )
        id_column = "trait_index" if search_type == "traits" else "id"
        label_column = "trait_label" if search_type == "traits" else "label"

        # Perform vector similarity search using DuckDB vector operations
        query = f"""
        SELECT
            {id_column},
            {label_column},
            array_cosine_similarity(vector, ?) as similarity
        FROM {table_name}
        WHERE vector IS NOT NULL
        AND array_cosine_similarity(vector, ?) >= ?
        ORDER BY similarity DESC
        LIMIT ?
        """

        params = (
            request.query_vector,
            request.query_vector,
            request.threshold,
            request.top_k,
        )

        results = service.trait_repo.execute_query(query, params)

        similarity_results = [
            VectorSimilarityResult(
                result_id=str(row[0]),
                result_label=row[1],
                similarity_score=float(row[2]),
                result_type=search_type[
                    :-1
                ],  # Remove 's' from 'traits' or 'efo'
            )
            for row in results
        ]

        return DataResponse(data=similarity_results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in vector similarity search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform vector similarity search: {str(e)}",
        ) from e


@router.post(
    "/trait-to-efo", response_model=DataResponse[list[TraitEFOMapping]]
)
async def bulk_trait_to_efo_mapping(
    request: TraitToEFORequest,
    service: SimilarityService = Depends(get_similarity_service),
) -> DataResponse[list[TraitEFOMapping]]:
    """Map multiple traits to EFO terms using vector similarity.

    Returns EFO term mappings for each requested trait based on
    cosine similarity between trait and EFO embeddings.
    """
    try:
        if len(request.trait_indices) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 trait indices allowed per request",
            )
        mappings = []

        for trait_index in request.trait_indices:
            # Get trait information
            trait_query = """
            SELECT trait_index, trait_label, vector
            FROM trait_embeddings
            WHERE trait_index = ?
            """
            trait_result = service.trait_repo.execute_query(
                trait_query, (trait_index,)
            )

            if not trait_result:
                continue  # Skip non-existent traits

            trait_label = trait_result[0][1]
            trait_vector = trait_result[0][2]

            if trait_vector is None:
                # No embedding available for this trait
                mappings.append(
                    TraitEFOMapping(
                        trait_index=trait_index,
                        trait_label=trait_label,
                        efo_mappings=[],
                    )
                )
                continue

            # Find similar EFO terms
            efo_query = """
            SELECT
                id,
                label,
                array_cosine_similarity(vector, ?) as similarity
            FROM efo_embeddings
            WHERE vector IS NOT NULL
            AND array_cosine_similarity(vector, ?) >= ?
            ORDER BY similarity DESC
            LIMIT ?
            """

            efo_params = (
                trait_vector,
                trait_vector,
                request.threshold,
                request.top_k,
            )
            efo_results = service.efo_repo.execute_query(efo_query, efo_params)

            efo_mappings = [
                SimilaritySearchResult(
                    query_id=str(trait_index),
                    query_label=trait_label,
                    result_id=row[0],
                    result_label=row[1],
                    similarity=float(row[2]),
                )
                for row in efo_results
            ]

            mappings.append(
                TraitEFOMapping(
                    trait_index=trait_index,
                    trait_label=trait_label,
                    efo_mappings=efo_mappings,
                )
            )

        return DataResponse(data=mappings)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk trait-to-EFO mapping: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform trait-to-EFO mapping: {str(e)}",
        ) from e


@router.get("/stats", response_model=DataResponse[SimilarityStatistics])
async def similarity_statistics(
    service: AnalyticsService = Depends(get_analytics_service),
) -> DataResponse[SimilarityStatistics]:
    """Get statistics about similarity computations in the database.

    Returns summary statistics including:
    - Total combinations and similarities
    - Similarity score distributions
    - Model comparison metrics
    """
    try:
        # Get basic counts
        total_combinations = service.similarity_repo.get_count(
            "query_combinations"
        )
        total_similarities = service.similarity_repo.get_count(
            "trait_similarities"
        )

        # Get average similarity
        avg_query = """
        SELECT AVG(trait_profile_similarity)
        FROM trait_similarities
        WHERE trait_profile_similarity IS NOT NULL
        """
        avg_result = service.similarity_repo.execute_query(avg_query)
        average_similarity = float(avg_result[0][0] or 0.0)

        # Get similarity distribution
        distribution_query = """
        SELECT
            CASE
                WHEN trait_profile_similarity < 0.3 THEN '0.0-0.3'
                WHEN trait_profile_similarity < 0.5 THEN '0.3-0.5'
                WHEN trait_profile_similarity < 0.7 THEN '0.5-0.7'
                WHEN trait_profile_similarity < 0.9 THEN '0.7-0.9'
                ELSE '0.9-1.0'
            END as range,
            COUNT(*) as count
        FROM trait_similarities
        WHERE trait_profile_similarity IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
        distribution_results = service.similarity_repo.execute_query(
            distribution_query
        )
        similarity_distribution = {
            row[0]: row[1] for row in distribution_results
        }

        # Get model comparison
        model_query = """
        SELECT
            qc.model,
            COUNT(*) as total_combinations,
            AVG(qc.trait_count) as avg_trait_count,
            COUNT(DISTINCT ts.id) as total_similarities
        FROM query_combinations qc
        LEFT JOIN trait_similarities ts ON qc.id = ts.query_combination_id
        GROUP BY qc.model
        ORDER BY total_combinations DESC
        """
        model_results = service.similarity_repo.execute_query(model_query)
        model_comparison = {
            row[0]: {
                "total_combinations": row[1],
                "avg_trait_count": round(float(row[2] or 0.0), 2),
                "total_similarities": row[3],
            }
            for row in model_results
        }

        statistics = SimilarityStatistics(
            total_combinations=total_combinations,
            total_similarities=total_similarities,
            average_similarity=round(average_similarity, 3),
            similarity_distribution=similarity_distribution,
            model_comparison=model_comparison,
        )

        return DataResponse(data=statistics)

    except Exception as e:
        logger.error(f"Error getting similarity statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get similarity statistics: {str(e)}",
        ) from e


@router.get("/models", response_model=DataResponse[list[str]])
async def get_available_similarity_models(
    service: SimilarityService = Depends(get_similarity_service),
) -> DataResponse[list[str]]:
    """Get list of models available in the similarity database.

    Returns all unique model names that have similarity computations.
    """
    try:
        query = "SELECT DISTINCT model FROM query_combinations ORDER BY model"
        results = service.similarity_repo.execute_query(query)
        models = [row[0] for row in results]

        return DataResponse(data=models)

    except Exception as e:
        logger.error(f"Error getting available similarity models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get available models: {str(e)}"
        ) from e


@router.get(
    "/combinations/{combination_id}",
    response_model=DataResponse[QueryCombination],
)
async def get_combination_details(
    combination_id: int,
    service: SimilarityService = Depends(get_similarity_service),
) -> DataResponse[QueryCombination]:
    """Get details about a specific query combination.

    Returns comprehensive information about a PMID-model combination.
    """
    try:
        query = """
        SELECT id, pmid, model, title, trait_count
        FROM query_combinations
        WHERE id = ?
        """
        results = service.similarity_repo.execute_query(
            query, (combination_id,)
        )

        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Combination with ID {combination_id} not found",
            )
        row = results[0]
        combination = QueryCombination(
            id=row[0],
            pmid=row[1],
            model=row[2],
            title=row[3],
            trait_count=row[4],
        )

        return DataResponse(data=combination)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting combination details for ID {combination_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get combination details: {str(e)}",
        ) from e


@router.get(
    "/combinations/{combination_id}/similarities",
    response_model=PaginatedDataResponse[TraitSimilarity],
)
async def get_combination_similarities(
    combination_id: int,
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=50, ge=1, le=200, description="Items per page"
    ),
    min_similarity: float = Query(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity"
    ),
    similarity_type: str = Query(
        default="trait_profile",
        description="Similarity type: trait_profile or jaccard",
    ),
    service: SimilarityService = Depends(get_similarity_service),
) -> PaginatedDataResponse[TraitSimilarity]:
    """Get all similarities for a specific query combination.

    Returns paginated list of similar combinations with filtering options.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)

        # Verify combination exists
        combination_query = "SELECT id FROM query_combinations WHERE id = ?"
        if not service.similarity_repo.execute_query(
            combination_query, (combination_id,)
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Combination with ID {combination_id} not found",
            )
        # Build similarity query
        similarity_column = (
            "trait_profile_similarity"
            if similarity_type == "trait_profile"
            else "trait_jaccard_similarity"
        )

        # Get total count
        count_query = f"""
        SELECT COUNT(*)
        FROM trait_similarities
        WHERE query_combination_id = ? AND {similarity_column} >= ?
        """
        total_count = service.similarity_repo.execute_query(
            count_query, (combination_id, min_similarity)
        )[0][0]

        # Get similarities
        query = f"""
        SELECT
            id, query_combination_id, similar_pmid, similar_model, similar_title,
            trait_profile_similarity, trait_jaccard_similarity,
            query_trait_count, similar_trait_count
        FROM trait_similarities
        WHERE query_combination_id = ? AND {similarity_column} >= ?
        ORDER BY {similarity_column} DESC
        LIMIT ? OFFSET ?
        """

        params = (
            combination_id,
            min_similarity,
            pagination.page_size,
            pagination.offset,
        )
        results = service.similarity_repo.execute_query(query, params)

        similarities = [
            TraitSimilarity(
                id=row[0],
                query_combination_id=row[1],
                similar_pmid=row[2],
                similar_model=row[3],
                similar_title=row[4],
                trait_profile_similarity=row[5],
                trait_jaccard_similarity=row[6],
                query_trait_count=row[7],
                similar_trait_count=row[8],
            )
            for row in results
        ]

        return PaginatedDataResponse(
            data=similarities,
            pagination=PaginationInfo.create(
                page=pagination.page,
                page_size=pagination.page_size,
                total_items=total_count,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting similarities for combination {combination_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get combination similarities: {str(e)}",
        ) from e
