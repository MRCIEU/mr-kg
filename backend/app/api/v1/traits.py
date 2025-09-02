"""Traits API endpoints for version 1."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.dependencies import get_database_service
from app.models.database import (
    PaginationParams,
    SimilaritySearchFilters,
    SimilaritySearchResult,
    TraitEmbedding,
)
from app.models.responses import DataResponse, PaginatedDataResponse
from app.services.database_service import AnalyticsService, TraitService

logger = logging.getLogger(__name__)
router = APIRouter()


# ==== Basic Pydantic Models for Traits API ====


class TraitStatsResponse(DataResponse[list[dict[str, Any]]]):
    """Response for trait statistics."""

    pass


class TraitListItem(BaseModel):
    """Individual trait item in listing."""

    trait_index: int
    trait_label: str
    appearance_count: int


class TraitListResponse(PaginatedDataResponse[list[TraitListItem]]):
    """Response for paginated trait listing."""

    pass


class TraitSearchResponse(PaginatedDataResponse[list[TraitListItem]]):
    """Response for trait search operations."""

    pass


class TraitDetailExtended(BaseModel):
    """Extended trait details with statistics."""

    trait: TraitEmbedding
    statistics: dict[str, Any]
    studies: list[dict[str, Any]]
    similar_traits: list[SimilaritySearchResult]
    efo_mappings: list[SimilaritySearchResult]


# ==== Endpoints Implementation ====


@router.get("/", response_model=TraitListResponse)
async def list_traits(
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=50, ge=1, le=1000, description="Items per page"
    ),
    order_by: str = Query(
        default="appearance_count",
        description="Sort field: appearance_count, trait_label",
    ),
    order_desc: bool = Query(default=True, description="Sort descending"),
    min_appearances: int | None = Query(
        default=None, description="Minimum appearance count"
    ),
    service: TraitService = Depends(get_database_service),
) -> TraitListResponse:
    """Get paginated list of traits with appearance counts.

    Returns traits from the trait_stats view, sorted by appearance count by default.
    Supports filtering by minimum appearance count and custom sorting.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)

        # Build query based on parameters
        where_conditions = []
        params: list[int | str] = []

        if min_appearances is not None:
            where_conditions.append("appearance_count >= ?")
            params.append(min_appearances)

        where_clause = (
            " AND ".join(where_conditions) if where_conditions else "1=1"
        )

        # Validate order_by field
        valid_order_fields = ["appearance_count", "trait_label", "trait_index"]
        if order_by not in valid_order_fields:
            order_by = "appearance_count"

        order_direction = "DESC" if order_desc else "ASC"

        # Get total count
        count_query = f"SELECT COUNT(*) FROM trait_stats WHERE {where_clause}"
        total_count = service.trait_repo.execute_query(
            count_query, tuple(params)
        )[0][0]

        # Get traits
        query = f"""
        SELECT trait_index, trait_label, appearance_count
        FROM trait_stats
        WHERE {where_clause}
        ORDER BY {order_by} {order_direction}
        LIMIT ? OFFSET ?
        """
        params.extend([pagination.page_size, pagination.offset])

        results = service.trait_repo.execute_query(query, tuple(params))
        traits = [
            TraitListItem(
                trait_index=row[0], trait_label=row[1], appearance_count=row[2]
            )
            for row in results
        ]

        return TraitListResponse(
            data=traits,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=(total_count + pagination.page_size - 1)
            // pagination.page_size,
            has_next=pagination.page * pagination.page_size < total_count,
            has_previous=pagination.page > 1,
        )

    except Exception as e:
        logger.error(f"Error listing traits: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list traits: {str(e)}"
        )


@router.get("/search", response_model=TraitSearchResponse)
async def search_traits(
    q: str = Query(..., description="Search query for trait labels"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=50, ge=1, le=500, description="Items per page"
    ),
    min_appearances: int | None = Query(
        default=None, description="Minimum appearance count"
    ),
    service: TraitService = Depends(get_database_service),
) -> TraitSearchResponse:
    """Search traits by label with fuzzy matching.

    Performs case-insensitive search using ILIKE pattern matching.
    Results are ordered by appearance count descending.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)

        # Build where conditions
        where_conditions = ["trait_label ILIKE ?"]
        params: list[int | str] = [f"%{q}%"]

        if min_appearances is not None:
            where_conditions.append("appearance_count >= ?")
            params.append(min_appearances)

        where_clause = " AND ".join(where_conditions)

        # Get total count
        count_query = f"SELECT COUNT(*) FROM trait_stats WHERE {where_clause}"
        total_count = service.trait_repo.execute_query(
            count_query, tuple(params)
        )[0][0]

        # Get search results
        query = f"""
        SELECT trait_index, trait_label, appearance_count
        FROM trait_stats
        WHERE {where_clause}
        ORDER BY appearance_count DESC, trait_label ASC
        LIMIT ? OFFSET ?
        """
        params.extend([pagination.page_size, pagination.offset])

        results = service.trait_repo.execute_query(query, tuple(params))
        traits = [
            TraitListItem(
                trait_index=row[0], trait_label=row[1], appearance_count=row[2]
            )
            for row in results
        ]

        return TraitSearchResponse(
            data=traits,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=(total_count + pagination.page_size - 1)
            // pagination.page_size,
            has_next=pagination.page * pagination.page_size < total_count,
            has_previous=pagination.page > 1,
        )

    except Exception as e:
        logger.error(f"Error searching traits with query '{q}': {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search traits: {str(e)}"
        )


@router.get("/{trait_index}", response_model=DataResponse[TraitDetailExtended])
async def get_trait_details(
    trait_index: int,
    include_studies: bool = Query(
        default=True, description="Include associated studies"
    ),
    include_similar: bool = Query(
        default=True, description="Include similar traits"
    ),
    include_efo: bool = Query(default=True, description="Include EFO mappings"),
    max_studies: int = Query(
        default=50, ge=1, le=200, description="Maximum studies to return"
    ),
    max_similar: int = Query(
        default=10, ge=1, le=50, description="Maximum similar traits"
    ),
    similarity_threshold: float = Query(
        default=0.3, ge=0.0, le=1.0, description="Similarity threshold"
    ),
    service: TraitService = Depends(get_database_service),
    analytics_service: AnalyticsService = Depends(get_database_service),
) -> DataResponse[TraitDetailExtended]:
    """Get detailed information about a specific trait.

    Returns comprehensive trait information including:
    - Basic trait data (index, label, embedding)
    - Statistics (study count, model distribution, publication years)
    - Associated studies (optional)
    - Similar traits (optional)
    - EFO term mappings (optional)
    """
    try:
        # Get basic trait info
        trait = service.trait_repo.get_trait_by_index(trait_index)
        if not trait:
            raise HTTPException(
                status_code=404,
                detail=f"Trait with index {trait_index} not found",
            )

        # Get trait statistics
        statistics = await analytics_service.get_trait_statistics(trait_index)

        # Initialize optional data
        studies = []
        similar_traits = []
        efo_mappings = []

        if include_studies:
            # Get associated studies
            study_query = """
            SELECT DISTINCT
                mr.id, mr.model, mr.pmid, mr.metadata, mr.results,
                mpd.title, mpd.journal, mpd.pub_date
            FROM model_results mr
            JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
            LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
            WHERE mrt.trait_index = ?
            ORDER BY mpd.pub_date DESC, mr.pmid, mr.model
            LIMIT ?
            """
            study_results = service.study_repo.execute_query(
                study_query, (trait_index, max_studies)
            )
            studies = [
                {
                    "id": row[0],
                    "model": row[1],
                    "pmid": row[2],
                    "metadata": row[3],
                    "results": row[4],
                    "title": row[5],
                    "journal": row[6],
                    "pub_date": row[7],
                }
                for row in study_results
            ]

        if include_similar:
            # Get similar traits using vector similarity
            similar_traits = service.trait_repo.find_similar_traits(
                trait_index, top_k=max_similar, threshold=similarity_threshold
            )

        if include_efo:
            # Get EFO mappings
            efo_mappings = service.efo_repo.find_trait_efo_mappings(
                trait_index, top_k=max_similar, threshold=similarity_threshold
            )

        trait_detail = TraitDetailExtended(
            trait=trait,
            statistics=statistics,
            studies=studies,
            similar_traits=similar_traits,
            efo_mappings=efo_mappings,
        )

        return DataResponse(data=trait_detail)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting trait details for index {trait_index}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get trait details: {str(e)}"
        )


@router.get(
    "/{trait_index}/studies",
    response_model=PaginatedDataResponse[list[dict[str, Any]]],
)
async def get_trait_studies(
    trait_index: int,
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(
        default=50, ge=1, le=200, description="Items per page"
    ),
    model: str | None = Query(default=None, description="Filter by model"),
    journal: str | None = Query(default=None, description="Filter by journal"),
    date_from: str | None = Query(
        default=None, description="Publication date from (YYYY-MM-DD)"
    ),
    date_to: str | None = Query(
        default=None, description="Publication date to (YYYY-MM-DD)"
    ),
    service: TraitService = Depends(get_database_service),
) -> PaginatedDataResponse[list[dict[str, Any]]]:
    """Get studies associated with a specific trait with filtering and pagination.

    Returns detailed study information for studies that mention the specified trait.
    Supports filtering by model, journal, and publication date range.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)

        # Verify trait exists
        trait = service.trait_repo.get_trait_by_index(trait_index)
        if not trait:
            raise HTTPException(
                status_code=404,
                detail=f"Trait with index {trait_index} not found",
            )

        # Build filter conditions
        where_conditions = ["mrt.trait_index = ?"]
        params: list[int | str] = [trait_index]

        if model:
            where_conditions.append("mr.model = ?")
            params.append(model)

        if journal:
            where_conditions.append("mpd.journal ILIKE ?")
            params.append(f"%{journal}%")

        if date_from:
            where_conditions.append("mpd.pub_date >= ?")
            params.append(date_from)

        if date_to:
            where_conditions.append("mpd.pub_date <= ?")
            params.append(date_to)

        where_clause = " AND ".join(where_conditions)

        # Get total count
        count_query = f"""
        SELECT COUNT(DISTINCT mr.id)
        FROM model_results mr
        JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        WHERE {where_clause}
        """
        total_count = service.study_repo.execute_query(
            count_query, tuple(params)
        )[0][0]

        # Get studies
        query = f"""
        SELECT DISTINCT
            mr.id, mr.model, mr.pmid, mr.metadata, mr.results,
            mpd.title, mpd.abstract, mpd.journal, mpd.pub_date, mpd.author_affil
        FROM model_results mr
        JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        WHERE {where_clause}
        ORDER BY mpd.pub_date DESC, mr.pmid, mr.model
        LIMIT ? OFFSET ?
        """
        params.extend([pagination.page_size, pagination.offset])

        results = service.study_repo.execute_query(query, tuple(params))
        studies = [
            {
                "id": row[0],
                "model": row[1],
                "pmid": row[2],
                "metadata": row[3],
                "results": row[4],
                "title": row[5],
                "abstract": row[6],
                "journal": row[7],
                "pub_date": row[8],
                "author_affil": row[9],
            }
            for row in results
        ]

        return PaginatedDataResponse(
            data=studies,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=(total_count + pagination.page_size - 1)
            // pagination.page_size,
            has_next=pagination.page * pagination.page_size < total_count,
            has_previous=pagination.page > 1,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting studies for trait {trait_index}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get trait studies: {str(e)}"
        )


@router.get(
    "/{trait_index}/similar",
    response_model=DataResponse[list[SimilaritySearchResult]],
)
async def get_similar_traits(
    trait_index: int,
    max_results: int = Query(
        default=10, ge=1, le=100, description="Maximum results to return"
    ),
    similarity_threshold: float = Query(
        default=0.3, ge=0.0, le=1.0, description="Minimum similarity"
    ),
    service: TraitService = Depends(get_database_service),
) -> DataResponse[list[SimilaritySearchResult]]:
    """Find traits similar to the specified trait using vector embeddings.

    Uses cosine similarity between trait embedding vectors to find semantically related traits.
    Results are sorted by similarity score in descending order.
    """
    try:
        # Verify trait exists
        trait = service.trait_repo.get_trait_by_index(trait_index)
        if not trait:
            raise HTTPException(
                status_code=404,
                detail=f"Trait with index {trait_index} not found",
            )

        # Find similar traits
        filters = SimilaritySearchFilters(
            max_results=max_results, min_similarity=similarity_threshold
        )

        similar_traits = await service.find_similar_traits(trait_index, filters)

        return DataResponse(data=similar_traits)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar traits for {trait_index}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to find similar traits: {str(e)}"
        )


@router.get(
    "/{trait_index}/efo-mappings",
    response_model=DataResponse[list[SimilaritySearchResult]],
)
async def get_trait_efo_mappings(
    trait_index: int,
    max_results: int = Query(
        default=10, ge=1, le=50, description="Maximum mappings to return"
    ),
    similarity_threshold: float = Query(
        default=0.3, ge=0.0, le=1.0, description="Minimum similarity"
    ),
    service: TraitService = Depends(get_database_service),
) -> DataResponse[list[SimilaritySearchResult]]:
    """Get EFO (Experimental Factor Ontology) term mappings for a trait.

    Uses cosine similarity between trait and EFO term embeddings to find
    relevant ontology terms that best represent the trait.
    """
    try:
        # Verify trait exists
        trait = service.trait_repo.get_trait_by_index(trait_index)
        if not trait:
            raise HTTPException(
                status_code=404,
                detail=f"Trait with index {trait_index} not found",
            )

        # Get EFO mappings
        efo_mappings = service.efo_repo.find_trait_efo_mappings(
            trait_index, top_k=max_results, threshold=similarity_threshold
        )

        return DataResponse(data=efo_mappings)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting EFO mappings for trait {trait_index}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get EFO mappings: {str(e)}"
        )


@router.get("/stats/overview", response_model=DataResponse[dict[str, Any]])
async def get_traits_overview(
    service: AnalyticsService = Depends(get_database_service),
) -> DataResponse[dict[str, Any]]:
    """Get overview statistics about traits in the database.

    Returns summary statistics including:
    - Total trait count
    - Top traits by appearance count
    - Distribution by appearance frequency
    - Model coverage statistics
    """
    try:
        # Get basic counts
        total_traits = service.trait_repo.get_count("trait_stats")
        total_appearances = service.trait_repo.execute_query(
            "SELECT SUM(appearance_count) FROM trait_stats"
        )[0][0]

        # Get top traits
        top_traits_query = """
        SELECT trait_index, trait_label, appearance_count
        FROM trait_stats
        ORDER BY appearance_count DESC
        LIMIT 20
        """
        top_traits_results = service.trait_repo.execute_query(top_traits_query)
        top_traits = [
            {
                "trait_index": row[0],
                "trait_label": row[1],
                "appearance_count": row[2],
            }
            for row in top_traits_results
        ]

        # Get appearance distribution
        distribution_query = """
        SELECT
            CASE
                WHEN appearance_count = 1 THEN '1'
                WHEN appearance_count BETWEEN 2 AND 5 THEN '2-5'
                WHEN appearance_count BETWEEN 6 AND 10 THEN '6-10'
                WHEN appearance_count BETWEEN 11 AND 25 THEN '11-25'
                WHEN appearance_count BETWEEN 26 AND 50 THEN '26-50'
                WHEN appearance_count BETWEEN 51 AND 100 THEN '51-100'
                ELSE '100+'
            END as range,
            COUNT(*) as trait_count
        FROM trait_stats
        GROUP BY 1
        ORDER BY 1
        """
        distribution_results = service.trait_repo.execute_query(
            distribution_query
        )
        appearance_distribution = {
            row[0]: row[1] for row in distribution_results
        }

        # Get model coverage stats
        model_coverage_query = """
        SELECT
            mr.model,
            COUNT(DISTINCT mrt.trait_index) as unique_traits,
            COUNT(*) as total_mentions
        FROM model_results mr
        JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
        GROUP BY mr.model
        ORDER BY unique_traits DESC
        """
        model_coverage_results = service.trait_repo.execute_query(
            model_coverage_query
        )
        model_coverage = [
            {"model": row[0], "unique_traits": row[1], "total_mentions": row[2]}
            for row in model_coverage_results
        ]

        overview = {
            "total_traits": total_traits,
            "total_appearances": total_appearances,
            "average_appearances": round(total_appearances / total_traits, 2)
            if total_traits > 0
            else 0,
            "top_traits": top_traits,
            "appearance_distribution": appearance_distribution,
            "model_coverage": model_coverage,
        }

        return DataResponse(data=overview)

    except Exception as e:
        logger.error(f"Error getting traits overview: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get traits overview: {str(e)}"
        )


@router.post("/bulk", response_model=DataResponse[list[TraitEmbedding]])
async def get_traits_bulk(
    trait_indices: list[int],
    service: TraitService = Depends(get_database_service),
) -> DataResponse[list[TraitEmbedding]]:
    """Get multiple traits by their indices in a single request.

    Useful for bulk operations and reducing API calls.
    Returns traits in the same order as requested indices.
    """
    try:
        if len(trait_indices) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Maximum 1000 trait indices allowed per request",
            )

        if not trait_indices:
            return DataResponse(data=[])

        # Get traits
        placeholders = ",".join("?" * len(trait_indices))
        query = f"""
        SELECT trait_index, trait_label, vector
        FROM trait_embeddings
        WHERE trait_index IN ({placeholders})
        """

        results = service.trait_repo.execute_query(query, tuple(trait_indices))

        # Create lookup dict for ordering
        traits_dict = {
            row[0]: TraitEmbedding(
                trait_index=row[0],
                trait_label=row[1],
                vector=row[2] if row[2] else None,
            )
            for row in results
        }

        # Return in requested order, None for missing traits
        traits = [traits_dict.get(idx) for idx in trait_indices]
        traits = [
            trait for trait in traits if trait is not None
        ]  # Filter out None values

        return DataResponse(data=traits)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting bulk traits: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get bulk traits: {str(e)}"
        )

