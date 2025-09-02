"""Studies API endpoints for version 1."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.dependencies import get_database_service
from app.models.database import (
    ModelResult,
    ModelResultTrait,
    MRPubmedData,
    PaginationParams,
    SimilaritySearchResult,
)
from app.models.responses import DataResponse, PaginatedDataResponse
from app.services.database_service import AnalyticsService, StudyService

logger = logging.getLogger(__name__)
router = APIRouter()


# ==== Pydantic Models for Studies API ====


class StudyListItem(BaseModel):
    """Individual study item in listing."""

    id: int
    model: str
    pmid: str
    title: str | None = None
    journal: str | None = None
    pub_date: str | None = None
    trait_count: int


class StudyListResponse(PaginatedDataResponse[list[StudyListItem]]):
    """Response for paginated study listing."""

    pass


class StudySearchResponse(PaginatedDataResponse[list[StudyListItem]]):
    """Response for study search operations."""

    pass


class StudyDetailExtended(BaseModel):
    """Extended study details with related information."""

    study: ModelResult
    pubmed_data: MRPubmedData | None = None
    traits: list[ModelResultTrait]
    similar_studies: list[SimilaritySearchResult]
    statistics: dict[str, Any]


class StudyAnalytics(BaseModel):
    """Analytics data for studies."""

    total_studies: int
    total_pmids: int
    model_distribution: dict[str, int]
    journal_distribution: dict[str, int]
    year_distribution: dict[str, int]
    trait_count_distribution: dict[str, int]


# ==== Endpoints Implementation ====


@router.get("/", response_model=StudyListResponse)
async def list_studies(
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
    min_trait_count: int | None = Query(
        default=None, description="Minimum trait count"
    ),
    order_by: str = Query(
        default="pub_date",
        description="Sort field: pub_date, trait_count, pmid",
    ),
    order_desc: bool = Query(default=True, description="Sort descending"),
    service: StudyService = Depends(get_database_service),
) -> StudyListResponse:
    """Get paginated list of studies with filtering and sorting.

    Returns studies with basic metadata and trait counts.
    Supports filtering by model, journal, date range, and trait count.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)

        # Build filter conditions
        where_conditions = []
        params: list[int | str] = []

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

        where_clause = (
            " AND ".join(where_conditions) if where_conditions else "1=1"
        )

        # Validate order_by field
        valid_order_fields = [
            "pub_date",
            "trait_count",
            "pmid",
            "model",
            "journal",
        ]
        if order_by not in valid_order_fields:
            order_by = "pub_date"

        order_direction = "DESC" if order_desc else "ASC"

        # Get total count
        count_query = f"""
        SELECT COUNT(*)
        FROM model_results mr
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        LEFT JOIN (
            SELECT model_result_id, COUNT(*) as trait_count
            FROM model_result_traits
            GROUP BY model_result_id
        ) tc ON mr.id = tc.model_result_id
        WHERE {where_clause}
        """
        if min_trait_count is not None:
            count_query += " AND COALESCE(tc.trait_count, 0) >= ?"
            params.append(min_trait_count)

        total_count = service.study_repo.execute_query(
            count_query, tuple(params)
        )[0][0]

        # Get studies
        # Handle trait_count ordering specially since it requires the subquery
        if order_by == "trait_count":
            order_clause = f"COALESCE(tc.trait_count, 0) {order_direction}"
        else:
            order_clause = (
                f"mpd.{order_by} {order_direction}"
                if order_by in ["pub_date", "journal"]
                else f"mr.{order_by} {order_direction}"
            )

        query = f"""
        SELECT
            mr.id, mr.model, mr.pmid,
            mpd.title, mpd.journal, mpd.pub_date,
            COALESCE(tc.trait_count, 0) as trait_count
        FROM model_results mr
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        LEFT JOIN (
            SELECT model_result_id, COUNT(*) as trait_count
            FROM model_result_traits
            GROUP BY model_result_id
        ) tc ON mr.id = tc.model_result_id
        WHERE {where_clause}
        """

        if min_trait_count is not None:
            query += " AND COALESCE(tc.trait_count, 0) >= ?"
            params.append(min_trait_count)

        query += f" ORDER BY {order_clause} LIMIT ? OFFSET ?"
        params.extend([pagination.page_size, pagination.offset])

        results = service.study_repo.execute_query(query, tuple(params))
        studies = [
            StudyListItem(
                id=row[0],
                model=row[1],
                pmid=row[2],
                title=row[3],
                journal=row[4],
                pub_date=row[5],
                trait_count=row[6] or 0,
            )
            for row in results
        ]

        return StudyListResponse(
            data=studies,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=(total_count + pagination.page_size - 1)
            // pagination.page_size,
            has_next=pagination.page * pagination.page_size < total_count,
            has_previous=pagination.page > 1,
        )

    except Exception as e:
        logger.error(f"Error listing studies: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list studies: {str(e)}"
        )


@router.get("/search", response_model=StudySearchResponse)
async def search_studies(
    q: str = Query(
        ..., description="Search query for PMID, title, or abstract"
    ),
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
    service: StudyService = Depends(get_database_service),
) -> StudySearchResponse:
    """Search studies by PMID, title, or abstract with fuzzy matching.

    Performs case-insensitive search across multiple fields.
    Results are ordered by relevance and publication date.
    """
    try:
        pagination = PaginationParams(page=page, page_size=page_size)

        # Build search conditions
        search_conditions = [
            "mr.pmid ILIKE ?",
            "mpd.title ILIKE ?",
            "mpd.abstract ILIKE ?",
        ]
        search_pattern = f"%{q}%"

        where_conditions = [f"({' OR '.join(search_conditions)})"]
        params: list[int | str] = [
            search_pattern,
            search_pattern,
            search_pattern,
        ]

        # Add additional filters
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
        SELECT COUNT(*)
        FROM model_results mr
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        LEFT JOIN (
            SELECT model_result_id, COUNT(*) as trait_count
            FROM model_result_traits
            GROUP BY model_result_id
        ) tc ON mr.id = tc.model_result_id
        WHERE {where_clause}
        """
        total_count = service.study_repo.execute_query(
            count_query, tuple(params)
        )[0][0]

        # Get search results
        query = f"""
        SELECT
            mr.id, mr.model, mr.pmid,
            mpd.title, mpd.journal, mpd.pub_date,
            COALESCE(tc.trait_count, 0) as trait_count
        FROM model_results mr
        LEFT JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        LEFT JOIN (
            SELECT model_result_id, COUNT(*) as trait_count
            FROM model_result_traits
            GROUP BY model_result_id
        ) tc ON mr.id = tc.model_result_id
        WHERE {where_clause}
        ORDER BY mpd.pub_date DESC, mr.pmid, mr.model
        LIMIT ? OFFSET ?
        """
        params.extend([pagination.page_size, pagination.offset])

        results = service.study_repo.execute_query(query, tuple(params))
        studies = [
            StudyListItem(
                id=row[0],
                model=row[1],
                pmid=row[2],
                title=row[3],
                journal=row[4],
                pub_date=row[5],
                trait_count=row[6] or 0,
            )
            for row in results
        ]

        return StudySearchResponse(
            data=studies,
            total_count=total_count,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=(total_count + pagination.page_size - 1)
            // pagination.page_size,
            has_next=pagination.page * pagination.page_size < total_count,
            has_previous=pagination.page > 1,
        )

    except Exception as e:
        logger.error(f"Error searching studies with query '{q}': {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search studies: {str(e)}"
        )


@router.get("/{study_id}", response_model=DataResponse[StudyDetailExtended])
async def get_study_details(
    study_id: int,
    include_similar: bool = Query(
        default=True, description="Include similar studies"
    ),
    include_traits: bool = Query(
        default=True, description="Include associated traits"
    ),
    max_similar: int = Query(
        default=10, ge=1, le=50, description="Maximum similar studies"
    ),
    similarity_threshold: float = Query(
        default=0.3, ge=0.0, le=1.0, description="Similarity threshold"
    ),
    service: StudyService = Depends(get_database_service),
    analytics_service: AnalyticsService = Depends(get_database_service),
) -> DataResponse[StudyDetailExtended]:
    """Get detailed information about a specific study.

    Returns comprehensive study information including:
    - Basic study data (model result, metadata)
    - PubMed metadata (title, abstract, journal, etc.)
    - Associated traits (optional)
    - Similar studies (optional)
    - Study statistics
    """
    try:
        # Get study details
        study_detail = await service.get_study_details(study_id)
        if not study_detail:
            raise HTTPException(
                status_code=404, detail=f"Study with ID {study_id} not found"
            )

        # Get study statistics
        stats_query = """
        SELECT
            COUNT(mrt.trait_index) as trait_count,
            COUNT(DISTINCT mrt.trait_index) as unique_trait_count
        FROM model_result_traits mrt
        WHERE mrt.model_result_id = ?
        """
        stats_result = service.study_repo.execute_query(
            stats_query, (study_id,)
        )
        statistics = {
            "trait_count": stats_result[0][0] if stats_result else 0,
            "unique_trait_count": stats_result[0][1] if stats_result else 0,
        }

        # Initialize optional data
        traits = study_detail.traits if include_traits else []
        similar_studies = (
            study_detail.similar_studies if include_similar else []
        )

        study_extended = StudyDetailExtended(
            study=study_detail.study,
            pubmed_data=study_detail.pubmed_data,
            traits=traits,
            similar_studies=similar_studies,
            statistics=statistics,
        )

        return DataResponse(data=study_extended)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting study details for ID {study_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get study details: {str(e)}"
        )


@router.get("/pmid/{pmid}", response_model=DataResponse[list[ModelResult]])
async def get_studies_by_pmid(
    pmid: str,
    service: StudyService = Depends(get_database_service),
) -> DataResponse[list[ModelResult]]:
    """Get all studies (model results) for a specific PMID.

    Returns all model results from different LLMs for the same research paper.
    Useful for comparing extraction results across models.
    """
    try:
        studies = await service.get_studies_by_pmid(pmid)

        if not studies:
            raise HTTPException(
                status_code=404, detail=f"No studies found for PMID {pmid}"
            )

        return DataResponse(data=studies)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting studies for PMID {pmid}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get studies for PMID: {str(e)}"
        )


@router.get(
    "/{study_id}/similar",
    response_model=DataResponse[list[SimilaritySearchResult]],
)
async def get_similar_studies(
    study_id: int,
    max_results: int = Query(
        default=10, ge=1, le=100, description="Maximum results to return"
    ),
    similarity_threshold: float = Query(
        default=0.3, ge=0.0, le=1.0, description="Minimum similarity"
    ),
    similarity_type: str = Query(
        default="trait_profile",
        description="Similarity type: trait_profile or jaccard",
    ),
    service: StudyService = Depends(get_database_service),
) -> DataResponse[list[SimilaritySearchResult]]:
    """Find studies similar to the specified study using trait profiles.

    Uses precomputed similarity scores from the trait profile database.
    Results are sorted by similarity score in descending order.
    """
    try:
        # Verify study exists
        study = service.study_repo.get_study_by_id(study_id)
        if not study:
            raise HTTPException(
                status_code=404, detail=f"Study with ID {study_id} not found"
            )

        # Find the combination in trait profile database
        combination = service.similarity_repo.find_combination(
            study.pmid, study.model
        )
        if not combination:
            return DataResponse(data=[])  # No similarities available

        # Get similarities using the specified type
        similarities = service.similarity_repo.get_similarities(
            combination.id,
            top_k=max_results,
            min_similarity=similarity_threshold,
            similarity_type=similarity_type,
        )

        # Convert to SimilaritySearchResult format
        results = [
            SimilaritySearchResult(
                query_id=str(combination.id),
                query_label=f"{combination.pmid} ({combination.model})",
                result_id=f"{sim.similar_pmid}_{sim.similar_model}",
                result_label=sim.similar_title,
                similarity=sim.trait_profile_similarity
                if similarity_type == "trait_profile"
                else sim.trait_jaccard_similarity,
            )
            for sim in similarities
        ]

        return DataResponse(data=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar studies for {study_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to find similar studies: {str(e)}"
        )


@router.get("/stats/overview", response_model=DataResponse[StudyAnalytics])
async def get_studies_overview(
    service: AnalyticsService = Depends(get_database_service),
) -> DataResponse[StudyAnalytics]:
    """Get overview statistics about studies in the database.

    Returns summary statistics including:
    - Total study and PMID counts
    - Distribution by models and journals
    - Publication year distribution
    - Trait count distribution
    """
    try:
        # Get basic counts
        total_studies = service.study_repo.get_count("model_results")
        total_pmids_query = "SELECT COUNT(DISTINCT pmid) FROM model_results"
        total_pmids = service.study_repo.execute_query(total_pmids_query)[0][0]

        # Get model distribution
        model_query = """
        SELECT model, COUNT(*) as count
        FROM model_results
        GROUP BY model
        ORDER BY count DESC
        """
        model_results = service.study_repo.execute_query(model_query)
        model_distribution = {row[0]: row[1] for row in model_results}

        # Get journal distribution
        journal_query = """
        SELECT mpd.journal, COUNT(*) as count
        FROM model_results mr
        JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        WHERE mpd.journal IS NOT NULL
        GROUP BY mpd.journal
        ORDER BY count DESC
        LIMIT 20
        """
        journal_results = service.study_repo.execute_query(journal_query)
        journal_distribution = {row[0]: row[1] for row in journal_results}

        # Get year distribution
        year_query = """
        SELECT SUBSTR(mpd.pub_date, 1, 4) as year, COUNT(*) as count
        FROM model_results mr
        JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        WHERE LENGTH(mpd.pub_date) >= 4
        GROUP BY year
        ORDER BY year DESC
        LIMIT 20
        """
        year_results = service.study_repo.execute_query(year_query)
        year_distribution = {row[0]: row[1] for row in year_results}

        # Get trait count distribution
        trait_count_query = """
        SELECT
            CASE
                WHEN trait_count = 0 THEN '0'
                WHEN trait_count BETWEEN 1 AND 5 THEN '1-5'
                WHEN trait_count BETWEEN 6 AND 10 THEN '6-10'
                WHEN trait_count BETWEEN 11 AND 20 THEN '11-20'
                WHEN trait_count BETWEEN 21 AND 50 THEN '21-50'
                ELSE '50+'
            END as range,
            COUNT(*) as count
        FROM (
            SELECT mr.id, COUNT(mrt.trait_index) as trait_count
            FROM model_results mr
            LEFT JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
            GROUP BY mr.id
        ) trait_counts
        GROUP BY 1
        ORDER BY 1
        """
        trait_count_results = service.study_repo.execute_query(
            trait_count_query
        )
        trait_count_distribution = {
            row[0]: row[1] for row in trait_count_results
        }

        analytics = StudyAnalytics(
            total_studies=total_studies,
            total_pmids=total_pmids,
            model_distribution=model_distribution,
            journal_distribution=journal_distribution,
            year_distribution=year_distribution,
            trait_count_distribution=trait_count_distribution,
        )

        return DataResponse(data=analytics)

    except Exception as e:
        logger.error(f"Error getting studies overview: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get studies overview: {str(e)}"
        )


@router.get("/models", response_model=DataResponse[list[str]])
async def get_available_models(
    service: StudyService = Depends(get_database_service),
) -> DataResponse[list[str]]:
    """Get list of available LLM models in the database.

    Returns all unique model names that have generated study results.
    """
    try:
        query = "SELECT DISTINCT model FROM model_results ORDER BY model"
        results = service.study_repo.execute_query(query)
        models = [row[0] for row in results]

        return DataResponse(data=models)

    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get available models: {str(e)}"
        )


@router.get("/journals", response_model=DataResponse[list[str]])
async def get_available_journals(
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of journals to return",
    ),
    service: StudyService = Depends(get_database_service),
) -> DataResponse[list[str]]:
    """Get list of available journals in the database.

    Returns journals ordered by frequency (most studies first).
    """
    try:
        query = """
        SELECT DISTINCT mpd.journal
        FROM mr_pubmed_data mpd
        JOIN model_results mr ON mpd.pmid = mr.pmid
        WHERE mpd.journal IS NOT NULL
        ORDER BY mpd.journal
        LIMIT ?
        """
        results = service.study_repo.execute_query(query, (limit,))
        journals = [row[0] for row in results]

        return DataResponse(data=journals)

    except Exception as e:
        logger.error(f"Error getting available journals: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available journals: {str(e)}",
        )

