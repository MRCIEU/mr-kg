"""Studies router for MR-KG API.

Provides endpoints for searching studies, retrieving extraction results,
autocomplete suggestions, and resource statistics.
"""

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.database import DatabaseError
from app.models import (
    ExtractionResponse,
    ResultInfo,
    StatisticsResponse,
    StudiesResponse,
    StudyAutocompleteResponse,
    StudyBrief,
    TraitAutocompleteResponse,
    TraitInfo,
)
from app.repositories import statistics as stats_repo
from app.repositories import vector_store as vs_repo

router = APIRouter(tags=["studies"])


@router.get("/studies", response_model=StudiesResponse)
async def list_studies(
    q: str | None = Query(
        default=None,
        description="Search query for title or PMID",
    ),
    trait: str | None = Query(
        default=None,
        description="Filter by trait label",
    ),
    model: str | None = Query(
        default=None,
        description="Filter by extraction model (default: gpt-5)",
    ),
    limit: int = Query(
        default=20,
        le=100,
        ge=1,
        description="Maximum results to return (max 100)",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Pagination offset",
    ),
) -> StudiesResponse:
    """Search and list studies with optional filtering.

    Returns a paginated list of studies matching the search criteria.
    Studies can be filtered by text search (title or PMID), trait label,
    and extraction model.
    """
    settings = get_settings()
    model_name = model if model else settings.default_model

    try:
        total, studies_data = vs_repo.get_studies(
            q=q,
            trait=trait,
            model=model_name,
            limit=limit,
            offset=offset,
        )
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    studies = [
        StudyBrief(
            pmid=s["pmid"],
            title=s["title"],
            pub_date=s["pub_date"],
            journal=s["journal"],
            model=s["model"],
        )
        for s in studies_data
    ]

    res = StudiesResponse(
        total=total,
        limit=limit,
        offset=offset,
        studies=studies,
    )
    return res


@router.get("/studies/{pmid}/extraction", response_model=ExtractionResponse)
async def get_extraction(
    pmid: str,
    model: str | None = Query(
        default=None,
        description="Extraction model (default: gpt-5)",
    ),
) -> ExtractionResponse:
    """Get extraction results for a specific study.

    Returns detailed extraction results including traits, exposure-outcome
    pairs, effect sizes, and metadata from the specified extraction model.
    """
    settings = get_settings()
    model_name = model if model else settings.default_model

    try:
        data = vs_repo.get_study_extraction(pmid=pmid, model=model_name)
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Study {pmid} not found for model {model_name}",
        )

    # ---- Parse traits ----
    traits = [
        TraitInfo(
            trait_index=t["trait_index"],
            trait_label=t["trait_label"],
            trait_id_in_result=t.get("trait_id_in_result"),
        )
        for t in data.get("traits", [])
    ]

    # ---- Parse results ----
    results = [
        ResultInfo(
            exposure=r.get("exposure", ""),
            outcome=r.get("outcome", ""),
            beta=r.get("beta"),
            odds_ratio=r.get("odds_ratio"),
            hazard_ratio=r.get("hazard_ratio"),
            ci_lower=r.get("ci_lower"),
            ci_upper=r.get("ci_upper"),
            p_value=r.get("p_value"),
            direction=r.get("direction"),
        )
        for r in data.get("results", [])
    ]

    res = ExtractionResponse(
        pmid=data["pmid"],
        model=data["model"],
        title=data["title"],
        pub_date=data.get("pub_date"),
        journal=data.get("journal"),
        abstract=data.get("abstract"),
        traits=traits,
        results=results,
        metadata=data.get("metadata", {}),
    )
    return res


@router.get("/traits/autocomplete", response_model=TraitAutocompleteResponse)
async def autocomplete_traits(
    q: str = Query(
        description="Search term for trait autocomplete (prefix match)",
        min_length=2,
    ),
    limit: int = Query(
        default=20,
        le=50,
        ge=1,
        description="Maximum suggestions to return (max 50)",
    ),
) -> TraitAutocompleteResponse:
    """Get trait autocomplete suggestions.

    Returns a list of trait labels that match the search term prefix.
    Minimum 2 characters required for search.
    """
    try:
        traits = vs_repo.search_traits(search_term=q, limit=limit)
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    res = TraitAutocompleteResponse(traits=traits)
    return res


@router.get("/studies/autocomplete", response_model=StudyAutocompleteResponse)
async def autocomplete_studies(
    q: str = Query(
        description="Search term for study autocomplete (substring match)",
        min_length=2,
    ),
    limit: int = Query(
        default=20,
        le=50,
        ge=1,
        description="Maximum suggestions to return (max 50)",
    ),
) -> StudyAutocompleteResponse:
    """Get study autocomplete suggestions.

    Returns a list of studies (pmid and title) where the title contains
    the search term. Minimum 2 characters required for search.
    """
    try:
        studies = vs_repo.search_studies(search_term=q, limit=limit)
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    res = StudyAutocompleteResponse(studies=studies)
    return res


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics() -> StatisticsResponse:
    """Get resource-wide statistics.

    Returns overall statistics about the database, model-specific
    similarity stats, and evidence stats.
    """
    try:
        overall = stats_repo.get_overall_statistics()
        model_similarity_stats = stats_repo.get_model_similarity_stats()
        model_evidence_stats = stats_repo.get_model_evidence_stats()
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    res = StatisticsResponse(
        overall=overall,
        model_similarity_stats=model_similarity_stats,
        model_evidence_stats=model_evidence_stats,
    )
    return res
