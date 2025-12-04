"""Similarity router for MR-KG API.

Provides endpoints for finding similar studies by trait profile and
evidence profile similarity.
"""

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.database import DatabaseError
from app.models import (
    EvidenceSimilarityResponse,
    SimilarStudyEvidence,
    SimilarStudyTrait,
    TraitSimilarityResponse,
)
from app.repositories import evidence_profile as ep_repo
from app.repositories import trait_profile as tp_repo

router = APIRouter(tags=["similarity"])


@router.get(
    "/studies/{pmid}/similar/trait",
    response_model=TraitSimilarityResponse,
)
async def get_similar_by_trait(
    pmid: str,
    model: str | None = Query(
        default=None,
        description="Extraction model (default: gpt-5)",
    ),
    limit: int = Query(
        default=10,
        le=50,
        ge=1,
        description="Maximum similar studies to return (max 50)",
    ),
) -> TraitSimilarityResponse:
    """Get similar studies by trait profile similarity.

    Returns studies with similar trait profiles based on semantic similarity
    (cosine similarity of trait embeddings) and Jaccard similarity
    (set overlap of traits).
    """
    settings = get_settings()
    model_name = model if model else settings.default_model

    try:
        data = tp_repo.get_similar_by_trait(
            pmid=pmid,
            model=model_name,
            limit=limit,
        )
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Study {pmid} not found for model {model_name}",
        )

    similar_studies = [
        SimilarStudyTrait(
            pmid=s["pmid"],
            title=s["title"],
            trait_profile_similarity=s["trait_profile_similarity"],
            trait_jaccard_similarity=s["trait_jaccard_similarity"],
            trait_count=s["trait_count"],
        )
        for s in data.get("similar_studies", [])
    ]

    res = TraitSimilarityResponse(
        query_pmid=data["query_pmid"],
        query_model=data["query_model"],
        query_title=data["query_title"],
        query_trait_count=data["query_trait_count"],
        similar_studies=similar_studies,
    )
    return res


@router.get(
    "/studies/{pmid}/similar/evidence",
    response_model=EvidenceSimilarityResponse,
)
async def get_similar_by_evidence(
    pmid: str,
    model: str | None = Query(
        default=None,
        description="Extraction model (default: gpt-5)",
    ),
    limit: int = Query(
        default=10,
        le=50,
        ge=1,
        description="Maximum similar studies to return (max 50)",
    ),
) -> EvidenceSimilarityResponse:
    """Get similar studies by evidence profile similarity.

    Returns studies with similar evidence profiles based on direction
    concordance (agreement in effect direction classifications) and
    matched exposure-outcome pairs.
    """
    settings = get_settings()
    model_name = model if model else settings.default_model

    try:
        data = ep_repo.get_similar_by_evidence(
            pmid=pmid,
            model=model_name,
            limit=limit,
        )
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Study {pmid} not found for model {model_name}",
        )

    similar_studies = [
        SimilarStudyEvidence(
            pmid=s["pmid"],
            title=s["title"],
            direction_concordance=s["direction_concordance"],
            matched_pairs=s["matched_pairs"],
            match_type_exact=s["match_type_exact"],
            match_type_fuzzy=s["match_type_fuzzy"],
            match_type_efo=s["match_type_efo"],
        )
        for s in data.get("similar_studies", [])
    ]

    res = EvidenceSimilarityResponse(
        query_pmid=data["query_pmid"],
        query_model=data["query_model"],
        query_title=data["query_title"],
        query_result_count=data["query_result_count"],
        similar_studies=similar_studies,
    )
    return res
