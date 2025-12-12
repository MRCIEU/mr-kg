"""Database client service for direct database access.

Provides functions for querying MR-KG databases directly, replacing the
previous API-based approach.
"""

import logging
import os
from typing import Any

import streamlit as st

from config import get_settings

logger = logging.getLogger(__name__)


def _configure_database_paths() -> None:
    """Configure database paths from webapp settings.

    Sets environment variables that common_funcs.repositories.config will read.
    This ensures the repository layer uses the webapp's configured paths.
    Must be called BEFORE importing from common_funcs.repositories.
    """
    settings = get_settings()
    os.environ.setdefault("VECTOR_STORE_PATH", settings.vector_store_path)
    os.environ.setdefault("TRAIT_PROFILE_PATH", settings.trait_profile_path)
    os.environ.setdefault(
        "EVIDENCE_PROFILE_PATH", settings.evidence_profile_path
    )


# Configure paths BEFORE importing from common_funcs.repositories
# This ensures the repository settings are loaded with correct paths
_configure_database_paths()

# Now import from common_funcs.repositories after paths are configured
from common_funcs.repositories import (  # noqa: E402
    get_available_models as _get_available_models,
    get_metric_availability,
    get_model_evidence_stats,
    get_model_similarity_stats,
    get_overall_statistics,
    get_studies,
    get_study_extraction,
    search_studies as _search_studies,
    search_traits as _search_traits,
)
from common_funcs.repositories import (  # noqa: E402
    get_similar_by_evidence as _get_evidence,
)
from common_funcs.repositories import (  # noqa: E402
    get_similar_by_trait as _get_trait,
)


# ==== Studies functions ====


@st.cache_data(ttl=300)
def search_studies(
    q: str | None = None,
    trait: str | None = None,
    model: str = "gpt-5",
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Search for studies with optional filtering.

    Args:
        q: Search query for title or PMID
        trait: Filter by trait label
        model: Extraction model filter
        limit: Maximum results to return
        offset: Pagination offset

    Returns:
        Dict with total count and list of studies
    """
    try:
        total, studies = get_studies(
            q=q,
            trait=trait,
            model=model,
            limit=limit,
            offset=offset,
        )
        res = {
            "total": total,
            "limit": limit,
            "offset": offset,
            "studies": studies,
        }
        return res
    except Exception as e:
        logger.error(f"Error searching studies: {e}")
        return {"total": 0, "limit": limit, "offset": offset, "studies": []}


def get_extraction(pmid: str, model: str = "gpt-5") -> dict[str, Any] | None:
    """Get extraction results for a specific study.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model

    Returns:
        Extraction data or None if not found
    """
    try:
        return get_study_extraction(pmid=pmid, model=model)
    except Exception as e:
        logger.error(f"Error getting extraction for {pmid}: {e}")
        return None


# ==== Similarity functions ====


def get_similar_by_trait(
    pmid: str,
    model: str = "gpt-5",
    limit: int = 10,
) -> dict[str, Any] | None:
    """Get similar studies by trait profile similarity.

    Args:
        pmid: PubMed ID of the query study
        model: Extraction model
        limit: Maximum similar studies to return

    Returns:
        Similarity data or None if not found
    """
    try:
        return _get_trait(pmid=pmid, model=model, limit=limit)
    except Exception as e:
        logger.error(f"Error getting trait similarity for {pmid}: {e}")
        return None


def get_similar_by_evidence(
    pmid: str,
    model: str = "gpt-5",
    limit: int = 10,
    compute_matched_pairs: bool = False,
) -> dict[str, Any] | None:
    """Get similar studies by evidence profile similarity.

    Args:
        pmid: PubMed ID of the query study
        model: Extraction model
        limit: Maximum similar studies to return
        compute_matched_pairs: If True, compute and return the actual matched
            evidence pairs (computationally expensive). If False, return None
            for matched_evidence_pairs field.

    Returns:
        Similarity data or None if not found
    """
    try:
        return _get_evidence(
            pmid=pmid,
            model=model,
            limit=limit,
            compute_matched_pairs=compute_matched_pairs,
        )
    except Exception as e:
        logger.error(f"Error getting evidence similarity for {pmid}: {e}")
        return None


def has_trait_similarity(pmid: str, model: str = "gpt-5") -> bool:
    """Check if a study has related studies by trait profile similarity.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model

    Returns:
        True if study has at least one related study by trait similarity
    """
    result = get_similar_by_trait(pmid=pmid, model=model, limit=1)
    if result is None:
        return False
    return len(result.get("similar_studies", [])) > 0


def has_evidence_similarity(pmid: str, model: str = "gpt-5") -> bool:
    """Check if a study has related studies by evidence profile similarity.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model

    Returns:
        True if study has at least one related study by evidence similarity
    """
    result = get_similar_by_evidence(pmid=pmid, model=model, limit=1)
    if result is None:
        return False
    return len(result.get("similar_studies", [])) > 0


def filter_studies_by_similarity(
    studies: list[dict[str, Any]],
    model: str = "gpt-5",
    require_trait_similarity: bool = False,
    require_evidence_similarity: bool = False,
) -> list[dict[str, Any]]:
    """Filter studies to only those with related studies.

    Args:
        studies: List of study dicts (must have 'pmid' key)
        model: Extraction model
        require_trait_similarity: If True, only keep studies with trait
            similarity results
        require_evidence_similarity: If True, only keep studies with evidence
            similarity results

    Returns:
        Filtered list of studies meeting the criteria
    """
    if not require_trait_similarity and not require_evidence_similarity:
        return studies

    filtered = []
    for study in studies:
        pmid = study.get("pmid")
        if not pmid:
            continue

        # Check trait similarity if required
        if require_trait_similarity and not has_trait_similarity(pmid, model):
            continue

        # Check evidence similarity if required
        if require_evidence_similarity and not has_evidence_similarity(
            pmid, model
        ):
            continue

        filtered.append(study)

    return filtered


# ==== Autocomplete functions ====


@st.cache_data(ttl=300)
def autocomplete_traits(
    q: str, model: str = "gpt-5", limit: int = 20
) -> list[str]:
    """Get trait autocomplete suggestions.

    Args:
        q: Search term (prefix match)
        model: Extraction model filter
        limit: Maximum suggestions to return

    Returns:
        List of trait labels matching the search term
    """
    if len(q) < 2:
        return []

    try:
        return _search_traits(search_term=q, model=model, limit=limit)
    except Exception as e:
        logger.error(f"Error in trait autocomplete: {e}")
        return []


@st.cache_data(ttl=300)
def autocomplete_studies(
    q: str, model: str = "gpt-5", limit: int = 20
) -> list[dict[str, Any]]:
    """Get study autocomplete suggestions.

    Args:
        q: Search term (substring match in title)
        model: Extraction model filter
        limit: Maximum suggestions to return

    Returns:
        List of dicts with pmid and title
    """
    if len(q) < 2:
        return []

    try:
        return _search_studies(search_term=q, model=model, limit=limit)
    except Exception as e:
        logger.error(f"Error in study autocomplete: {e}")
        return []


@st.cache_data(ttl=300)
def search_study_by_pmid(
    pmid: str, model: str = "gpt-5"
) -> list[dict[str, Any]]:
    """Search for a study by exact PMID match.

    Args:
        pmid: PubMed ID to search for (exact match)
        model: Extraction model filter

    Returns:
        List with single dict (pmid and title) if found, empty list otherwise
    """
    from common_funcs.repositories.connection import (
        get_vector_store_connection,
    )

    try:
        conn = get_vector_store_connection()

        query = """
            SELECT DISTINCT mr.pmid, mpd.title
            FROM model_results mr
            JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
            WHERE mr.pmid = ?
                AND mr.model = ?
        """

        result = conn.execute(query, [pmid, model]).fetchall()
        res = [{"pmid": row[0], "title": row[1]} for row in result]
        return res
    except Exception as e:
        logger.error(f"Error in PMID search: {e}")
        return []


# ==== Statistics functions ====


@st.cache_data(ttl=3600)
def get_statistics() -> dict[str, Any] | None:
    """Get resource-wide statistics.

    Returns:
        Statistics data or None on error
    """
    try:
        overall = get_overall_statistics()
        model_sim_stats = get_model_similarity_stats()
        model_ev_stats = get_model_evidence_stats()
        metric_avail = get_metric_availability()

        res = {
            "overall": overall,
            "model_similarity_stats": model_sim_stats,
            "model_evidence_stats": model_ev_stats,
            "metric_availability": metric_avail,
        }
        return res
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return None


def get_available_models() -> list[str]:
    """Get list of available extraction models.

    Returns:
        List of model names
    """
    try:
        return _get_available_models()
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return []


# ==== Health check ====


def check_database_health() -> dict[str, Any]:
    """Check database connectivity status.

    Returns:
        Health status dict with database connectivity info
    """
    from pathlib import Path

    settings = get_settings()

    databases = {
        "vector_store": Path(settings.vector_store_path).exists(),
        "trait_profile": Path(settings.trait_profile_path).exists(),
        "evidence_profile": Path(settings.evidence_profile_path).exists(),
    }

    all_healthy = all(databases.values())

    res = {
        "status": "healthy" if all_healthy else "unhealthy",
        "databases": databases,
    }
    return res
