"""Services package for the webapp."""

from services.db_client import (
    autocomplete_studies,
    autocomplete_traits,
    check_database_health,
    filter_studies_by_similarity,
    get_available_models,
    get_extraction,
    get_similar_by_evidence,
    get_similar_by_trait,
    get_statistics,
    has_evidence_similarity,
    has_trait_similarity,
    search_studies,
)

__all__ = [
    "autocomplete_studies",
    "autocomplete_traits",
    "check_database_health",
    "filter_studies_by_similarity",
    "get_available_models",
    "get_extraction",
    "get_similar_by_evidence",
    "get_similar_by_trait",
    "get_statistics",
    "has_evidence_similarity",
    "has_trait_similarity",
    "search_studies",
]
