"""Services package for the webapp."""

from services.db_client import (
    autocomplete_studies,
    autocomplete_traits,
    check_database_health,
    get_available_models,
    get_extraction,
    get_similar_by_evidence,
    get_similar_by_trait,
    get_statistics,
    search_studies,
)

__all__ = [
    "autocomplete_studies",
    "autocomplete_traits",
    "check_database_health",
    "get_available_models",
    "get_extraction",
    "get_similar_by_evidence",
    "get_similar_by_trait",
    "get_statistics",
    "search_studies",
]
