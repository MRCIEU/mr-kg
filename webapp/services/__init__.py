"""Services package for the webapp."""

from services.api_client import (
    autocomplete_studies,
    autocomplete_traits,
    check_health,
    get_extraction,
    get_similar_by_evidence,
    get_similar_by_trait,
    get_statistics,
    search_studies,
)

__all__ = [
    "search_studies",
    "get_extraction",
    "get_similar_by_trait",
    "get_similar_by_evidence",
    "autocomplete_traits",
    "autocomplete_studies",
    "get_statistics",
    "check_health",
]
