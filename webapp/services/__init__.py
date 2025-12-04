"""Services package for the webapp."""

from services.api_client import (
    AutocompleteResult,
    autocomplete_studies,
    autocomplete_studies_with_status,
    autocomplete_traits,
    autocomplete_traits_with_status,
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
    "autocomplete_traits_with_status",
    "autocomplete_studies_with_status",
    "AutocompleteResult",
    "get_statistics",
    "check_health",
]
