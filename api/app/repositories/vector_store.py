"""Data access layer for vector store database.

Re-exports vector store functions from common_funcs for backward compatibility.
"""

from common_funcs.repositories import (
    get_available_models,
    get_studies,
    get_study_extraction,
    search_studies,
    search_traits,
)

__all__ = [
    "search_traits",
    "search_studies",
    "get_studies",
    "get_study_extraction",
    "get_available_models",
]
