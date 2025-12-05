"""Data access layer for MR-KG databases.

This module provides shared repository functions for accessing DuckDB databases.
Used by both the API and webapp components.
"""

from common_funcs.repositories.config import get_settings, RepositorySettings
from common_funcs.repositories.connection import (
    DatabaseError,
    get_vector_store_connection,
    get_trait_profile_connection,
    get_evidence_profile_connection,
    vector_store_connection,
    trait_profile_connection,
    evidence_profile_connection,
    close_all_connections,
)
from common_funcs.repositories.vector_store import (
    search_traits,
    search_studies,
    get_studies,
    get_study_extraction,
    get_available_models,
)
from common_funcs.repositories.trait_profile import get_similar_by_trait
from common_funcs.repositories.evidence_profile import get_similar_by_evidence
from common_funcs.repositories.statistics import (
    get_overall_statistics,
    get_model_similarity_stats,
    get_model_evidence_stats,
    get_metric_availability,
)

__all__ = [
    # Config
    "get_settings",
    "RepositorySettings",
    # Connection
    "DatabaseError",
    "get_vector_store_connection",
    "get_trait_profile_connection",
    "get_evidence_profile_connection",
    "vector_store_connection",
    "trait_profile_connection",
    "evidence_profile_connection",
    "close_all_connections",
    # Vector store
    "search_traits",
    "search_studies",
    "get_studies",
    "get_study_extraction",
    "get_available_models",
    # Trait profile
    "get_similar_by_trait",
    # Evidence profile
    "get_similar_by_evidence",
    # Statistics
    "get_overall_statistics",
    "get_model_similarity_stats",
    "get_model_evidence_stats",
    "get_metric_availability",
]
