"""Database connection management for MR-KG API.

Re-exports connection utilities from common_funcs for backward compatibility.
"""

from common_funcs.repositories import (
    DatabaseError,
    close_all_connections,
    evidence_profile_connection,
    get_evidence_profile_connection,
    get_trait_profile_connection,
    get_vector_store_connection,
    trait_profile_connection,
    vector_store_connection,
)

__all__ = [
    "DatabaseError",
    "get_vector_store_connection",
    "get_trait_profile_connection",
    "get_evidence_profile_connection",
    "vector_store_connection",
    "trait_profile_connection",
    "evidence_profile_connection",
    "close_all_connections",
]
