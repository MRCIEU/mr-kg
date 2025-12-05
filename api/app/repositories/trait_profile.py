"""Data access layer for trait profile database.

Re-exports trait profile functions from common_funcs for backward compatibility.
"""

from common_funcs.repositories import get_similar_by_trait

__all__ = [
    "get_similar_by_trait",
]
