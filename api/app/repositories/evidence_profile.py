"""Data access layer for evidence profile database.

Re-exports evidence profile functions from common_funcs for backward
compatibility.
"""

from common_funcs.repositories import get_similar_by_evidence

__all__ = [
    "get_similar_by_evidence",
]
