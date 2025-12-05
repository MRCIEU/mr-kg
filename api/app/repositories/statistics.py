"""Data access layer for aggregate statistics.

Re-exports statistics functions from common_funcs for backward compatibility.
"""

from common_funcs.repositories import (
    get_metric_availability,
    get_model_evidence_stats,
    get_model_similarity_stats,
    get_overall_statistics,
)

__all__ = [
    "get_overall_statistics",
    "get_model_similarity_stats",
    "get_model_evidence_stats",
    "get_metric_availability",
]
