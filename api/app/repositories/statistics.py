"""Data access layer for aggregate statistics.

Provides query functions for resource-wide statistics from all databases.
"""

from typing import Any

from app.database import (
    get_evidence_profile_connection,
    get_trait_profile_connection,
    get_vector_store_connection,
)


def get_overall_statistics() -> dict[str, Any]:
    """Get overall resource statistics from vector_store.db.

    Returns:
        Dict with total_papers, total_traits, total_models, total_extractions
    """
    conn = get_vector_store_connection()

    query = """
        SELECT
            (SELECT COUNT(DISTINCT pmid) FROM mr_pubmed_data) as total_papers,
            (SELECT COUNT(*) FROM trait_embeddings) as total_traits,
            (SELECT COUNT(DISTINCT model) FROM model_results) as total_models,
            (SELECT COUNT(*) FROM model_results) as total_extractions
    """

    result = conn.execute(query).fetchone()

    res = {
        "total_papers": result[0],
        "total_traits": result[1],
        "total_models": result[2],
        "total_extractions": result[3],
    }
    return res


def get_model_similarity_stats() -> list[dict[str, Any]]:
    """Get model statistics from trait_profile_db.db.

    Returns:
        List of dicts with model stats (model, total_combinations,
        avg_trait_count, etc.)
    """
    conn = get_trait_profile_connection()

    query = """
        SELECT * FROM model_similarity_stats
    """

    result = conn.execute(query).fetchall()
    columns = [
        "model",
        "total_combinations",
        "avg_trait_count",
        "min_trait_count",
        "max_trait_count",
        "total_similarity_pairs",
    ]

    res = [dict(zip(columns, row)) for row in result]
    return res


def get_model_evidence_stats() -> list[dict[str, Any]]:
    """Get evidence statistics from evidence_profile_db.db.

    Returns:
        List of dicts with model evidence stats (model, total_combinations,
        avg_result_count, etc.)
    """
    conn = get_evidence_profile_connection()

    query = """
        SELECT * FROM model_evidence_stats
    """

    result = conn.execute(query).fetchall()
    columns = [
        "model",
        "total_combinations",
        "avg_result_count",
        "avg_completeness",
        "min_result_count",
        "max_result_count",
        "total_similarity_pairs",
    ]

    res = [dict(zip(columns, row)) for row in result]
    return res


def get_metric_availability() -> dict[str, Any]:
    """Get metric availability from evidence_profile_db.db.

    Returns:
        Dict with metric availability statistics
    """
    conn = get_evidence_profile_connection()

    query = """
        SELECT * FROM metric_availability
    """

    result = conn.execute(query).fetchone()

    if result is None:
        res = {}
        return res

    columns = [
        "total_comparisons",
        "direction_concordance_available",
        "effect_size_similarity_available",
        "composite_equal_available",
        "composite_direction_available",
        "statistical_consistency_available",
        "precision_concordance_available",
        "direction_concordance_pct",
        "effect_size_similarity_pct",
        "composite_equal_pct",
        "composite_direction_pct",
        "statistical_consistency_pct",
        "precision_concordance_pct",
    ]

    res = dict(zip(columns, result))
    return res
