"""Data access layer for evidence profile database.

Provides query functions for evidence profile similarity searches
from evidence_profile_db.db.
"""

from typing import Any

from common_funcs.repositories.connection import (
    get_evidence_profile_connection,
)


def get_similar_by_evidence(
    pmid: str,
    model: str,
    limit: int = 10,
) -> dict[str, Any] | None:
    """Get similar studies by evidence profile similarity.

    Args:
        pmid: PubMed ID of the query study
        model: Extraction model name
        limit: Maximum number of similar studies to return

    Returns:
        Dict with query info and list of similar studies, or None if not found
    """
    conn = get_evidence_profile_connection()

    query = """
        SELECT
            qc.pmid as query_pmid,
            qc.title as query_title,
            qc.result_count as query_result_count,
            es.similar_pmid as pmid,
            es.similar_title as title,
            es.direction_concordance,
            es.matched_pairs,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo
        FROM query_combinations qc
        JOIN evidence_similarities es ON qc.id = es.query_combination_id
        WHERE qc.pmid = ? AND qc.model = ?
        ORDER BY es.direction_concordance DESC
        LIMIT ?
    """

    result = conn.execute(query, [pmid, model, limit]).fetchall()

    if not result:
        # ---- Check if query combination exists ----
        check_query = """
            SELECT pmid, title, result_count
            FROM query_combinations
            WHERE pmid = ? AND model = ?
        """
        check_result = conn.execute(check_query, [pmid, model]).fetchone()

        if check_result is None:
            return None

        # Query combination exists but no similar studies found
        res = {
            "query_pmid": check_result[0],
            "query_model": model,
            "query_title": check_result[1],
            "query_result_count": check_result[2],
            "similar_studies": [],
        }
        return res

    # ---- Build response from first row for query info ----
    first_row = result[0]
    similar_studies = [
        {
            "pmid": row[3],
            "title": row[4],
            "direction_concordance": row[5],
            "matched_pairs": row[6],
            "match_type_exact": bool(row[7]),
            "match_type_fuzzy": bool(row[8]),
            "match_type_efo": bool(row[9]),
        }
        for row in result
    ]

    res = {
        "query_pmid": first_row[0],
        "query_model": model,
        "query_title": first_row[1],
        "query_result_count": first_row[2],
        "similar_studies": similar_studies,
    }
    return res
