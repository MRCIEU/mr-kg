"""Data access layer for trait profile database.

Provides query functions for trait profile similarity searches
from trait_profile_db.db.
"""

from typing import Any

from app.database import get_trait_profile_connection


def get_similar_by_trait(
    pmid: str,
    model: str,
    limit: int = 10,
) -> dict[str, Any] | None:
    """Get similar studies by trait profile similarity.

    Args:
        pmid: PubMed ID of the query study
        model: Extraction model name
        limit: Maximum number of similar studies to return

    Returns:
        Dict with query info and list of similar studies, or None if not found
    """
    conn = get_trait_profile_connection()

    query = """
        SELECT
            qc.pmid as query_pmid,
            qc.title as query_title,
            qc.trait_count as query_trait_count,
            ts.similar_pmid as pmid,
            ts.similar_title as title,
            ts.trait_profile_similarity,
            ts.trait_jaccard_similarity,
            ts.similar_trait_count as trait_count
        FROM query_combinations qc
        JOIN trait_similarities ts ON qc.id = ts.query_combination_id
        WHERE qc.pmid = ? AND qc.model = ?
        ORDER BY ts.trait_profile_similarity DESC
        LIMIT ?
    """

    result = conn.execute(query, [pmid, model, limit]).fetchall()

    if not result:
        # ---- Check if query combination exists ----
        check_query = """
            SELECT pmid, title, trait_count
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
            "query_trait_count": check_result[2],
            "similar_studies": [],
        }
        return res

    # ---- Build response from first row for query info ----
    first_row = result[0]
    similar_studies = [
        {
            "pmid": row[3],
            "title": row[4],
            "trait_profile_similarity": row[5],
            "trait_jaccard_similarity": row[6],
            "trait_count": row[7],
        }
        for row in result
    ]

    res = {
        "query_pmid": first_row[0],
        "query_model": model,
        "query_title": first_row[1],
        "query_trait_count": first_row[2],
        "similar_studies": similar_studies,
    }
    return res
