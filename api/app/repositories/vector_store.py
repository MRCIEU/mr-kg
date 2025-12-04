"""Data access layer for vector store database.

Provides query functions for trait autocomplete, study search, and extraction
results from vector_store.db.
"""

import json
from typing import Any

from app.database import get_vector_store_connection


def search_traits(search_term: str, model: str, limit: int = 20) -> list[str]:
    """Autocomplete suggestions for traits (prefix search).

    Args:
        search_term: Search term for prefix matching
        model: Filter by extraction model
        limit: Maximum number of suggestions to return

    Returns:
        List of trait labels matching the search term that have
        extraction results for the specified model
    """
    conn = get_vector_store_connection()

    query = """
        SELECT DISTINCT mrt.trait_label
        FROM model_result_traits mrt
        JOIN model_results mr ON mrt.model_result_id = mr.id
        WHERE mrt.trait_label ILIKE ? || '%'
            AND mr.model = ?
        ORDER BY mrt.trait_label
        LIMIT ?
    """

    result = conn.execute(query, [search_term, model, limit]).fetchall()
    res = [row[0] for row in result]
    return res


def search_studies(
    search_term: str, model: str, limit: int = 20
) -> list[dict[str, Any]]:
    """Autocomplete suggestions for studies (pmid, title).

    Args:
        search_term: Search term for substring matching in title
        model: Filter by extraction model
        limit: Maximum number of suggestions to return

    Returns:
        List of dicts with pmid and title for studies that have
        extraction results for the specified model
    """
    conn = get_vector_store_connection()

    query = """
        SELECT DISTINCT mr.pmid, mpd.title
        FROM model_results mr
        JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        WHERE mpd.title ILIKE '%' || ? || '%'
            AND mr.model = ?
        ORDER BY mpd.pub_date DESC
        LIMIT ?
    """

    result = conn.execute(query, [search_term, model, limit]).fetchall()
    res = [{"pmid": row[0], "title": row[1]} for row in result]
    return res


def get_studies(
    q: str | None,
    trait: str | None,
    model: str,
    limit: int,
    offset: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Search studies with filtering.

    Args:
        q: Optional search query for title or PMID
        trait: Optional filter by trait label
        model: Filter by extraction model
        limit: Maximum results to return
        offset: Pagination offset

    Returns:
        Tuple of (total_count, list of study dicts)
    """
    conn = get_vector_store_connection()

    # ---- Build base query ----
    base_query = """
        SELECT DISTINCT
            mr.pmid,
            mpd.title,
            mpd.pub_date,
            mpd.journal,
            mr.model
        FROM model_results mr
        JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        WHERE mr.model = ?
    """

    count_query = """
        SELECT COUNT(DISTINCT mr.pmid)
        FROM model_results mr
        JOIN mr_pubmed_data mpd ON mr.pmid = mpd.pmid
        WHERE mr.model = ?
    """

    params: list[Any] = [model]
    count_params: list[Any] = [model]

    # ---- Add text search filter ----
    if q:
        text_filter = """
            AND (
                mpd.title ILIKE '%' || ? || '%'
                OR mr.pmid = ?
            )
        """
        base_query += text_filter
        count_query += text_filter
        params.extend([q, q])
        count_params.extend([q, q])

    # ---- Add trait filter ----
    if trait:
        trait_filter = """
            AND mr.id IN (
                SELECT model_result_id
                FROM model_result_traits
                WHERE trait_label = ?
            )
        """
        base_query += trait_filter
        count_query += trait_filter
        params.append(trait)
        count_params.append(trait)

    # ---- Add ordering and pagination ----
    base_query += " ORDER BY mpd.pub_date DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    # ---- Execute queries ----
    count_result = conn.execute(count_query, count_params).fetchone()
    total = count_result[0] if count_result else 0

    result = conn.execute(base_query, params).fetchall()
    studies = [
        {
            "pmid": row[0],
            "title": row[1],
            "pub_date": row[2],
            "journal": row[3],
            "model": row[4],
        }
        for row in result
    ]

    res = (total, studies)
    return res


def get_study_extraction(pmid: str, model: str) -> dict[str, Any] | None:
    """Get extraction results from pmid_model_analysis view.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model name

    Returns:
        Dict with extraction results or None if not found
    """
    conn = get_vector_store_connection()

    query = """
        SELECT
            pmid,
            model,
            title,
            pub_date,
            journal,
            abstract,
            traits,
            results,
            metadata
        FROM pmid_model_analysis
        WHERE pmid = ? AND model = ?
    """

    result = conn.execute(query, [pmid, model]).fetchone()

    if result is None:
        return None

    # ---- Parse JSON fields ----
    traits_data = result[6]
    results_data = result[7]
    metadata_data = result[8]

    # Handle traits - it's a list of structs from DuckDB
    if isinstance(traits_data, list):
        traits = [
            {
                "trait_index": t["trait_index"]
                if isinstance(t, dict)
                else t[0],
                "trait_label": t["trait_label"]
                if isinstance(t, dict)
                else t[1],
                "trait_id_in_result": t.get("trait_id_in_result")
                if isinstance(t, dict)
                else t[2],
            }
            for t in traits_data
        ]
    else:
        traits = []

    # Handle results - stored as JSON
    if isinstance(results_data, str):
        results = json.loads(results_data)
    elif results_data is None:
        results = []
    else:
        results = results_data

    # Handle metadata - stored as JSON
    if isinstance(metadata_data, str):
        metadata = json.loads(metadata_data)
    elif metadata_data is None:
        metadata = {}
    else:
        metadata = metadata_data

    res = {
        "pmid": result[0],
        "model": result[1],
        "title": result[2],
        "pub_date": result[3],
        "journal": result[4],
        "abstract": result[5],
        "traits": traits,
        "results": results,
        "metadata": metadata,
    }
    return res


def get_available_models() -> list[str]:
    """Get list of available extraction models.

    Returns:
        List of model names
    """
    conn = get_vector_store_connection()

    query = """
        SELECT DISTINCT model
        FROM model_results
        ORDER BY model
    """

    result = conn.execute(query).fetchall()
    res = [row[0] for row in result]
    return res
