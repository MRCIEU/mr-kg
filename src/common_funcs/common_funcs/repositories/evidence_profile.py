"""Data access layer for evidence profile database.

Provides query functions for evidence profile similarity searches
from evidence_profile_db.db.
"""

import json
from typing import Any

from common_funcs.repositories.connection import (
    get_evidence_profile_connection,
    get_vector_store_connection,
)


def _get_study_evidence_pairs(pmid: str, model: str) -> list[dict[str, Any]]:
    """Get evidence pairs (exposure-outcome) for a study from vector store.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model name

    Returns:
        List of evidence pair dicts with exposure, outcome, and direction
    """
    conn = get_vector_store_connection()

    query = """
        SELECT results
        FROM model_results
        WHERE pmid = ? AND model = ?
    """

    result = conn.execute(query, [pmid, model]).fetchone()

    if result is None:
        return []

    results_data = result[0]

    # ---- Parse JSON if needed ----
    if isinstance(results_data, str):
        results = json.loads(results_data)
    elif results_data is None:
        results = []
    else:
        results = results_data

    # ---- Extract exposure-outcome pairs with direction ----
    pairs = []
    for r in results:
        exposure = r.get("exposure", "")
        outcome = r.get("outcome", "")
        direction = r.get("direction", "")

        if exposure and outcome:
            pairs.append(
                {
                    "exposure": exposure,
                    "outcome": outcome,
                    "direction": direction,
                }
            )

    return pairs


def _get_study_evidence_pairs_with_traits(
    pmid: str, model: str
) -> list[dict[str, Any]]:
    """Get evidence pairs with trait indices for a study.

    This retrieves the exposure/outcome trait indices which can be used
    for fuzzy matching based on trait embedding similarity.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model name

    Returns:
        List of evidence pair dicts with exposure, outcome, direction,
        and trait indices
    """
    conn = get_vector_store_connection()

    # ---- Get results JSON ----
    results_query = """
        SELECT results
        FROM model_results
        WHERE pmid = ? AND model = ?
    """
    result = conn.execute(results_query, [pmid, model]).fetchone()

    if result is None:
        return []

    results_data = result[0]
    if isinstance(results_data, str):
        results = json.loads(results_data)
    elif results_data is None:
        results = []
    else:
        results = results_data

    # ---- Get trait mappings ----
    traits_query = """
        SELECT mrt.trait_label, mrt.trait_index, mrt.trait_id_in_result
        FROM model_result_traits mrt
        JOIN model_results mr ON mrt.model_result_id = mr.id
        WHERE mr.pmid = ? AND mr.model = ?
    """
    trait_rows = conn.execute(traits_query, [pmid, model]).fetchall()

    # Build lookup: trait_id_in_result -> trait_index
    trait_id_to_index = {}
    for row in trait_rows:
        trait_label, trait_index, trait_id = row
        if trait_id:
            trait_id_to_index[trait_id] = (trait_index, trait_label)

    # ---- Build pairs with trait indices ----
    pairs = []
    for r in results:
        exposure = r.get("exposure", "")
        outcome = r.get("outcome", "")
        direction = r.get("direction", "")
        exposure_id = r.get("exposure_id", "")
        outcome_id = r.get("outcome_id", "")

        if not (exposure and outcome):
            continue

        exp_info = trait_id_to_index.get(exposure_id, (None, exposure))
        out_info = trait_id_to_index.get(outcome_id, (None, outcome))

        pairs.append(
            {
                "exposure": exposure,
                "outcome": outcome,
                "direction": direction,
                "exposure_trait_index": exp_info[0],
                "outcome_trait_index": out_info[0],
            }
        )

    return pairs


def _compute_matched_evidence_pairs(
    query_pairs: list[dict[str, Any]],
    similar_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the intersection of evidence pairs between two studies.

    Matches are based on exposure-outcome trait indices when available,
    falling back to case-insensitive name matching.

    Args:
        query_pairs: Evidence pairs from query study (with trait indices)
        similar_pairs: Evidence pairs from similar study (with trait indices)

    Returns:
        List of matched evidence pair dicts with both query and similar info
    """
    # ---- Build lookup for query pairs by trait indices ----
    query_by_indices = {}
    query_by_names = {}

    for p in query_pairs:
        exp_idx = p.get("exposure_trait_index")
        out_idx = p.get("outcome_trait_index")

        if exp_idx is not None and out_idx is not None:
            key = (exp_idx, out_idx)
            query_by_indices[key] = p

        # Also build name-based lookup as fallback
        name_key = (p["exposure"].lower(), p["outcome"].lower())
        query_by_names[name_key] = p

    # ---- Find matches ----
    matched = []
    for sp in similar_pairs:
        qp = None

        # Try trait index matching first
        exp_idx = sp.get("exposure_trait_index")
        out_idx = sp.get("outcome_trait_index")

        if exp_idx is not None and out_idx is not None:
            key = (exp_idx, out_idx)
            qp = query_by_indices.get(key)

        # Fall back to name matching
        if qp is None:
            name_key = (sp["exposure"].lower(), sp["outcome"].lower())
            qp = query_by_names.get(name_key)

        if qp is not None:
            matched.append(
                {
                    "exposure": sp["exposure"],
                    "outcome": sp["outcome"],
                    "query_direction": qp.get("direction", ""),
                    "similar_direction": sp.get("direction", ""),
                }
            )

    return matched


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

    similar_studies = []
    for row in result:
        similar_pmid = row[3]

        # ---- Get similar study evidence pairs ----
        evidence_pairs = _get_study_evidence_pairs(similar_pmid, model)

        similar_studies.append(
            {
                "pmid": similar_pmid,
                "title": row[4],
                "direction_concordance": row[5],
                "matched_pairs": row[6],
                "match_type_exact": bool(row[7]),
                "match_type_fuzzy": bool(row[8]),
                "match_type_efo": bool(row[9]),
                "evidence_pairs": evidence_pairs,
            }
        )

    res = {
        "query_pmid": first_row[0],
        "query_model": model,
        "query_title": first_row[1],
        "query_result_count": first_row[2],
        "similar_studies": similar_studies,
    }
    return res
