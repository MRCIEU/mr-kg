"""Data access layer for evidence profile database.

Provides query functions for evidence profile similarity searches
from evidence_profile_db.db.
"""

import json
import math
from typing import Any

from common_funcs.repositories.connection import (
    get_evidence_profile_connection,
    get_vector_store_connection,
)

# ==== Constants ====

# Matching thresholds (same as used in preprocessing pipeline)
FUZZY_SIMILARITY_THRESHOLD = 0.70
EFO_SIMILARITY_THRESHOLD = 0.50


# ==== Embedding and similarity functions ====


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def _get_trait_embeddings(trait_indices: set[int]) -> dict[int, list[float]]:
    """Load trait embeddings for specified trait indices.

    Args:
        trait_indices: Set of trait indices to load embeddings for

    Returns:
        Dict mapping trait_index to embedding vector
    """
    if not trait_indices:
        return {}

    conn = get_vector_store_connection()

    # Build parameterized query for the indices
    placeholders = ",".join("?" * len(trait_indices))
    query = f"""
        SELECT trait_index, vector
        FROM trait_embeddings
        WHERE trait_index IN ({placeholders})
    """

    rows = conn.execute(query, list(trait_indices)).fetchall()

    res = {}
    for row in rows:
        trait_index, vector = row
        res[trait_index] = vector

    return res


def _get_trait_efo_mappings(trait_indices: set[int]) -> dict[int, str]:
    """Load best EFO mappings for specified trait indices.

    Args:
        trait_indices: Set of trait indices to get EFO mappings for

    Returns:
        Dict mapping trait_index to best matching EFO ID
    """
    if not trait_indices:
        return {}

    conn = get_vector_store_connection()

    # Get best EFO match for each trait (above threshold)
    placeholders = ",".join("?" * len(trait_indices))
    query = f"""
        SELECT trait_index, efo_id, similarity
        FROM (
            SELECT
                trait_index,
                efo_id,
                similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY trait_index ORDER BY similarity DESC
                ) as rn
            FROM trait_efo_similarity_search
            WHERE trait_index IN ({placeholders})
              AND similarity >= ?
        )
        WHERE rn = 1
    """

    params = list(trait_indices) + [EFO_SIMILARITY_THRESHOLD]
    rows = conn.execute(query, params).fetchall()

    res = {}
    for row in rows:
        trait_index, efo_id, _ = row
        res[trait_index] = efo_id

    return res


# ==== Evidence pair retrieval functions ====


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

    # Build lookup: trait_label (normalized) -> trait_index
    # Also build lookup by trait_id_in_result for backward compatibility
    trait_label_to_index: dict[str, int] = {}
    trait_id_to_index: dict[str, int] = {}
    for row in trait_rows:
        trait_label, trait_index, trait_id = row
        if trait_label:
            # Normalize label for matching (lowercase, strip whitespace)
            normalized_label = trait_label.strip().lower()
            trait_label_to_index[normalized_label] = trait_index
        if trait_id:
            trait_id_to_index[str(trait_id)] = trait_index

    # ---- Build pairs with trait indices ----
    pairs = []
    for r in results:
        exposure = r.get("exposure", "")
        outcome = r.get("outcome", "")
        direction = r.get("direction", "")

        if not (exposure and outcome):
            continue

        # Try to find trait index by ID first, then by label
        exposure_id = r.get("exposure_id", "")
        outcome_id = r.get("outcome_id", "")

        exp_trait_index = None
        out_trait_index = None

        # Try by ID first
        if exposure_id:
            exp_trait_index = trait_id_to_index.get(str(exposure_id))
        if outcome_id:
            out_trait_index = trait_id_to_index.get(str(outcome_id))

        # Fall back to matching by label
        if exp_trait_index is None:
            normalized_exp = exposure.strip().lower()
            exp_trait_index = trait_label_to_index.get(normalized_exp)

        if out_trait_index is None:
            normalized_out = outcome.strip().lower()
            out_trait_index = trait_label_to_index.get(normalized_out)

        pairs.append(
            {
                "exposure": exposure,
                "outcome": outcome,
                "direction": direction,
                "exposure_trait_index": exp_trait_index,
                "outcome_trait_index": out_trait_index,
            }
        )

    return pairs


def _compute_matched_evidence_pairs_tiered(
    query_pairs: list[dict[str, Any]],
    similar_pairs: list[dict[str, Any]],
    match_type_exact: bool,
    match_type_fuzzy: bool,
    match_type_efo: bool,
) -> list[dict[str, Any]]:
    """Compute matched evidence pairs using tiered matching strategy.

    Matches pairs using three tiers (same as preprocessing pipeline):
    1. Exact: Same trait indices for both exposure and outcome
    2. Fuzzy: Cosine similarity >= FUZZY_SIMILARITY_THRESHOLD for both traits
    3. EFO: Same EFO category mapping for both traits

    Args:
        query_pairs: Evidence pairs from query study (with trait indices)
        similar_pairs: Evidence pairs from similar study (with trait indices)
        match_type_exact: Whether exact matches were found during preprocessing
        match_type_fuzzy: Whether fuzzy matches were found during preprocessing
        match_type_efo: Whether EFO matches were found during preprocessing

    Returns:
        List of matched evidence pair dicts with query/similar info and match_type
    """
    matched = []
    matched_query_keys: set[tuple[int | None, int | None]] = set()
    matched_similar_indices: set[int] = set()

    # ---- Collect all trait indices for embedding/EFO lookup ----
    all_trait_indices: set[int] = set()
    for p in query_pairs + similar_pairs:
        exp_idx = p.get("exposure_trait_index")
        out_idx = p.get("outcome_trait_index")
        if exp_idx is not None:
            all_trait_indices.add(exp_idx)
        if out_idx is not None:
            all_trait_indices.add(out_idx)

    # ---- Tier 1: Exact matching (by trait indices) ----
    if match_type_exact:
        similar_by_indices: dict[tuple[int, int], tuple[int, dict]] = {}
        for idx, sp in enumerate(similar_pairs):
            exp_idx = sp.get("exposure_trait_index")
            out_idx = sp.get("outcome_trait_index")
            if exp_idx is not None and out_idx is not None:
                key = (exp_idx, out_idx)
                if key not in similar_by_indices:
                    similar_by_indices[key] = (idx, sp)

        for qp in query_pairs:
            q_exp_idx = qp.get("exposure_trait_index")
            q_out_idx = qp.get("outcome_trait_index")
            if q_exp_idx is None or q_out_idx is None:
                continue

            key = (q_exp_idx, q_out_idx)
            if key in similar_by_indices:
                sim_idx, sp = similar_by_indices[key]
                if sim_idx not in matched_similar_indices:
                    matched.append(
                        {
                            "query_exposure": qp["exposure"],
                            "query_outcome": qp["outcome"],
                            "query_direction": qp.get("direction", ""),
                            "similar_exposure": sp["exposure"],
                            "similar_outcome": sp["outcome"],
                            "similar_direction": sp.get("direction", ""),
                            "match_type": "exact",
                        }
                    )
                    matched_query_keys.add((q_exp_idx, q_out_idx))
                    matched_similar_indices.add(sim_idx)

    # ---- Tier 2: Fuzzy matching (by embedding similarity) ----
    if match_type_fuzzy:
        trait_embeddings = _get_trait_embeddings(all_trait_indices)

        for qp in query_pairs:
            q_exp_idx = qp.get("exposure_trait_index")
            q_out_idx = qp.get("outcome_trait_index")
            if q_exp_idx is None or q_out_idx is None:
                continue
            if (q_exp_idx, q_out_idx) in matched_query_keys:
                continue
            if q_exp_idx not in trait_embeddings:
                continue
            if q_out_idx not in trait_embeddings:
                continue

            q_exp_vec = trait_embeddings[q_exp_idx]
            q_out_vec = trait_embeddings[q_out_idx]

            best_match: tuple[int, dict, float] | None = None
            best_score = 0.0

            for sim_idx, sp in enumerate(similar_pairs):
                if sim_idx in matched_similar_indices:
                    continue

                s_exp_idx = sp.get("exposure_trait_index")
                s_out_idx = sp.get("outcome_trait_index")
                if s_exp_idx is None or s_out_idx is None:
                    continue
                if s_exp_idx not in trait_embeddings:
                    continue
                if s_out_idx not in trait_embeddings:
                    continue

                s_exp_vec = trait_embeddings[s_exp_idx]
                s_out_vec = trait_embeddings[s_out_idx]

                exp_sim = _cosine_similarity(q_exp_vec, s_exp_vec)
                out_sim = _cosine_similarity(q_out_vec, s_out_vec)

                if (
                    exp_sim >= FUZZY_SIMILARITY_THRESHOLD
                    and out_sim >= FUZZY_SIMILARITY_THRESHOLD
                ):
                    combined_score = (exp_sim + out_sim) / 2
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = (sim_idx, sp, combined_score)

            if best_match is not None:
                sim_idx, sp, _ = best_match
                matched.append(
                    {
                        "query_exposure": qp["exposure"],
                        "query_outcome": qp["outcome"],
                        "query_direction": qp.get("direction", ""),
                        "similar_exposure": sp["exposure"],
                        "similar_outcome": sp["outcome"],
                        "similar_direction": sp.get("direction", ""),
                        "match_type": "fuzzy",
                    }
                )
                matched_query_keys.add((q_exp_idx, q_out_idx))
                matched_similar_indices.add(sim_idx)

    # ---- Tier 3: EFO matching (by EFO category) ----
    if match_type_efo:
        trait_efo_map = _get_trait_efo_mappings(all_trait_indices)

        # Build lookup for similar pairs by EFO category
        similar_by_efo: dict[tuple[str, str], list[tuple[int, dict]]] = {}
        for sim_idx, sp in enumerate(similar_pairs):
            if sim_idx in matched_similar_indices:
                continue

            s_exp_idx = sp.get("exposure_trait_index")
            s_out_idx = sp.get("outcome_trait_index")
            if s_exp_idx is None or s_out_idx is None:
                continue
            if (
                s_exp_idx not in trait_efo_map
                or s_out_idx not in trait_efo_map
            ):
                continue

            s_exp_efo = trait_efo_map[s_exp_idx]
            s_out_efo = trait_efo_map[s_out_idx]
            key = (s_exp_efo, s_out_efo)

            if key not in similar_by_efo:
                similar_by_efo[key] = []
            similar_by_efo[key].append((sim_idx, sp))

        for qp in query_pairs:
            q_exp_idx = qp.get("exposure_trait_index")
            q_out_idx = qp.get("outcome_trait_index")
            if q_exp_idx is None or q_out_idx is None:
                continue
            if (q_exp_idx, q_out_idx) in matched_query_keys:
                continue
            if (
                q_exp_idx not in trait_efo_map
                or q_out_idx not in trait_efo_map
            ):
                continue

            q_exp_efo = trait_efo_map[q_exp_idx]
            q_out_efo = trait_efo_map[q_out_idx]
            key = (q_exp_efo, q_out_efo)

            if key in similar_by_efo:
                for sim_idx, sp in similar_by_efo[key]:
                    if sim_idx in matched_similar_indices:
                        continue

                    matched.append(
                        {
                            "query_exposure": qp["exposure"],
                            "query_outcome": qp["outcome"],
                            "query_direction": qp.get("direction", ""),
                            "similar_exposure": sp["exposure"],
                            "similar_outcome": sp["outcome"],
                            "similar_direction": sp.get("direction", ""),
                            "match_type": "efo",
                        }
                    )
                    matched_query_keys.add((q_exp_idx, q_out_idx))
                    matched_similar_indices.add(sim_idx)
                    break  # One match per query pair

    return matched


def get_similar_by_evidence(
    pmid: str,
    model: str,
    limit: int = 10,
    compute_matched_pairs: bool = False,
) -> dict[str, Any] | None:
    """Get similar studies by evidence profile similarity.

    Args:
        pmid: PubMed ID of the query study
        model: Extraction model name
        limit: Maximum number of similar studies to return
        compute_matched_pairs: If True, compute and return the actual matched
            evidence pairs (computationally expensive). If False, return None
            for matched_evidence_pairs field.

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

    # ---- Get query study evidence pairs (once, for matching) if needed ----
    query_pairs = None
    if compute_matched_pairs:
        query_pairs = _get_study_evidence_pairs_with_traits(pmid, model)

    similar_studies = []
    for row in result:
        similar_pmid = row[3]
        match_type_exact = bool(row[7])
        match_type_fuzzy = bool(row[8])
        match_type_efo = bool(row[9])

        # ---- Compute matched pairs if requested ----
        matched_evidence_pairs: list[dict[str, Any]] | None = None
        if compute_matched_pairs and query_pairs is not None:
            # Get similar study evidence pairs with trait indices
            similar_pairs = _get_study_evidence_pairs_with_traits(
                similar_pmid, model
            )

            # Compute matched pairs using tiered matching
            matched_evidence = _compute_matched_evidence_pairs_tiered(
                query_pairs,
                similar_pairs,
                match_type_exact,
                match_type_fuzzy,
                match_type_efo,
            )

            # Format evidence pairs for response
            matched_evidence_pairs = [
                {
                    "query_exposure": m["query_exposure"],
                    "query_outcome": m["query_outcome"],
                    "query_direction": m["query_direction"],
                    "similar_exposure": m["similar_exposure"],
                    "similar_outcome": m["similar_outcome"],
                    "similar_direction": m["similar_direction"],
                    "match_type": m["match_type"],
                }
                for m in matched_evidence
            ]

        similar_studies.append(
            {
                "pmid": similar_pmid,
                "title": row[4],
                "direction_concordance": row[5],
                "matched_pairs": row[6],
                "match_type_exact": match_type_exact,
                "match_type_fuzzy": match_type_fuzzy,
                "match_type_efo": match_type_efo,
                "matched_evidence_pairs": matched_evidence_pairs,
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
