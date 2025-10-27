"""Compute evidence profile similarities between PMID-model combinations.

This script processes evidence profiles to compute pairwise similarities
based on quantitative causal evidence:

1. Loads evidence profiles from JSON output of preprocess-evidence-profiles.py
2. Matches exposure-outcome pairs by exact trait indices
3. Computes four similarity metrics:
   - effect_size_similarity: Pearson correlation of harmonized effect sizes
   - direction_concordance: Proportion of concordant effect directions
   - statistical_consistency: Cohen's kappa for significance patterns
   - evidence_overlap: Jaccard similarity of significant findings
4. Computes two composite scores:
   - composite_equal: Equal-weighted average of all metrics
   - composite_direction: Direction-prioritized weighted average
5. Keeps only top-10 most similar results per combination
6. Only compares within same model (gpt-4 vs gpt-4, etc.)
7. Supports HPC array jobs with multiprocessing

IMPORTANT: Requires minimum 3 matched pairs for similarity computation.
Only intra-model comparisons are performed.

NOTE: This should be run as part of an HPC array job.
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from yiutils.chunking import calculate_chunk_start_end
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "processed" / "evidence-profiles"
OUTPUT_DIR = DATA_DIR / "output" / "evidence-similarities"
DEFAULT_INPUT_FILE = INPUT_DIR / "evidence-profiles.json"
DB_DIR = DATA_DIR / "db"
DEFAULT_DB_PATH = DB_DIR / "vector_store.db"

# ==== Constants ====

MIN_MATCHED_PAIRS = 1
MIN_PAIRS_FOR_CORRELATION = 3
TOP_K_DEFAULT = 50
DEFAULT_FUZZY_THRESHOLD = 0.70


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments with dry_run, array_length,
        and array_id options
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually processing",
    )

    # ---- --array-length ----
    parser.add_argument(
        "--array-length",
        type=int,
        default=20,
        help="Total number of array chunks for parallel processing",
    )

    # ---- --array-id ----
    parser.add_argument(
        "--array-id",
        type=int,
        default=0,
        help="Current array chunk ID (0-based indexing)",
    )

    # ---- --input-file ----
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Path to evidence profiles JSON file",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for similarity files",
    )

    # ---- --top-k ----
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help="Number of top similar results to keep for each combination",
    )

    # ---- --workers ----
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes for multiprocessing",
    )

    # ---- --min-matched-pairs ----
    parser.add_argument(
        "--min-matched-pairs",
        type=int,
        default=MIN_MATCHED_PAIRS,
        help="Minimum number of matched pairs required for similarity",
    )

    # ---- --use-fuzzy-matching ----
    parser.add_argument(
        "--use-fuzzy-matching",
        action="store_true",
        help="Use fuzzy trait matching based on embedding similarity",
    )

    # ---- --fuzzy-threshold ----
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="Similarity threshold for fuzzy matching (default: 0.95)",
    )

    # ---- --database-path ----
    parser.add_argument(
        "--database-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to vector_store.db for trait embeddings",
    )

    # ---- --use-efo-matching ----
    parser.add_argument(
        "--use-efo-matching",
        action="store_true",
        help="Use EFO-based category matching as third tier",
    )

    # ---- --efo-similarity-threshold ----
    parser.add_argument(
        "--efo-similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum trait-EFO similarity for category mapping (default: 0.5)",
    )

    return parser.parse_args()


def load_evidence_profiles(input_file: Path) -> List[Dict]:
    """Load evidence profiles from JSON file.

    Args:
        input_file: Path to evidence profiles JSON file

    Returns:
        List of evidence profile dictionaries
    """
    logger.info(f"Loading evidence profiles from: {input_file}")

    with input_file.open("r") as f:
        profiles = json.load(f)

    logger.info(f"Loaded {len(profiles)} evidence profiles")
    return profiles


def load_trait_embeddings(db_path: Path) -> Dict[int, np.ndarray]:
    """Load trait embeddings from vector_store.db.

    Args:
        db_path: Path to vector_store.db database

    Returns:
        Dictionary mapping trait_index to embedding vector
    """
    logger.info(f"Loading trait embeddings from: {db_path}")
    conn = duckdb.connect(str(db_path), read_only=True)

    query = "SELECT trait_index, vector FROM trait_embeddings"
    results = conn.execute(query).fetchall()

    trait_embeddings = {row[0]: np.array(row[1]) for row in results}

    logger.info(f"Loaded {len(trait_embeddings)} trait embeddings")
    conn.close()
    return trait_embeddings


def load_trait_efo_mappings(
    db_path: Path, min_similarity: float = 0.5
) -> Dict[int, str]:
    """Load top EFO term for each trait index using in-memory computation.

    For each trait, selects the EFO term with highest similarity score
    above the minimum threshold. Computes similarities in-memory using
    numpy to avoid expensive database CROSS JOIN operations.

    Args:
        db_path: Path to vector_store.db
        min_similarity: Minimum similarity threshold for EFO mapping

    Returns:
        Dictionary mapping trait_index to best EFO ID
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info(
        f"Loading trait-EFO mappings from: {db_path} "
        f"(min_similarity={min_similarity})"
    )
    conn = duckdb.connect(str(db_path), read_only=True)

    logger.info("Loading trait vectors...")
    trait_query = """
    SELECT trait_index, vector
    FROM trait_embeddings
    ORDER BY trait_index
    """
    trait_results = conn.execute(trait_query).fetchall()
    trait_indices = [r[0] for r in trait_results]
    trait_vectors = np.array([r[1] for r in trait_results])

    logger.info("Loading EFO vectors...")
    efo_query = """
    SELECT id, vector
    FROM efo_embeddings
    ORDER BY id
    """
    efo_results = conn.execute(efo_query).fetchall()
    efo_ids = [r[0] for r in efo_results]
    efo_vectors = np.array([r[1] for r in efo_results])

    conn.close()

    logger.info(
        f"Computing similarities for {len(trait_indices)} traits "
        f"and {len(efo_ids)} EFO terms..."
    )

    similarities = cosine_similarity(trait_vectors, efo_vectors)

    trait_efo_map = {}
    for i, trait_idx in enumerate(trait_indices):
        trait_sims = similarities[i]
        max_idx = np.argmax(trait_sims)
        max_sim = trait_sims[max_idx]

        if max_sim >= min_similarity:
            trait_efo_map[trait_idx] = efo_ids[max_idx]

    logger.info(
        f"Loaded EFO mappings for {len(trait_efo_map)} traits "
        f"(out of {len(trait_indices)} total)"
    )

    return trait_efo_map


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    dot_product = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1))
    norm2 = float(np.linalg.norm(vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    res = dot_product / (norm1 * norm2)
    return res


def match_exposure_outcome_pairs_fuzzy(
    query_results: List[Dict],
    similar_results: List[Dict],
    trait_embeddings: Dict[int, np.ndarray],
    threshold: float = 0.95,
) -> List[Tuple[Dict, Dict]]:
    """Match exposure-outcome pairs using fuzzy trait matching.

    Pairs are matched if both exposure and outcome traits have
    similarity >= threshold.

    Args:
        query_results: Results from query combination
        similar_results: Results from similar combination
        trait_embeddings: Dictionary of trait embeddings
        threshold: Similarity threshold for matching (default: 0.95)

    Returns:
        List of matched pairs as (query_result, similar_result) tuples
    """
    matched_pairs = []
    used_similar_indices = set()

    for query_result in query_results:
        q_exp_idx = query_result["exposure_trait_index"]
        q_out_idx = query_result["outcome_trait_index"]

        if (
            q_exp_idx not in trait_embeddings
            or q_out_idx not in trait_embeddings
        ):
            continue

        q_exp_vec = trait_embeddings[q_exp_idx]
        q_out_vec = trait_embeddings[q_out_idx]

        best_match = None
        best_score = 0.0

        for sim_idx, similar_result in enumerate(similar_results):
            if sim_idx in used_similar_indices:
                continue

            s_exp_idx = similar_result["exposure_trait_index"]
            s_out_idx = similar_result["outcome_trait_index"]

            if (
                s_exp_idx not in trait_embeddings
                or s_out_idx not in trait_embeddings
            ):
                continue

            s_exp_vec = trait_embeddings[s_exp_idx]
            s_out_vec = trait_embeddings[s_out_idx]

            exp_sim = cosine_similarity(q_exp_vec, s_exp_vec)
            out_sim = cosine_similarity(q_out_vec, s_out_vec)

            if exp_sim >= threshold and out_sim >= threshold:
                combined_score = (exp_sim + out_sim) / 2
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (sim_idx, similar_result)

        if best_match is not None:
            matched_pairs.append((query_result, best_match[1]))
            used_similar_indices.add(best_match[0])

    return matched_pairs


def match_exposure_outcome_pairs(
    query_results: List[Dict], similar_results: List[Dict]
) -> List[Tuple[Dict, Dict]]:
    """Match exposure-outcome pairs by exact trait indices.

    Args:
        query_results: Results from query combination
        similar_results: Results from similar combination

    Returns:
        List of matched pairs as (query_result, similar_result) tuples
    """
    similar_dict = {}
    for result in similar_results:
        key = (result["exposure_trait_index"], result["outcome_trait_index"])
        similar_dict[key] = result

    matched_pairs = []
    for query_result in query_results:
        key = (
            query_result["exposure_trait_index"],
            query_result["outcome_trait_index"],
        )
        if key in similar_dict:
            matched_pairs.append((query_result, similar_dict[key]))

    return matched_pairs


def match_exposure_outcome_pairs_efo(
    query_results: List[Dict],
    similar_results: List[Dict],
    trait_efo_map: Dict[int, str],
) -> List[Tuple[Dict, Dict]]:
    """Match exposure-outcome pairs using EFO categories.

    Pairs are matched if both exposure and outcome traits map to
    the same EFO terms (category-level matching).

    Args:
        query_results: Results from query combination
        similar_results: Results from similar combination
        trait_efo_map: Dictionary mapping trait_index to EFO ID

    Returns:
        List of matched pairs as (query_result, similar_result) tuples
    """
    similar_dict = {}
    for result in similar_results:
        exp_idx = result["exposure_trait_index"]
        out_idx = result["outcome_trait_index"]

        if exp_idx not in trait_efo_map or out_idx not in trait_efo_map:
            continue

        exp_efo = trait_efo_map[exp_idx]
        out_efo = trait_efo_map[out_idx]
        key = (exp_efo, out_efo)

        if key not in similar_dict:
            similar_dict[key] = []
        similar_dict[key].append(result)

    matched_pairs = []
    for query_result in query_results:
        q_exp_idx = query_result["exposure_trait_index"]
        q_out_idx = query_result["outcome_trait_index"]

        if q_exp_idx not in trait_efo_map or q_out_idx not in trait_efo_map:
            continue

        q_exp_efo = trait_efo_map[q_exp_idx]
        q_out_efo = trait_efo_map[q_out_idx]
        key = (q_exp_efo, q_out_efo)

        if key in similar_dict:
            for similar_result in similar_dict[key]:
                matched_pairs.append((query_result, similar_result))

    return matched_pairs


def match_exposure_outcome_pairs_tiered(
    query_results: List[Dict],
    similar_results: List[Dict],
    trait_embeddings: Optional[Dict[int, np.ndarray]] = None,
    trait_efo_map: Optional[Dict[int, str]] = None,
    fuzzy_threshold: float = 0.80,
) -> List[Tuple[Dict, Dict, str]]:
    """Match pairs using three-tier approach with match type tracking.

    Tries matching in order of precision:
    1. Exact: Same trait indices
    2. Fuzzy: Similar trait embeddings (cosine >= threshold)
    3. EFO: Same EFO categories

    Each query result is matched at most once per similar result,
    with preference given to higher-precision matches.

    Args:
        query_results: Results from query combination
        similar_results: Results from similar combination
        trait_embeddings: Dictionary of trait embeddings for fuzzy matching
        trait_efo_map: Dictionary of trait-to-EFO mappings for category matching
        fuzzy_threshold: Similarity threshold for fuzzy matching

    Returns:
        List of (query_result, similar_result, match_type) tuples
        match_type in ["exact", "fuzzy", "efo"]
    """
    matched_pairs_with_type = []
    matched_query_indices = set()

    exact_pairs = match_exposure_outcome_pairs(query_results, similar_results)
    for q_res, s_res in exact_pairs:
        q_idx = (
            q_res["exposure_trait_index"],
            q_res["outcome_trait_index"],
        )
        s_idx = (
            s_res["exposure_trait_index"],
            s_res["outcome_trait_index"],
        )
        pair_key = (q_idx, s_idx)

        if pair_key not in matched_query_indices:
            matched_pairs_with_type.append((q_res, s_res, "exact"))
            matched_query_indices.add(pair_key)

    if trait_embeddings is not None:
        unmatched_query = [
            r
            for r in query_results
            if (
                r["exposure_trait_index"],
                r["outcome_trait_index"],
            )
            not in {
                (m[0]["exposure_trait_index"], m[0]["outcome_trait_index"])
                for m in matched_pairs_with_type
            }
        ]
        unmatched_similar = similar_results

        fuzzy_pairs = match_exposure_outcome_pairs_fuzzy(
            unmatched_query,
            unmatched_similar,
            trait_embeddings,
            fuzzy_threshold,
        )

        for q_res, s_res in fuzzy_pairs:
            q_idx = (
                q_res["exposure_trait_index"],
                q_res["outcome_trait_index"],
            )
            s_idx = (
                s_res["exposure_trait_index"],
                s_res["outcome_trait_index"],
            )
            pair_key = (q_idx, s_idx)

            if pair_key not in matched_query_indices:
                matched_pairs_with_type.append((q_res, s_res, "fuzzy"))
                matched_query_indices.add(pair_key)

    if trait_efo_map is not None:
        unmatched_query = [
            r
            for r in query_results
            if (
                r["exposure_trait_index"],
                r["outcome_trait_index"],
            )
            not in {
                (m[0]["exposure_trait_index"], m[0]["outcome_trait_index"])
                for m in matched_pairs_with_type
            }
        ]
        unmatched_similar = similar_results

        efo_pairs = match_exposure_outcome_pairs_efo(
            unmatched_query, unmatched_similar, trait_efo_map
        )

        for q_res, s_res in efo_pairs:
            q_idx = (
                q_res["exposure_trait_index"],
                q_res["outcome_trait_index"],
            )
            s_idx = (
                s_res["exposure_trait_index"],
                s_res["outcome_trait_index"],
            )
            pair_key = (q_idx, s_idx)

            if pair_key not in matched_query_indices:
                matched_pairs_with_type.append((q_res, s_res, "efo"))
                matched_query_indices.add(pair_key)

    return matched_pairs_with_type


def compute_effect_size_similarity(
    matched_pairs: List[Tuple[Dict, Dict]],
) -> Optional[float]:
    """Compute Pearson correlation of harmonized effect sizes (all types).

    WARNING: Low data availability (~3.82% of comparisons). This metric
    requires harmonized effect sizes from abstract extraction, which is
    limited by abstract-only access and extraction success rates.

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Pearson correlation coefficient, or None if computation not possible
    """
    if len(matched_pairs) < MIN_PAIRS_FOR_CORRELATION:
        return None

    query_effects = [
        pair[0]["harmonized_effect_size"] for pair in matched_pairs
    ]
    similar_effects = [
        pair[1]["harmonized_effect_size"] for pair in matched_pairs
    ]

    if len(set(query_effects)) < 2 or len(set(similar_effects)) < 2:
        return None

    try:
        correlation, _ = pearsonr(query_effects, similar_effects)
        return float(correlation) if not np.isnan(correlation) else None
    except Exception as e:
        logger.warning(f"Error computing effect size correlation: {e}")
        return None


def compute_effect_size_similarity_by_type(
    matched_pairs: List[Tuple[Dict, Dict]],
) -> Dict[str, Optional[float]]:
    """Compute effect size similarities stratified by effect type.

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Dictionary with 'within_type' and 'cross_type' similarity scores
    """
    # Separate within-type and cross-type pairs
    within_type_pairs = []
    cross_type_pairs = []

    for query_result, similar_result in matched_pairs:
        query_type = query_result["effect_size_type"]
        similar_type = similar_result["effect_size_type"]

        if query_type == similar_type:
            within_type_pairs.append((query_result, similar_result))
        else:
            cross_type_pairs.append((query_result, similar_result))

    # Compute within-type similarity
    within_type_sim = None
    if len(within_type_pairs) >= MIN_PAIRS_FOR_CORRELATION:
        within_type_sim = compute_effect_size_similarity(within_type_pairs)

    # Compute cross-type similarity
    cross_type_sim = None
    if len(cross_type_pairs) >= MIN_PAIRS_FOR_CORRELATION:
        cross_type_sim = compute_effect_size_similarity(cross_type_pairs)

    return {
        "within_type": within_type_sim,
        "cross_type": cross_type_sim,
        "n_within_type": len(within_type_pairs),
        "n_cross_type": len(cross_type_pairs),
    }


def compute_direction_concordance(
    matched_pairs: List[Tuple[Dict, Dict]],
) -> float:
    """Compute proportion of concordant effect directions.

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Direction concordance score (-1.0 to 1.0)
    """
    if not matched_pairs:
        return 0.0

    concordant = 0
    discordant = 0

    for query_result, similar_result in matched_pairs:
        query_dir = query_result["direction"]
        similar_dir = similar_result["direction"]

        if query_dir == 0 or similar_dir == 0:
            continue

        if query_dir == similar_dir:
            concordant += 1
        else:
            discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.0

    res = (concordant - discordant) / total
    return res


def compute_statistical_consistency(
    matched_pairs: List[Tuple[Dict, Dict]],
) -> Optional[float]:
    """Compute Cohen's kappa for significance pattern agreement.

    WARNING: This metric has ~0.27% data availability due to matching sparsity.
    82% of comparisons have only 1 matched trait pair, but this metric requires
    >= 3 matched pairs. Expected to return None in >99% of cases.

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Cohen's kappa coefficient, or None if computation not possible
    """
    if len(matched_pairs) < MIN_PAIRS_FOR_CORRELATION:
        return None

    query_significance = [pair[0]["is_significant"] for pair in matched_pairs]
    similar_significance = [
        pair[1]["is_significant"] for pair in matched_pairs
    ]

    if len(set(query_significance)) < 2 or len(set(similar_significance)) < 2:
        return None

    try:
        kappa = cohen_kappa_score(query_significance, similar_significance)
        return float(kappa) if not np.isnan(kappa) else None
    except Exception as e:
        logger.warning(f"Error computing statistical consistency: {e}")
        return None


def compute_null_concordance(matched_pairs: List[Tuple[Dict, Dict]]) -> float:
    """Compute concordance of null/non-significant results.

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Proportion of pairs where both are non-significant (0.0 to 1.0)
    """
    if not matched_pairs:
        return 0.0

    both_null = 0
    total = len(matched_pairs)

    for query_result, similar_result in matched_pairs:
        if (
            not query_result["is_significant"]
            and not similar_result["is_significant"]
        ):
            both_null += 1

    res = both_null / total if total > 0 else 0.0
    return res


def compute_evidence_overlap(matched_pairs: List[Tuple[Dict, Dict]]) -> float:
    """Compute Jaccard similarity of significant findings.

    Edge case handling:
    - If both studies have zero significant findings, returns 0.0
      (not 1.0, to avoid inflating similarity for underpowered studies)
    - If only one study has significant findings, returns 0.0

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Jaccard similarity coefficient (0.0 to 1.0)
    """
    if not matched_pairs:
        return 0.0

    query_significant = set()
    similar_significant = set()

    for query_result, similar_result in matched_pairs:
        key = (
            query_result["exposure_trait_index"],
            query_result["outcome_trait_index"],
        )

        if query_result["is_significant"]:
            query_significant.add(key)
        if similar_result["is_significant"]:
            similar_significant.add(key)

    if not query_significant and not similar_significant:
        return 0.0

    if not query_significant or not similar_significant:
        return 0.0

    intersection = len(query_significant & similar_significant)
    union = len(query_significant | similar_significant)

    res = intersection / union if union > 0 else 0.0
    return res


def compute_precision_concordance(
    matched_pairs: List[Tuple[Dict, Dict]],
) -> Optional[float]:
    """Compute similarity of effect estimate precision using CI widths.

    WARNING: Very low data availability (~3.33% of comparisons). This metric
    requires confidence intervals from abstract extraction AND at least 3
    matched trait pairs for correlation. The combination of matching sparsity
    (82% of comparisons have only 1 matched pair) and CI extraction limitations
    results in extremely limited applicability.

    Measures how similar two studies are in terms of the precision
    of their effect estimates by comparing confidence interval widths.
    Uses Spearman correlation of log-transformed CI widths to handle
    the skewed distribution typical of effect size uncertainty.

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Spearman correlation of log CI widths, or None if insufficient
        valid pairs for computation
    """
    if len(matched_pairs) < MIN_PAIRS_FOR_CORRELATION:
        return None

    query_widths = []
    similar_widths = []

    for query_result, similar_result in matched_pairs:
        query_ci_lower = query_result.get("ci_lower")
        query_ci_upper = query_result.get("ci_upper")
        similar_ci_lower = similar_result.get("ci_lower")
        similar_ci_upper = similar_result.get("ci_upper")

        if (
            query_ci_lower is not None
            and query_ci_upper is not None
            and similar_ci_lower is not None
            and similar_ci_upper is not None
        ):
            query_width = abs(query_ci_upper - query_ci_lower)
            similar_width = abs(similar_ci_upper - similar_ci_lower)

            if query_width > 0 and similar_width > 0:
                query_widths.append(query_width)
                similar_widths.append(similar_width)

    if len(query_widths) < MIN_PAIRS_FOR_CORRELATION:
        return None

    try:
        log_query_widths = np.log(query_widths)
        log_similar_widths = np.log(similar_widths)

        if np.any(np.isnan(log_query_widths)) or np.any(
            np.isnan(log_similar_widths)
        ):
            return None

        correlation, _ = spearmanr(log_query_widths, log_similar_widths)
        return float(correlation) if not np.isnan(correlation) else None
    except Exception as e:
        logger.warning(f"Error computing precision concordance: {e}")
        return None


def compute_composite_scores(
    effect_similarity: Optional[float],
    direction_concordance: float,
    statistical_consistency: Optional[float],
    evidence_overlap: float,
    query_completeness: float,
    similar_completeness: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute composite similarity scores with consistent normalization.

    All metrics are normalized to [0, 1] scale before combination:
    - Effect similarity: Pearson r in [-1, 1] → normalized to [0, 1]
    - Direction concordance: in [-1, 1] → normalized to [0, 1]
    - Statistical consistency: Cohen's kappa in [-1, 1] → normalized to [0, 1]
    - Evidence overlap: Jaccard in [0, 1] → already normalized

    Quality weighting: Composite scores are multiplied by
    min(query_completeness, similar_completeness) to down-weight
    comparisons involving low-quality data.

    Missing data handling: If critical metrics are None (effect_similarity
    or statistical_consistency), those components are excluded from the
    composite score calculation rather than being imputed.

    Args:
        effect_similarity: Effect size similarity (nullable)
        direction_concordance: Direction concordance score
        statistical_consistency: Statistical consistency (nullable)
        evidence_overlap: Evidence overlap score
        query_completeness: Data completeness of query study [0, 1]
        similar_completeness: Data completeness of similar study [0, 1]

    Returns:
        Tuple of (composite_equal, composite_direction), both nullable
        Returns (None, None) if insufficient non-null metrics available
    """
    # Normalize all metrics to [0, 1]
    effect_norm = (
        (effect_similarity + 1) / 2 if effect_similarity is not None else None
    )
    direction_norm = (direction_concordance + 1) / 2
    consistency_norm = (
        (statistical_consistency + 1) / 2
        if statistical_consistency is not None
        else None
    )
    overlap_norm = evidence_overlap

    # Count available metrics
    available_metrics = [
        m
        for m in [effect_norm, direction_norm, consistency_norm, overlap_norm]
        if m is not None
    ]

    # Require at least 2 non-null metrics including direction
    if len(available_metrics) < 2:
        return None, None

    # Equal weighting: average of available metrics
    composite_equal = sum(available_metrics) / len(available_metrics)

    # Direction-prioritized weighting
    # Adjust weights based on available metrics
    total_weight = 0.0
    weighted_sum = 0.0

    # Direction is always available (required)
    weighted_sum += 0.50 * direction_norm
    total_weight += 0.50

    if effect_norm is not None:
        weighted_sum += 0.20 * effect_norm
        total_weight += 0.20

    if consistency_norm is not None:
        weighted_sum += 0.15 * consistency_norm
        total_weight += 0.15

    # Evidence overlap always available
    weighted_sum += 0.15 * overlap_norm
    total_weight += 0.15

    composite_direction = (
        weighted_sum / total_weight if total_weight > 0 else None
    )

    # Apply quality weighting
    quality_weight = min(query_completeness, similar_completeness)

    if composite_equal is not None:
        composite_equal *= quality_weight
    if composite_direction is not None:
        composite_direction *= quality_weight

    return composite_equal, composite_direction


def compute_pairwise_similarity(
    query_profile: Dict,
    similar_profile: Dict,
    min_matched_pairs: int,
    trait_embeddings: Optional[Dict[int, np.ndarray]] = None,
    trait_efo_map: Optional[Dict[int, str]] = None,
    fuzzy_threshold: float = 0.80,
) -> Optional[Dict]:
    """Compute similarity between two evidence profiles.

    Args:
        query_profile: Query evidence profile
        similar_profile: Similar evidence profile
        min_matched_pairs: Minimum matched pairs required
        trait_embeddings: Dictionary of trait embeddings for fuzzy matching
        trait_efo_map: Dictionary of trait-to-EFO mappings for category matching
        fuzzy_threshold: Similarity threshold for fuzzy matching

    Returns:
        Similarity dictionary or None if insufficient matches or
        insufficient non-null metrics for composite scores
    """
    matched_pairs_with_type = match_exposure_outcome_pairs_tiered(
        query_profile["results"],
        similar_profile["results"],
        trait_embeddings,
        trait_efo_map,
        fuzzy_threshold,
    )

    if len(matched_pairs_with_type) < min_matched_pairs:
        return None

    matched_pairs = [(q, s) for q, s, _ in matched_pairs_with_type]
    match_types = [mt for _, _, mt in matched_pairs_with_type]

    # Compute all similarity metrics
    effect_similarity = compute_effect_size_similarity(matched_pairs)
    effect_by_type = compute_effect_size_similarity_by_type(matched_pairs)
    direction_concordance = compute_direction_concordance(matched_pairs)
    statistical_consistency = compute_statistical_consistency(matched_pairs)
    evidence_overlap = compute_evidence_overlap(matched_pairs)
    null_concordance = compute_null_concordance(matched_pairs)
    precision_concordance = compute_precision_concordance(matched_pairs)

    # Get data completeness
    query_completeness = query_profile["data_completeness"]
    similar_completeness = similar_profile["data_completeness"]

    # Compute composite scores with quality weighting
    composite_equal, composite_direction = compute_composite_scores(
        effect_similarity,
        direction_concordance,
        statistical_consistency,
        evidence_overlap,
        query_completeness,
        similar_completeness,
    )

    # Exclude if insufficient metrics for composite scores
    if composite_equal is None or composite_direction is None:
        return None

    match_type_counts = {
        "exact": match_types.count("exact"),
        "fuzzy": match_types.count("fuzzy"),
        "efo": match_types.count("efo"),
    }

    res = {
        "similar_pmid": similar_profile["pmid"],
        "similar_model": similar_profile["model"],
        "similar_title": similar_profile["title"],
        "similar_publication_year": similar_profile.get("publication_year"),
        "matched_pairs": len(matched_pairs),
        "match_type_exact": match_type_counts["exact"],
        "match_type_fuzzy": match_type_counts["fuzzy"],
        "match_type_efo": match_type_counts["efo"],
        "effect_size_similarity": effect_similarity,
        "effect_size_within_type": effect_by_type["within_type"],
        "effect_size_cross_type": effect_by_type["cross_type"],
        "n_within_type_pairs": effect_by_type["n_within_type"],
        "n_cross_type_pairs": effect_by_type["n_cross_type"],
        "direction_concordance": direction_concordance,
        "statistical_consistency": statistical_consistency,
        "evidence_overlap": evidence_overlap,
        "null_concordance": null_concordance,
        "precision_concordance": precision_concordance,
        "composite_similarity_equal": composite_equal,
        "composite_similarity_direction": composite_direction,
        "query_result_count": query_profile["result_count"],
        "similar_result_count": similar_profile["result_count"],
        "query_completeness": query_completeness,
        "similar_completeness": similar_completeness,
    }

    return res


def compute_similarities_for_single_query(args_tuple) -> Dict:
    """Worker function to compute similarities for a single query combination.

    This function is designed to work with multiprocessing.Pool.
    Only compares with results from the same model as the query.

    Args:
        args_tuple: Tuple containing (query_profile, all_profiles, top_k,
                    min_matched_pairs, trait_embeddings, trait_efo_map,
                    fuzzy_threshold)

    Returns:
        Dictionary containing similarity results for the query
    """
    (
        query_profile,
        all_profiles,
        top_k,
        min_matched_pairs,
        trait_embeddings,
        trait_efo_map,
        fuzzy_threshold,
    ) = args_tuple

    similarities = []

    same_model_profiles = [
        p for p in all_profiles if p["model"] == query_profile["model"]
    ]

    for similar_profile in same_model_profiles:
        if (
            query_profile["pmid"] == similar_profile["pmid"]
            and query_profile["model"] == similar_profile["model"]
        ):
            continue

        similarity = compute_pairwise_similarity(
            query_profile,
            similar_profile,
            min_matched_pairs,
            trait_embeddings,
            trait_efo_map,
            fuzzy_threshold,
        )

        if similarity is not None:
            similarities.append(similarity)

    similarities.sort(
        key=lambda x: x["composite_similarity_direction"], reverse=True
    )
    top_similarities = similarities[:top_k]

    res = {
        "query_pmid": query_profile["pmid"],
        "query_model": query_profile["model"],
        "query_title": query_profile["title"],
        "query_publication_year": query_profile.get("publication_year"),
        "query_result_count": query_profile["result_count"],
        "complete_result_count": query_profile["complete_result_count"],
        "data_completeness": query_profile["data_completeness"],
        "top_similarities": top_similarities,
    }

    return res


def compute_similarities_for_chunk(
    profiles_chunk: List[Dict],
    all_profiles: List[Dict],
    top_k: int,
    min_matched_pairs: int,
    workers: int,
    trait_embeddings: Optional[Dict[int, np.ndarray]] = None,
    trait_efo_map: Optional[Dict[int, str]] = None,
    fuzzy_threshold: float = 0.80,
) -> List[Dict]:
    """Compute similarities for a chunk of profiles using multiprocessing.

    Args:
        profiles_chunk: Chunk of query profiles to process
        all_profiles: All profiles for comparison
        top_k: Number of top similar results to keep
        min_matched_pairs: Minimum matched pairs required
        workers: Number of worker processes
        trait_embeddings: Optional trait embeddings for fuzzy matching
        trait_efo_map: Optional trait-to-EFO mappings for category matching
        fuzzy_threshold: Similarity threshold for fuzzy matching

    Returns:
        List of similarity records for the chunk
    """
    worker_args = [
        (
            query_profile,
            all_profiles,
            top_k,
            min_matched_pairs,
            trait_embeddings,
            trait_efo_map,
            fuzzy_threshold,
        )
        for query_profile in profiles_chunk
    ]

    logger.info(
        f"Starting multiprocessing with {workers} workers for {len(profiles_chunk)} queries"
    )

    with multiprocessing.Pool(processes=workers) as pool:
        similarity_records = list(
            tqdm(
                pool.imap(compute_similarities_for_single_query, worker_args),
                total=len(worker_args),
                desc="Computing similarities",
            )
        )

    logger.info(
        f"Completed similarity computation for {len(similarity_records)} queries"
    )

    total_exact = 0
    total_fuzzy = 0
    total_efo = 0
    total_matched_pairs = 0
    total_comparisons = 0

    for record in similarity_records:
        for sim in record["top_similarities"]:
            total_comparisons += 1
            total_exact += sim["match_type_exact"]
            total_fuzzy += sim["match_type_fuzzy"]
            total_efo += sim["match_type_efo"]
            total_matched_pairs += sim["matched_pairs"]

    if total_matched_pairs > 0:
        exact_pct = (total_exact / total_matched_pairs) * 100
        fuzzy_pct = (total_fuzzy / total_matched_pairs) * 100
        efo_pct = (total_efo / total_matched_pairs) * 100

        logger.info(
            f"Match type distribution across {total_comparisons} comparisons:"
        )
        logger.info(f"  Total matched pairs: {total_matched_pairs}")
        logger.info(f"  Exact matches: {total_exact} ({exact_pct:.2f}%)")
        logger.info(f"  Fuzzy matches: {total_fuzzy} ({fuzzy_pct:.2f}%)")
        logger.info(f"  EFO matches: {total_efo} ({efo_pct:.2f}%)")

        if trait_efo_map is not None and total_efo > 0:
            logger.info(
                f"  EFO contribution rate: {efo_pct:.2f}% "
                f"(expected: 1-2%, alert if >5%)"
            )
            if efo_pct > 5.0:
                logger.warning(
                    f"HIGH EFO CONTRIBUTION: {efo_pct:.2f}% exceeds 5% threshold"
                )

    return similarity_records


def main():
    """Main function to compute evidence profile similarities.

    This function:
    1. Loads evidence profiles from JSON
    2. Processes them in chunks for parallel processing
    3. Computes evidence similarities for the assigned chunk
    4. Keeps only the top-k most similar results for each combination
    """
    args = make_args()

    logger.info("Checking file paths and basic setup...")

    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    logger.info(f"✓ Input file exists: {input_file}")

    if args.dry_run:
        logger.info("Dry run completed. Exiting without processing.")
        return 0

    all_profiles = load_evidence_profiles(input_file)

    total_profiles = len(all_profiles)
    start_idx, end_idx = calculate_chunk_start_end(
        chunk_id=args.array_id,
        num_chunks=args.array_length,
        data_length=total_profiles,
    )

    if start_idx is None or end_idx is None:
        logger.warning(
            f"Chunk {args.array_id} is out of range. No profiles to process."
        )
        return 0

    logger.info(
        f"Processing chunk {args.array_id}/{args.array_length}: "
        f"profiles [{start_idx} to {end_idx}) (total: {end_idx - start_idx})"
    )

    profiles_chunk = all_profiles[start_idx:end_idx]

    logger.info(f"Chunk contains {len(profiles_chunk)} profiles to process.")
    logger.info(f"Using {args.workers} worker processes for multiprocessing")
    logger.info(f"Minimum matched pairs required: {args.min_matched_pairs}")

    trait_embeddings = None
    trait_efo_map = None
    db_path = Path(args.database_path)

    if args.use_fuzzy_matching or args.use_efo_matching:
        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            logger.error("Fuzzy and EFO matching require vector_store.db")
            return 1

    if args.use_fuzzy_matching:
        logger.info("Fuzzy matching enabled")
        logger.info(f"Fuzzy matching threshold: {args.fuzzy_threshold}")
        trait_embeddings = load_trait_embeddings(db_path)
    else:
        logger.info("Fuzzy matching disabled")

    if args.use_efo_matching:
        logger.info("EFO category matching enabled")
        logger.info(
            f"EFO similarity threshold: {args.efo_similarity_threshold}"
        )
        trait_efo_map = load_trait_efo_mappings(
            db_path, args.efo_similarity_threshold
        )
    else:
        logger.info("EFO matching disabled")

    if not args.use_fuzzy_matching and not args.use_efo_matching:
        logger.info("Using exact trait index matching only")

    start_time = time.time()
    similarity_records = compute_similarities_for_chunk(
        profiles_chunk,
        all_profiles,
        args.top_k,
        args.min_matched_pairs,
        args.workers,
        trait_embeddings,
        trait_efo_map,
        args.fuzzy_threshold,
    )

    processing_time = time.time() - start_time
    logger.info(
        f"Similarity computation completed in {processing_time:.2f} seconds"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"evidence_similarities_chunk_{args.array_id}.json"
    )

    logger.info(f"Writing {len(similarity_records)} records to: {output_path}")
    with output_path.open("w") as f:
        json.dump(similarity_records, f, indent=2)

    logger.info("Processing completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
