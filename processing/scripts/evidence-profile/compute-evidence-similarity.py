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

import numpy as np
from loguru import logger
from scipy.stats import pearsonr
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

# ==== Constants ====

MIN_MATCHED_PAIRS = 3
MIN_PAIRS_FOR_CORRELATION = 3
TOP_K_DEFAULT = 10


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


def compute_effect_size_similarity(
    matched_pairs: List[Tuple[Dict, Dict]],
) -> Optional[float]:
    """Compute Pearson correlation of harmonized effect sizes (all types).

    Args:
        matched_pairs: List of matched (query, similar) result pairs

    Returns:
        Pearson correlation coefficient, or None if computation not possible
    """
    if len(matched_pairs) < MIN_PAIRS_FOR_CORRELATION:
        return None

    query_effects = [pair[0]["harmonized_effect_size"] for pair in matched_pairs]
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

    query_effects = [
        pair[0]["harmonized_effect_size"] for pair in matched_pairs
    ]
    similar_effects = [
        pair[1]["harmonized_effect_size"] for pair in matched_pairs
    ]

    try:
        correlation, _ = pearsonr(query_effects, similar_effects)
        return float(correlation) if not np.isnan(correlation) else None
    except Exception as e:
        logger.warning(f"Error computing effect size correlation: {e}")
        return None


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
    effect_norm = (effect_similarity + 1) / 2 if effect_similarity is not None else None
    direction_norm = (direction_concordance + 1) / 2
    consistency_norm = (
        (statistical_consistency + 1) / 2
        if statistical_consistency is not None
        else None
    )
    overlap_norm = evidence_overlap

    # Count available metrics
    available_metrics = [
        m for m in [effect_norm, direction_norm, consistency_norm, overlap_norm]
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

    composite_direction = weighted_sum / total_weight if total_weight > 0 else None

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
) -> Optional[Dict]:
    """Compute similarity between two evidence profiles.

    Args:
        query_profile: Query evidence profile
        similar_profile: Similar evidence profile
        min_matched_pairs: Minimum matched pairs required

    Returns:
        Similarity dictionary or None if insufficient matches or
        insufficient non-null metrics for composite scores
    """
    matched_pairs = match_exposure_outcome_pairs(
        query_profile["results"], similar_profile["results"]
    )

    if len(matched_pairs) < min_matched_pairs:
        return None

    # Compute all similarity metrics
    effect_similarity = compute_effect_size_similarity(matched_pairs)
    effect_by_type = compute_effect_size_similarity_by_type(matched_pairs)
    direction_concordance = compute_direction_concordance(matched_pairs)
    statistical_consistency = compute_statistical_consistency(matched_pairs)
    evidence_overlap = compute_evidence_overlap(matched_pairs)
    null_concordance = compute_null_concordance(matched_pairs)

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

    res = {
        "similar_pmid": similar_profile["pmid"],
        "similar_model": similar_profile["model"],
        "similar_title": similar_profile["title"],
        "similar_publication_year": similar_profile.get("publication_year"),
        "matched_pairs": len(matched_pairs),
        "effect_size_similarity": effect_similarity,
        "effect_size_within_type": effect_by_type["within_type"],
        "effect_size_cross_type": effect_by_type["cross_type"],
        "n_within_type_pairs": effect_by_type["n_within_type"],
        "n_cross_type_pairs": effect_by_type["n_cross_type"],
        "direction_concordance": direction_concordance,
        "statistical_consistency": statistical_consistency,
        "evidence_overlap": evidence_overlap,
        "null_concordance": null_concordance,
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
        args_tuple: Tuple containing (query_profile, all_profiles, top_k, min_matched_pairs)

    Returns:
        Dictionary containing similarity results for the query
    """
    query_profile, all_profiles, top_k, min_matched_pairs = args_tuple

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
            query_profile, similar_profile, min_matched_pairs
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
) -> List[Dict]:
    """Compute similarities for a chunk of profiles using multiprocessing.

    Args:
        profiles_chunk: Chunk of query profiles to process
        all_profiles: All profiles for comparison
        top_k: Number of top similar results to keep
        min_matched_pairs: Minimum matched pairs required
        workers: Number of worker processes

    Returns:
        List of similarity records for the chunk
    """
    worker_args = [
        (query_profile, all_profiles, top_k, min_matched_pairs)
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

    start_time = time.time()
    similarity_records = compute_similarities_for_chunk(
        profiles_chunk,
        all_profiles,
        args.top_k,
        args.min_matched_pairs,
        args.workers,
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
