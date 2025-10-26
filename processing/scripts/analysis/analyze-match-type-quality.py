"""Analyze quality stratification by match type.

This script analyzes direction concordance and other similarity metrics
stratified by match type (exact, fuzzy, EFO) to understand if match
quality varies by matching strategy.

Quality indicators assessed:
- Direction concordance by match type
- Effect size similarity by match type
- Statistical consistency by match type
- Match type distribution in high vs low similarity pairs

Expected patterns:
- Exact matches: Highest quality (same traits)
- Fuzzy matches: Medium quality (similar traits)
- EFO matches: Lower quality (category-level matching)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output" / "evidence-similarities"


def load_similarity_chunks(output_dir: Path) -> List[Dict]:
    """Load all similarity chunk files.

    Args:
        output_dir: Directory containing similarity chunk files

    Returns:
        List of all similarity records from all chunks
    """
    chunk_files = sorted(output_dir.glob("evidence_similarities_chunk_*.json"))
    logger.info(f"Found {len(chunk_files)} similarity chunk files")

    all_records = []
    for chunk_file in chunk_files:
        with chunk_file.open("r") as f:
            chunk_data = json.load(f)
            all_records.extend(chunk_data)

    logger.info(f"Loaded {len(all_records)} similarity records")
    return all_records


def stratify_by_match_type(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Stratify similarity comparisons by predominant match type.

    A comparison is classified by its predominant match type
    (the type with the most matched pairs).

    Args:
        records: List of similarity records

    Returns:
        Dictionary mapping match type to list of similarity comparisons
    """
    stratified = {"exact": [], "fuzzy": [], "efo": [], "mixed": []}

    for record in records:
        for sim in record["top_similarities"]:
            exact_count = sim["match_type_exact"]
            fuzzy_count = sim["match_type_fuzzy"]
            efo_count = sim["match_type_efo"]

            max_count = max(exact_count, fuzzy_count, efo_count)

            if max_count == 0:
                continue

            type_counts = Counter(
                {
                    "exact": exact_count,
                    "fuzzy": fuzzy_count,
                    "efo": efo_count,
                }
            )
            most_common = type_counts.most_common(2)

            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                stratified["mixed"].append(sim)
            else:
                predominant_type = most_common[0][0]
                stratified[predominant_type].append(sim)

    logger.info("\nMatch type stratification:")
    for match_type, sims in stratified.items():
        logger.info(f"  {match_type}: {len(sims)} comparisons")

    return stratified


def compute_metric_statistics(
    similarities: List[Dict], metric_name: str
) -> Dict:
    """Compute statistics for a similarity metric.

    Args:
        similarities: List of similarity comparisons
        metric_name: Name of metric to analyze

    Returns:
        Dictionary with mean, median, std, and non-null count
    """
    values = [
        sim[metric_name]
        for sim in similarities
        if sim.get(metric_name) is not None
    ]

    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "count": 0,
            "total": len(similarities),
        }

    res = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "count": len(values),
        "total": len(similarities),
    }
    return res


def analyze_quality_by_match_type(stratified: Dict[str, List[Dict]]) -> None:
    """Analyze similarity metric quality by match type.

    Args:
        stratified: Dictionary mapping match type to similarities
    """
    logger.info("\n===== QUALITY ANALYSIS BY MATCH TYPE =====")

    metrics = [
        "direction_concordance",
        "effect_size_similarity",
        "statistical_consistency",
        "precision_concordance",
        "composite_similarity_direction",
    ]

    for metric in metrics:
        logger.info(f"\n{metric.upper()}:")

        for match_type in ["exact", "fuzzy", "efo", "mixed"]:
            sims = stratified[match_type]
            if not sims:
                continue

            stats = compute_metric_statistics(sims, metric)

            if stats["count"] == 0:
                logger.info(f"  {match_type}: No data")
            else:
                coverage_pct = (
                    100.0 * stats["count"] / stats["total"]
                    if stats["total"] > 0
                    else 0
                )
                logger.info(
                    f"  {match_type}: "
                    f"mean={stats['mean']:.3f}, "
                    f"median={stats['median']:.3f}, "
                    f"std={stats['std']:.3f}, "
                    f"coverage={coverage_pct:.1f}% ({stats['count']}/{stats['total']})"
                )


def analyze_match_type_distribution_by_similarity(
    records: List[Dict],
) -> None:
    """Analyze match type distribution in high vs low similarity pairs.

    Args:
        records: List of similarity records
    """
    logger.info("\n===== MATCH TYPE DISTRIBUTION BY SIMILARITY QUARTILE =====")

    all_similarities = []
    for record in records:
        for sim in record["top_similarities"]:
            composite = sim.get("composite_similarity_direction")
            if composite is not None:
                all_similarities.append(
                    {
                        "composite": composite,
                        "exact": sim["match_type_exact"],
                        "fuzzy": sim["match_type_fuzzy"],
                        "efo": sim["match_type_efo"],
                        "total_pairs": sim["matched_pairs"],
                    }
                )

    if not all_similarities:
        logger.warning("No similarity data available")
        return

    all_similarities.sort(key=lambda x: x["composite"])

    n = len(all_similarities)
    quartiles = {
        "Q1 (lowest)": all_similarities[: n // 4],
        "Q2": all_similarities[n // 4 : n // 2],
        "Q3": all_similarities[n // 2 : 3 * n // 4],
        "Q4 (highest)": all_similarities[3 * n // 4 :],
    }

    for quartile_name, quartile_data in quartiles.items():
        total_pairs = sum(s["total_pairs"] for s in quartile_data)
        total_exact = sum(s["exact"] for s in quartile_data)
        total_fuzzy = sum(s["fuzzy"] for s in quartile_data)
        total_efo = sum(s["efo"] for s in quartile_data)

        exact_pct = 100.0 * total_exact / total_pairs if total_pairs > 0 else 0
        fuzzy_pct = 100.0 * total_fuzzy / total_pairs if total_pairs > 0 else 0
        efo_pct = 100.0 * total_efo / total_pairs if total_pairs > 0 else 0

        logger.info(f"\n{quartile_name}:")
        logger.info(f"  N comparisons: {len(quartile_data)}")
        logger.info(f"  Total matched pairs: {total_pairs}")
        logger.info(f"  Exact: {total_exact} ({exact_pct:.1f}%)")
        logger.info(f"  Fuzzy: {total_fuzzy} ({fuzzy_pct:.1f}%)")
        logger.info(f"  EFO: {total_efo} ({efo_pct:.1f}%)")


def analyze_match_type_vs_completeness(
    stratified: Dict[str, List[Dict]],
) -> None:
    """Analyze relationship between match type and data completeness.

    Args:
        stratified: Dictionary mapping match type to similarities
    """
    logger.info("\n===== DATA COMPLETENESS BY MATCH TYPE =====")

    for match_type in ["exact", "fuzzy", "efo", "mixed"]:
        sims = stratified[match_type]
        if not sims:
            continue

        query_completeness = [
            sim["query_completeness"]
            for sim in sims
            if "query_completeness" in sim
        ]
        similar_completeness = [
            sim["similar_completeness"]
            for sim in sims
            if "similar_completeness" in sim
        ]

        if query_completeness and similar_completeness:
            min_completeness = [
                min(q, s)
                for q, s in zip(query_completeness, similar_completeness)
            ]

            logger.info(f"\n{match_type}:")
            logger.info(
                f"  Query completeness: "
                f"mean={np.mean(query_completeness):.3f}, "
                f"median={np.median(query_completeness):.3f}"
            )
            logger.info(
                f"  Similar completeness: "
                f"mean={np.mean(similar_completeness):.3f}, "
                f"median={np.median(similar_completeness):.3f}"
            )
            logger.info(
                f"  Min completeness: "
                f"mean={np.mean(min_completeness):.3f}, "
                f"median={np.median(min_completeness):.3f}"
            )


def main():
    """Run quality stratification analysis."""
    logger.info("===== MATCH TYPE QUALITY STRATIFICATION ANALYSIS =====\n")

    records = load_similarity_chunks(OUTPUT_DIR)

    if not records:
        logger.error("No similarity records found")
        return 1

    stratified = stratify_by_match_type(records)

    analyze_quality_by_match_type(stratified)

    analyze_match_type_distribution_by_similarity(records)

    analyze_match_type_vs_completeness(stratified)

    logger.info("\n===== ANALYSIS COMPLETE =====")

    return 0


if __name__ == "__main__":
    exit(main())
