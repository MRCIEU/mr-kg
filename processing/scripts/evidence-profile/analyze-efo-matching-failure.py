"""Analyze EFO matching failure in batch 3 evidence profile similarity.

This script investigates why EFO matching contributed only 0.2% of matches
(106 out of 7,889 total matched pairs) despite being enabled with a 0.50
similarity threshold.

The investigation covers:
1. EFO mapping coverage in vector_store.db
2. Distribution of EFO similarity scores
3. Analysis of actual EFO matches in batch 3 output
4. Threshold sensitivity simulation
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
VECTOR_STORE_PATH = DB_DIR / "vector_store.db"
EVIDENCE_PROFILE_DB_PATH = DB_DIR / "evidence_profile_db.db"
EVIDENCE_PROFILES_PATH = (
    DATA_DIR / "processed" / "evidence-profiles" / "evidence-profiles.json"
)
BATCH3_OUTPUT_DIR = DATA_DIR / "output" / "bc4-12799864" / "results"


def load_trait_efo_similarity_distribution(
    db_path: Path,
) -> Tuple[pd.DataFrame, Dict]:
    """Analyze distribution of trait-EFO similarities.

    Args:
        db_path: Path to vector_store.db

    Returns:
        Tuple of (similarity distribution DataFrame, summary statistics dict)
    """
    logger.info("Loading trait-EFO similarity distribution...")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Check if trait_efo_similarity_search table exists
    tables_query = "SHOW TABLES"
    tables = conn.execute(tables_query).fetchall()
    table_names = [t[0] for t in tables]

    if "trait_efo_similarity_search" not in table_names:
        logger.warning(
            "trait_efo_similarity_search table not found, computing similarities"
        )
        return compute_trait_efo_similarities_in_memory(db_path)

    # Load trait vectors
    trait_query = """
    SELECT trait_index, vector
    FROM trait_embeddings
    ORDER BY trait_index
    """
    trait_results = conn.execute(trait_query).fetchall()
    trait_indices = [r[0] for r in trait_results]
    trait_vectors = np.array([r[1] for r in trait_results])

    # Load EFO vectors
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

    # Get max similarity per trait
    max_similarities = np.max(similarities, axis=1)

    # Create distribution
    similarity_ranges = [
        (0.70, 1.00, "0.70+"),
        (0.60, 0.70, "0.60-0.70"),
        (0.50, 0.60, "0.50-0.60"),
        (0.40, 0.50, "0.40-0.50"),
        (0.30, 0.40, "0.30-0.40"),
        (0.00, 0.30, "<0.30"),
    ]

    distribution_data = []
    for low, high, label in similarity_ranges:
        count = np.sum((max_similarities >= low) & (max_similarities < high))
        percentage = (count / len(max_similarities)) * 100
        distribution_data.append(
            {
                "range": label,
                "count": int(count),
                "percentage": round(percentage, 2),
            }
        )

    df_dist = pd.DataFrame(distribution_data)

    # Summary statistics
    summary = {
        "total_traits": len(trait_indices),
        "traits_above_0.80": int(np.sum(max_similarities >= 0.80)),
        "traits_above_0.70": int(np.sum(max_similarities >= 0.70)),
        "traits_above_0.60": int(np.sum(max_similarities >= 0.60)),
        "traits_above_0.50": int(np.sum(max_similarities >= 0.50)),
        "traits_above_0.40": int(np.sum(max_similarities >= 0.40)),
        "traits_above_0.30": int(np.sum(max_similarities >= 0.30)),
        "mean_similarity": float(np.mean(max_similarities)),
        "median_similarity": float(np.median(max_similarities)),
        "std_similarity": float(np.std(max_similarities)),
    }

    return df_dist, summary


def compute_trait_efo_similarities_in_memory(
    db_path: Path,
) -> Tuple[pd.DataFrame, Dict]:
    """Compute trait-EFO similarities in memory.

    Args:
        db_path: Path to vector_store.db

    Returns:
        Tuple of (similarity distribution DataFrame, summary statistics dict)
    """
    logger.info("Computing trait-EFO similarities in memory...")
    conn = duckdb.connect(str(db_path), read_only=True)

    # Load trait vectors
    trait_query = """
    SELECT trait_index, vector
    FROM trait_embeddings
    ORDER BY trait_index
    """
    trait_results = conn.execute(trait_query).fetchall()
    trait_indices = [r[0] for r in trait_results]
    trait_vectors = np.array([r[1] for r in trait_results])

    # Load EFO vectors
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

    # Get max similarity per trait
    max_similarities = np.max(similarities, axis=1)

    # Create distribution
    similarity_ranges = [
        (0.70, 1.00, "0.70+"),
        (0.60, 0.70, "0.60-0.70"),
        (0.50, 0.60, "0.50-0.60"),
        (0.40, 0.50, "0.40-0.50"),
        (0.30, 0.40, "0.30-0.40"),
        (0.00, 0.30, "<0.30"),
    ]

    distribution_data = []
    for low, high, label in similarity_ranges:
        count = np.sum((max_similarities >= low) & (max_similarities < high))
        percentage = (count / len(max_similarities)) * 100
        distribution_data.append(
            {
                "range": label,
                "count": int(count),
                "percentage": round(percentage, 2),
            }
        )

    df_dist = pd.DataFrame(distribution_data)

    # Summary statistics
    summary = {
        "total_traits": len(trait_indices),
        "traits_above_0.80": int(np.sum(max_similarities >= 0.80)),
        "traits_above_0.70": int(np.sum(max_similarities >= 0.70)),
        "traits_above_0.60": int(np.sum(max_similarities >= 0.60)),
        "traits_above_0.50": int(np.sum(max_similarities >= 0.50)),
        "traits_above_0.40": int(np.sum(max_similarities >= 0.40)),
        "traits_above_0.30": int(np.sum(max_similarities >= 0.30)),
        "mean_similarity": float(np.mean(max_similarities)),
        "median_similarity": float(np.median(max_similarities)),
        "std_similarity": float(np.std(max_similarities)),
    }

    return df_dist, summary


def analyze_evidence_profile_trait_coverage(
    evidence_profiles_path: Path, db_path: Path, threshold: float = 0.50
) -> Dict:
    """Analyze what percentage of traits in evidence profiles have EFO mappings.

    Args:
        evidence_profiles_path: Path to evidence profiles JSON
        db_path: Path to vector_store.db
        threshold: EFO similarity threshold

    Returns:
        Dictionary with coverage statistics
    """
    logger.info("Analyzing evidence profile trait coverage...")

    # Load evidence profiles
    with evidence_profiles_path.open("r") as f:
        profiles = json.load(f)

    # Extract all unique trait indices from evidence profiles
    trait_indices_in_profiles = set()
    for profile in profiles:
        for result in profile["results"]:
            trait_indices_in_profiles.add(result["exposure_trait_index"])
            trait_indices_in_profiles.add(result["outcome_trait_index"])

    logger.info(
        f"Found {len(trait_indices_in_profiles)} unique traits in evidence profiles"
    )

    # Load trait-EFO mappings at threshold
    conn = duckdb.connect(str(db_path), read_only=True)

    # Load trait vectors
    trait_query = """
    SELECT trait_index, vector
    FROM trait_embeddings
    ORDER BY trait_index
    """
    trait_results = conn.execute(trait_query).fetchall()
    trait_indices_all = [r[0] for r in trait_results]
    trait_vectors = np.array([r[1] for r in trait_results])

    # Load EFO vectors
    efo_query = """
    SELECT id, vector
    FROM efo_embeddings
    ORDER BY id
    """
    efo_results = conn.execute(efo_query).fetchall()
    efo_ids = [r[0] for r in efo_results]
    efo_vectors = np.array([r[1] for r in efo_results])

    conn.close()

    logger.info("Computing trait-EFO similarities...")
    similarities = cosine_similarity(trait_vectors, efo_vectors)

    # Build trait-to-EFO mapping
    trait_efo_map = {}
    for i, trait_idx in enumerate(trait_indices_all):
        trait_sims = similarities[i]
        max_idx = np.argmax(trait_sims)
        max_sim = trait_sims[max_idx]

        if max_sim >= threshold:
            trait_efo_map[trait_idx] = (efo_ids[max_idx], max_sim)

    # Check coverage of traits in evidence profiles
    traits_in_profiles_with_efo = [
        t for t in trait_indices_in_profiles if t in trait_efo_map
    ]

    coverage = {
        "total_unique_traits_in_profiles": len(trait_indices_in_profiles),
        "traits_with_efo_mapping": len(traits_in_profiles_with_efo),
        "coverage_percentage": round(
            (len(traits_in_profiles_with_efo) / len(trait_indices_in_profiles))
            * 100,
            2,
        ),
        "threshold_used": threshold,
    }

    return coverage


def parse_batch3_results(output_dir: Path) -> Dict:
    """Parse batch 3 output files and analyze EFO match statistics.

    Args:
        output_dir: Path to batch 3 results directory

    Returns:
        Dictionary with match statistics and examples
    """
    logger.info(f"Parsing batch 3 results from: {output_dir}")

    chunk_files = sorted(output_dir.glob("evidence_similarities_chunk_*.json"))
    logger.info(f"Found {len(chunk_files)} chunk files")

    total_queries = 0
    total_matches = 0
    match_type_counts = {"exact": 0, "fuzzy": 0, "efo": 0}
    efo_match_examples = []

    for chunk_file in chunk_files:
        with chunk_file.open("r") as f:
            records = json.load(f)

        total_queries += len(records)

        for record in records:
            for sim in record["top_similarities"]:
                total_matches += 1
                match_type_counts["exact"] += sim.get("match_type_exact", 0)
                match_type_counts["fuzzy"] += sim.get("match_type_fuzzy", 0)
                match_type_counts["efo"] += sim.get("match_type_efo", 0)

                # Collect EFO match examples
                if sim.get("match_type_efo", 0) > 0 and len(efo_match_examples) < 10:
                    efo_match_examples.append(
                        {
                            "query_pmid": record["query_pmid"],
                            "query_model": record["query_model"],
                            "similar_pmid": sim["similar_pmid"],
                            "similar_model": sim["similar_model"],
                            "matched_pairs": sim["matched_pairs"],
                            "efo_matches": sim["match_type_efo"],
                            "exact_matches": sim["match_type_exact"],
                            "fuzzy_matches": sim["match_type_fuzzy"],
                            "composite_similarity": sim.get(
                                "composite_similarity_direction"
                            ),
                        }
                    )

    # Calculate percentages
    total_match_count = sum(match_type_counts.values())

    results = {
        "total_queries": total_queries,
        "total_match_instances": total_matches,
        "total_matched_pairs": total_match_count,
        "match_type_counts": match_type_counts,
        "match_type_percentages": {
            k: round((v / total_match_count) * 100, 2) if total_match_count > 0 else 0
            for k, v in match_type_counts.items()
        },
        "efo_match_examples": efo_match_examples,
    }

    return results


def simulate_threshold_sensitivity(
    evidence_profiles_path: Path, db_path: Path, thresholds: List[float]
) -> pd.DataFrame:
    """Simulate expected EFO matches at different thresholds.

    Args:
        evidence_profiles_path: Path to evidence profiles JSON
        db_path: Path to vector_store.db
        thresholds: List of similarity thresholds to test

    Returns:
        DataFrame with expected matches at each threshold
    """
    logger.info("Simulating threshold sensitivity...")

    # Load evidence profiles (use first 100 for simulation)
    with evidence_profiles_path.open("r") as f:
        profiles = json.load(f)

    profiles_sample = profiles[:100]
    logger.info(f"Using {len(profiles_sample)} profiles for simulation")

    # Load trait embeddings and EFO embeddings
    conn = duckdb.connect(str(db_path), read_only=True)

    trait_query = """
    SELECT trait_index, vector
    FROM trait_embeddings
    ORDER BY trait_index
    """
    trait_results = conn.execute(trait_query).fetchall()
    trait_indices = [r[0] for r in trait_results]
    trait_vectors = np.array([r[1] for r in trait_results])

    efo_query = """
    SELECT id, vector
    FROM efo_embeddings
    ORDER BY id
    """
    efo_results = conn.execute(efo_query).fetchall()
    efo_ids = [r[0] for r in efo_results]
    efo_vectors = np.array([r[1] for r in efo_results])

    conn.close()

    logger.info("Computing trait-EFO similarities...")
    similarities = cosine_similarity(trait_vectors, efo_vectors)

    simulation_results = []

    for threshold in thresholds:
        logger.info(f"Testing threshold: {threshold}")

        # Build trait-to-EFO mapping
        trait_efo_map = {}
        for i, trait_idx in enumerate(trait_indices):
            trait_sims = similarities[i]
            max_idx = np.argmax(trait_sims)
            max_sim = trait_sims[max_idx]

            if max_sim >= threshold:
                trait_efo_map[trait_idx] = efo_ids[max_idx]

        # Count potential EFO matches
        efo_matchable_pairs = 0
        total_pairs = 0

        for i, profile1 in enumerate(profiles_sample):
            for profile2 in profiles_sample[i + 1 :]:
                if profile1["model"] != profile2["model"]:
                    continue

                # Check if any pairs could match via EFO
                for result1 in profile1["results"]:
                    exp1 = result1["exposure_trait_index"]
                    out1 = result1["outcome_trait_index"]

                    if exp1 not in trait_efo_map or out1 not in trait_efo_map:
                        continue

                    exp1_efo = trait_efo_map[exp1]
                    out1_efo = trait_efo_map[out1]

                    for result2 in profile2["results"]:
                        total_pairs += 1
                        exp2 = result2["exposure_trait_index"]
                        out2 = result2["outcome_trait_index"]

                        if exp2 not in trait_efo_map or out2 not in trait_efo_map:
                            continue

                        exp2_efo = trait_efo_map[exp2]
                        out2_efo = trait_efo_map[out2]

                        if exp1_efo == exp2_efo and out1_efo == out2_efo:
                            efo_matchable_pairs += 1

        simulation_results.append(
            {
                "threshold": threshold,
                "traits_with_efo": len(trait_efo_map),
                "efo_matchable_pairs": efo_matchable_pairs,
                "total_pairs_checked": total_pairs,
                "match_rate": (
                    round((efo_matchable_pairs / total_pairs) * 100, 4)
                    if total_pairs > 0
                    else 0
                ),
            }
        )

    df_sim = pd.DataFrame(simulation_results)
    return df_sim


def main():
    """Main analysis function."""
    logger.info("=" * 60)
    logger.info("EFO Matching Failure Investigation")
    logger.info("=" * 60)

    # ---- Section 1: EFO Mapping Coverage ----
    logger.info("\n1. EFO MAPPING COVERAGE")
    logger.info("-" * 60)

    df_dist, summary = load_trait_efo_similarity_distribution(VECTOR_STORE_PATH)

    logger.info(f"\nTotal traits in vector_store: {summary['total_traits']}")
    logger.info(
        f"Traits with EFO mapping (>= 0.80): {summary['traits_above_0.80']} "
        f"({round(summary['traits_above_0.80'] / summary['total_traits'] * 100, 2)}%)"
    )
    logger.info(
        f"Traits with EFO mapping (>= 0.70): {summary['traits_above_0.70']} "
        f"({round(summary['traits_above_0.70'] / summary['total_traits'] * 100, 2)}%)"
    )
    logger.info(
        f"Traits with EFO mapping (>= 0.60): {summary['traits_above_0.60']} "
        f"({round(summary['traits_above_0.60'] / summary['total_traits'] * 100, 2)}%)"
    )
    logger.info(
        f"Traits with EFO mapping (>= 0.50): {summary['traits_above_0.50']} "
        f"({round(summary['traits_above_0.50'] / summary['total_traits'] * 100, 2)}%)"
    )
    logger.info(
        f"Mean similarity: {round(summary['mean_similarity'], 4)}, "
        f"Median: {round(summary['median_similarity'], 4)}"
    )

    # Coverage of traits in evidence profiles
    coverage_0_80 = analyze_evidence_profile_trait_coverage(
        EVIDENCE_PROFILES_PATH, VECTOR_STORE_PATH, threshold=0.80
    )
    coverage_0_50 = analyze_evidence_profile_trait_coverage(
        EVIDENCE_PROFILES_PATH, VECTOR_STORE_PATH, threshold=0.50
    )

    logger.info(
        f"\nTraits in evidence profiles: {coverage_0_80['total_unique_traits_in_profiles']}"
    )
    logger.info(
        f"Evidence traits with EFO mapping (>= 0.80): {coverage_0_80['traits_with_efo_mapping']} "
        f"({coverage_0_80['coverage_percentage']}%)"
    )
    logger.info(
        f"Evidence traits with EFO mapping (>= 0.50): {coverage_0_50['traits_with_efo_mapping']} "
        f"({coverage_0_50['coverage_percentage']}%)"
    )

    # ---- Section 2: Similarity Distribution ----
    logger.info("\n2. SIMILARITY DISTRIBUTION")
    logger.info("-" * 60)
    logger.info("\n" + df_dist.to_string(index=False))

    # ---- Section 3: Actual EFO Matches in Batch 3 ----
    logger.info("\n3. ACTUAL EFO MATCHES IN BATCH 3")
    logger.info("-" * 60)

    batch3_results = parse_batch3_results(BATCH3_OUTPUT_DIR)

    logger.info(f"\nTotal queries processed: {batch3_results['total_queries']}")
    logger.info(
        f"Total match instances: {batch3_results['total_match_instances']}"
    )
    logger.info(
        f"Total matched pairs: {batch3_results['total_matched_pairs']}"
    )
    logger.info(
        f"\nMatch type counts:"
        f"\n  - Exact: {batch3_results['match_type_counts']['exact']} "
        f"({batch3_results['match_type_percentages']['exact']}%)"
        f"\n  - Fuzzy: {batch3_results['match_type_counts']['fuzzy']} "
        f"({batch3_results['match_type_percentages']['fuzzy']}%)"
        f"\n  - EFO: {batch3_results['match_type_counts']['efo']} "
        f"({batch3_results['match_type_percentages']['efo']}%)"
    )

    if batch3_results["efo_match_examples"]:
        logger.info(f"\nEFO match examples (showing first 10):")
        for i, example in enumerate(batch3_results["efo_match_examples"], 1):
            logger.info(
                f"\n  Example {i}:"
                f"\n    Query: PMID={example['query_pmid']}, Model={example['query_model']}"
                f"\n    Similar: PMID={example['similar_pmid']}, Model={example['similar_model']}"
                f"\n    Matched pairs: {example['matched_pairs']}"
                f"\n    Match types - Exact: {example['exact_matches']}, "
                f"Fuzzy: {example['fuzzy_matches']}, EFO: {example['efo_matches']}"
                f"\n    Composite similarity: {example['composite_similarity']}"
            )
    else:
        logger.info("\n  No EFO match examples found!")

    # ---- Section 4: Threshold Sensitivity ----
    logger.info("\n4. THRESHOLD SENSITIVITY ANALYSIS")
    logger.info("-" * 60)

    thresholds = [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    df_sim = simulate_threshold_sensitivity(
        EVIDENCE_PROFILES_PATH, VECTOR_STORE_PATH, thresholds
    )

    logger.info("\n" + df_sim.to_string(index=False))

    # ---- Section 5: Root Cause Analysis ----
    logger.info("\n5. ROOT CAUSE ANALYSIS")
    logger.info("-" * 60)

    # Determine root cause based on findings
    if summary["traits_above_0.80"] < summary["total_traits"] * 0.10:
        logger.info(
            "\nROOT CAUSE: Very few traits have high-similarity EFO mappings"
        )
        logger.info(
            f"Only {round(summary['traits_above_0.80'] / summary['total_traits'] * 100, 2)}% "
            f"of traits have EFO similarity >= 0.80"
        )
        logger.info(
            "This suggests MR traits are poorly represented by EFO ontology"
        )
    elif coverage_0_80["coverage_percentage"] < 10:
        logger.info(
            "\nROOT CAUSE: Evidence profile traits lack EFO mappings at threshold 0.80"
        )
        logger.info(
            f"Only {coverage_0_80['coverage_percentage']}% of evidence profile traits "
            f"have EFO mappings"
        )
    elif batch3_results["match_type_counts"]["efo"] == 0:
        logger.info(
            "\nROOT CAUSE: EFO matching implementation may not be working correctly"
        )
        logger.info(
            "Despite having trait-EFO mappings, no EFO matches were found in batch 3"
        )
    else:
        logger.info(
            "\nROOT CAUSE: Combination of low EFO mapping coverage and high threshold"
        )
        logger.info(
            f"EFO matches: {batch3_results['match_type_counts']['efo']} "
            f"({batch3_results['match_type_percentages']['efo']}%)"
        )

    # ---- Section 6: Recommendations ----
    logger.info("\n6. RECOMMENDATIONS")
    logger.info("-" * 60)

    if summary["traits_above_0.50"] > summary["traits_above_0.80"] * 2:
        logger.info(
            "\n✓ RECOMMENDATION: Lower EFO similarity threshold to 0.50 or 0.40"
        )
        logger.info(
            f"  - At 0.50: {summary['traits_above_0.50']} traits "
            f"({round(summary['traits_above_0.50'] / summary['total_traits'] * 100, 2)}%)"
        )
        logger.info(
            f"  - At 0.40: {summary['traits_above_0.40']} traits "
            f"({round(summary['traits_above_0.40'] / summary['total_traits'] * 100, 2)}%)"
        )
    else:
        logger.info(
            "\n✗ RECOMMENDATION: EFO ontology may not be appropriate for MR traits"
        )
        logger.info(
            "  - Consider using disease-specific ontologies (e.g., MONDO, DOID)"
        )
        logger.info(
            "  - Or rely solely on fuzzy semantic matching without ontology constraints"
        )

    if batch3_results["match_type_counts"]["fuzzy"] > 0:
        logger.info(
            f"\n✓ Fuzzy matching is working well: {batch3_results['match_type_counts']['fuzzy']} matches"
        )
        logger.info(
            "  Consider relying on fuzzy matching as primary similarity method"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Investigation Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
