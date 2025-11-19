"""Extract trait pairs with similarity expansion potential for Case Study 1.

This script analyzes expansion potential using trait embeddings to find
semantically similar trait pairs beyond exact name matches. Due to computational
constraints, this version computes POTENTIAL expansion (i.e., how many similar
traits exist) rather than executing full similarity-based matching.

The output shows:
- How many similar exposures/outcomes exist for each trait pair
- Estimated expansion potential (similar_exposures × similar_outcomes)
- Which pairs would benefit most from similarity expansion

This serves as a feasibility analysis for future full-scale similarity expansion.

Outputs:
- multi_study_pairs_similarity_expanded.csv: Pairs with expansion metrics
- expansion_comparison.csv: Comparison with exact matches
- expansion_examples.csv: Top 20 pairs with highest expansion potential
- similarity_expansion_metadata.json: Runtime stats and summary
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"


# ==== Argument parsing ====


def make_args():
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Perform dry run without executing extraction",
    )

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration file (default: {DEFAULT_CONFIG})",
    )

    # ---- --similarity-threshold ----
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Override similarity threshold from config (0.0-1.0)",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory from config",
    )

    res = parser.parse_args()
    return res


# ==== Configuration loading ====


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config


# ==== Simplified expansion analysis ====


def analyze_expansion_potential(
    evidence_conn: duckdb.DuckDBPyConnection,
    vector_conn: duckdb.DuckDBPyConnection,
    min_study_count: int,
    similarity_threshold: float = 0.9,
) -> Tuple[pd.DataFrame, Dict]:
    """Analyze similarity expansion potential for trait pairs.

    Computes how many similar traits exist for each exposure/outcome
    to estimate expansion opportunity without executing full expansion.

    Args:
        evidence_conn: Connection to evidence_profile_db
        vector_conn: Connection to vector_store.db
        min_study_count: Minimum studies for inclusion
        similarity_threshold: Base similarity threshold (0.0-1.0)

    Returns:
        Tuple of (analysis_df, metadata)
    """
    start_time = time.time()

    # ---- load baseline pairs ----
    logger.info("Loading baseline exact-match pairs...")
    exact_pairs_path = (
        PROJECT_ROOT
        / "data/processed/case-study-cs1/raw_pairs/multi_study_pairs.csv"
    )
    exact_pairs_df = pd.read_csv(exact_pairs_path)
    logger.info(f"Loaded {len(exact_pairs_df)} baseline pairs")

    # ---- extract unique traits ----
    logger.info("Extracting unique traits from pairs...")
    all_traits = set()
    for idx, row in exact_pairs_df.iterrows():
        trait_pairs = json.loads(row["trait_pairs_json"])
        for pair in trait_pairs:
            all_traits.add(pair["exposure"])
            all_traits.add(pair["outcome"])

    logger.info(f"Found {len(all_traits)} unique traits")

    # ---- batch query similarity counts for all traits ----
    logger.info(
        f"Querying similarity counts (threshold={similarity_threshold})..."
    )
    trait_similarity_counts = {}

    for trait in tqdm(all_traits, desc="Computing similarity counts"):
        try:
            result = vector_conn.execute(
                """
                SELECT COUNT(*) as count
                FROM trait_similarity_search
                WHERE query_label = ? AND similarity >= ?
                """,
                [trait, similarity_threshold],
            ).fetchone()
            trait_similarity_counts[trait] = result[0] if result else 0
        except Exception as e:
            logger.warning(f"Failed to query trait '{trait}': {e}")
            trait_similarity_counts[trait] = 0

    # ---- compute expansion potential for each study ----
    logger.info("Computing expansion potential for each study...")
    expanded_rows = []

    for idx, row in exact_pairs_df.iterrows():
        trait_pairs = json.loads(row["trait_pairs_json"])

        exposure_counts = []
        outcome_counts = []

        for pair in trait_pairs:
            exposure = pair["exposure"]
            outcome = pair["outcome"]

            exp_count = trait_similarity_counts.get(exposure, 0)
            out_count = trait_similarity_counts.get(outcome, 0)

            exposure_counts.append(exp_count)
            outcome_counts.append(out_count)

        # ---- compute summary statistics ----
        mean_exposure_sim = np.mean(exposure_counts) if exposure_counts else 0
        mean_outcome_sim = np.mean(outcome_counts) if outcome_counts else 0
        max_exposure_sim = max(exposure_counts) if exposure_counts else 0
        max_outcome_sim = max(outcome_counts) if outcome_counts else 0

        # ---- estimate expansion potential ----
        expansion_potential = min(max_exposure_sim * max_outcome_sim, 10000)

        expanded_row = row.to_dict()
        expanded_row.update(
            {
                "original_study_count": row["study_count"],
                "expanded_study_count": row["study_count"],
                "expansion_ratio": 1.0,
                "mean_similar_exposures": float(mean_exposure_sim),
                "mean_similar_outcomes": float(mean_outcome_sim),
                "max_similar_exposures": int(max_exposure_sim),
                "max_similar_outcomes": int(max_outcome_sim),
                "expansion_potential": int(expansion_potential),
                "mean_concordance_exact": row["mean_direction_concordance"],
                "mean_concordance_expanded": row["mean_direction_concordance"],
                "concordance_delta": 0.0,
            }
        )
        expanded_rows.append(expanded_row)

    expanded_df = pd.DataFrame(expanded_rows)

    # ---- compute metadata ----
    elapsed_time = time.time() - start_time
    metadata = {
        "script": "case_study_1_extract_pairs_similarity_expanded.py",
        "mode": "expansion_potential_analysis",
        "similarity_threshold": similarity_threshold,
        "min_study_count": min_study_count,
        "unique_traits_analyzed": len(all_traits),
        "total_pairs": len(expanded_df),
        "pairs_with_expansion_potential": int(
            (expanded_df["expansion_potential"] > 1).sum()
        ),
        "mean_expansion_potential": float(
            expanded_df["expansion_potential"].mean()
        ),
        "median_expansion_potential": float(
            expanded_df["expansion_potential"].median()
        ),
        "max_expansion_potential": float(
            expanded_df["expansion_potential"].max()
        ),
        "mean_similar_exposures": float(
            expanded_df["mean_similar_exposures"].mean()
        ),
        "mean_similar_outcomes": float(
            expanded_df["mean_similar_outcomes"].mean()
        ),
        "runtime_seconds": elapsed_time,
    }

    logger.info(f"Analysis complete in {elapsed_time:.1f}s")
    logger.info(
        f"Pairs with expansion potential (>1): "
        f"{metadata['pairs_with_expansion_potential']} "
        f"({metadata['pairs_with_expansion_potential'] / len(expanded_df) * 100:.1f}%)"
    )

    return expanded_df, metadata


# ==== Comparison and output generation ====


def generate_comparison(
    exact_df: pd.DataFrame, expanded_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate side-by-side comparison showing expansion potential.

    Args:
        exact_df: Baseline exact-match pairs
        expanded_df: Pairs with expansion potential analysis

    Returns:
        Comparison DataFrame
    """
    comparison_rows = []

    for idx, exact_row in exact_df.iterrows():
        pmid = exact_row["study1_pmid"]
        model = exact_row["study1_model"]

        expanded_row = expanded_df[
            (expanded_df["study1_pmid"] == pmid)
            & (expanded_df["study1_model"] == model)
        ]

        if len(expanded_row) == 0:
            continue

        expanded_row = expanded_row.iloc[0]

        comparison_rows.append(
            {
                "pmid": pmid,
                "model": model,
                "exact_count": exact_row["study_count"],
                "expansion_potential": expanded_row["expansion_potential"],
                "mean_similar_exposures": expanded_row[
                    "mean_similar_exposures"
                ],
                "mean_similar_outcomes": expanded_row["mean_similar_outcomes"],
                "max_similar_exposures": expanded_row["max_similar_exposures"],
                "max_similar_outcomes": expanded_row["max_similar_outcomes"],
                "mean_concordance": exact_row["mean_direction_concordance"],
            }
        )

    res = pd.DataFrame(comparison_rows)
    return res


def generate_expansion_examples(
    expanded_df: pd.DataFrame, top_n: int = 20
) -> pd.DataFrame:
    """Extract top N pairs with highest expansion potential.

    Args:
        expanded_df: Pairs with expansion potential
        top_n: Number of examples to extract

    Returns:
        DataFrame with expansion examples
    """
    sorted_df = expanded_df.sort_values("expansion_potential", ascending=False)
    top_expanded = sorted_df.head(top_n)

    examples = []
    for idx, row in top_expanded.iterrows():
        trait_pairs = json.loads(row["trait_pairs_json"])

        exposures = [p["exposure"] for p in trait_pairs]
        outcomes = [p["outcome"] for p in trait_pairs]

        examples.append(
            {
                "pmid": row["study1_pmid"],
                "model": row["study1_model"],
                "title": row["title"][:80] + "..."
                if len(row["title"]) > 80
                else row["title"],
                "original_exposures": "; ".join(exposures[:2]),
                "original_outcomes": "; ".join(outcomes[:2]),
                "expansion_potential": row["expansion_potential"],
                "original_study_count": row["original_study_count"],
                "max_similar_exposures": row["max_similar_exposures"],
                "max_similar_outcomes": row["max_similar_outcomes"],
                "mean_concordance": row["mean_concordance_exact"],
            }
        )

    res = pd.DataFrame(examples)
    return res


def generate_summary_stats(
    exact_df: pd.DataFrame, expanded_df: pd.DataFrame
) -> Dict:
    """Generate summary statistics for expansion potential.

    Args:
        exact_df: Baseline exact-match pairs
        expanded_df: Pairs with expansion potential

    Returns:
        Dictionary with summary statistics
    """
    res = {
        "overall": {
            "total_pairs": len(exact_df),
            "pairs_with_potential": int(
                (expanded_df["expansion_potential"] > 1).sum()
            ),
            "pct_with_potential": float(
                (expanded_df["expansion_potential"] > 1).sum()
                / len(exact_df)
                * 100
            ),
            "mean_expansion_potential": float(
                expanded_df["expansion_potential"].mean()
            ),
            "median_expansion_potential": float(
                expanded_df["expansion_potential"].median()
            ),
        },
        "similarity_distribution": {
            "mean_similar_exposures": float(
                expanded_df["mean_similar_exposures"].mean()
            ),
            "median_similar_exposures": float(
                expanded_df["mean_similar_exposures"].median()
            ),
            "mean_similar_outcomes": float(
                expanded_df["mean_similar_outcomes"].mean()
            ),
            "median_similar_outcomes": float(
                expanded_df["mean_similar_outcomes"].median()
            ),
            "max_similar_exposures": int(
                expanded_df["max_similar_exposures"].max()
            ),
            "max_similar_outcomes": int(
                expanded_df["max_similar_outcomes"].max()
            ),
        },
    }
    return res


# ==== Main execution ====


def main():
    """Execute similarity expansion potential analysis."""
    args = make_args()

    # ---- load configuration ----

    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    cs1_config = config["case_study_1"]
    db_config = config["databases"]
    output_config = config["output"]["case_study_1"]

    # ---- resolve paths ----

    evidence_db = PROJECT_ROOT / db_config["evidence_profile"]
    vector_store_db = PROJECT_ROOT / db_config["vector_store"]
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = (
            PROJECT_ROOT / output_config["base"] / "similarity_expanded"
        )

    # ---- get similarity threshold ----
    if args.similarity_threshold:
        similarity_threshold = args.similarity_threshold
    else:
        similarity_threshold = cs1_config.get("similarity_expansion", {}).get(
            "similarity_threshold", 0.9
        )

    # ---- validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating configuration and paths")
        logger.info(f"Evidence database: {evidence_db}")
        logger.info(f"Vector store database: {vector_store_db}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Min study count: {cs1_config['min_study_count']}")
        logger.info(f"Similarity threshold: {similarity_threshold}")

        if not evidence_db.exists():
            logger.error(f"Evidence database not found: {evidence_db}")
            return 1

        if not vector_store_db.exists():
            logger.error(f"Vector store database not found: {vector_store_db}")
            return 1

        logger.info("Dry run complete - configuration validated")
        return 0

    # ---- setup ----

    if not evidence_db.exists():
        logger.error(f"Evidence database not found: {evidence_db}")
        return 1

    if not vector_store_db.exists():
        logger.error(f"Vector store database not found: {vector_store_db}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ---- connect to databases ----

    logger.info("Connecting to databases...")
    try:
        evidence_conn = duckdb.connect(str(evidence_db), read_only=True)
        vector_conn = duckdb.connect(str(vector_store_db), read_only=True)
    except Exception as e:
        logger.error(f"Failed to connect to databases: {e}")
        return 1

    # ---- load baseline exact-match pairs ----

    exact_pairs_path = (
        PROJECT_ROOT / output_config["raw_pairs"] / "multi_study_pairs.csv"
    )
    if not exact_pairs_path.exists():
        logger.error(
            f"Baseline pairs not found: {exact_pairs_path}\n"
            "Run case_study_1_extract_pairs.py first"
        )
        evidence_conn.close()
        vector_conn.close()
        return 1

    exact_pairs_df = pd.read_csv(exact_pairs_path)

    # ---- analyze expansion potential ----

    min_study_count = cs1_config["min_study_count"]
    expanded_df, metadata = analyze_expansion_potential(
        evidence_conn,
        vector_conn,
        min_study_count,
        similarity_threshold,
    )

    if len(expanded_df) == 0:
        logger.warning("No pairs found for analysis")
        evidence_conn.close()
        vector_conn.close()
        return 0

    # ---- generate outputs ----

    logger.info("Generating output files...")
    comparison_df = generate_comparison(exact_pairs_df, expanded_df)
    examples_df = generate_expansion_examples(expanded_df, top_n=20)
    summary_stats = generate_summary_stats(exact_pairs_df, expanded_df)

    # ---- save outputs ----

    expanded_file = output_dir / "multi_study_pairs_similarity_expanded.csv"
    comparison_file = output_dir / "expansion_comparison.csv"
    examples_file = output_dir / "expansion_examples.csv"
    metadata_file = output_dir / "similarity_expansion_metadata.json"

    expanded_df.to_csv(expanded_file, index=False)
    logger.info(f"Saved expanded pairs: {expanded_file}")

    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"Saved comparison: {comparison_file}")

    examples_df.to_csv(examples_file, index=False)
    logger.info(f"Saved examples: {examples_file}")

    metadata["summary_statistics"] = summary_stats

    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_file}")

    # ---- print summary ----

    logger.info("\n" + "=" * 70)
    logger.info("SIMILARITY EXPANSION POTENTIAL ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Mode: {metadata['mode']}")
    logger.info(f"Total pairs analyzed: {metadata['total_pairs']}")
    logger.info(f"Unique traits: {metadata['unique_traits_analyzed']}")
    logger.info(
        f"Pairs with expansion potential: "
        f"{metadata['pairs_with_expansion_potential']} "
        f"({summary_stats['overall']['pct_with_potential']:.1f}%)"
    )
    logger.info(
        f"Mean expansion potential: "
        f"{metadata['mean_expansion_potential']:.1f}x"
    )
    logger.info(
        f"Median expansion potential: "
        f"{metadata['median_expansion_potential']:.1f}x"
    )
    logger.info(
        f"Max expansion potential: {metadata['max_expansion_potential']:.0f}x"
    )
    logger.info("\nSimilarity distribution:")
    logger.info(
        f"  Mean similar exposures: {metadata['mean_similar_exposures']:.1f}"
    )
    logger.info(
        f"  Mean similar outcomes: {metadata['mean_similar_outcomes']:.1f}"
    )
    logger.info(
        f"  Max similar exposures: "
        f"{summary_stats['similarity_distribution']['max_similar_exposures']}"
    )
    logger.info(
        f"  Max similar outcomes: "
        f"{summary_stats['similarity_distribution']['max_similar_outcomes']}"
    )
    logger.info(f"\nRuntime: {metadata['runtime_seconds']:.1f}s")
    logger.info("=" * 70)
    logger.info("\nNote: This analysis computes POTENTIAL expansion only.")
    logger.info(
        "Full similarity-based matching would require additional computation."
    )

    # ---- cleanup ----

    evidence_conn.close()
    vector_conn.close()
    logger.info("\nAnalysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
