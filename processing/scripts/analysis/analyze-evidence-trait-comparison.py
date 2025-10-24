"""Compare evidence similarity with trait similarity.

This script analyzes the relationship between trait profile similarity and
evidence profile similarity by joining both databases and computing:
- Correlations between trait and evidence similarity scores
- Quadrant classification (high/low trait vs high/low evidence)
- Identification of interesting cases (divergent or analogous findings)

Outputs:
- trait-vs-evidence-correlation.csv: Correlation statistics by model
- quadrant-classification.csv: Full dataset with quadrant labels
- interesting-cases.csv: Cases in divergent/analogous quadrants
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
DEFAULT_EVIDENCE_DB = DB_DIR / "evidence_profile_db.db"
DEFAULT_TRAIT_DB = DB_DIR / "trait_profile_db.db"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed" / "evidence-profiles" / "analysis"


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
        help="Perform dry run without executing analysis",
    )

    # ---- --evidence-db ----
    parser.add_argument(
        "--evidence-db",
        type=Path,
        default=DEFAULT_EVIDENCE_DB,
        help=f"Path to evidence profile database (default: {DEFAULT_EVIDENCE_DB})",
    )

    # ---- --trait-db ----
    parser.add_argument(
        "--trait-db",
        type=Path,
        default=DEFAULT_TRAIT_DB,
        help=f"Path to trait profile database (default: {DEFAULT_TRAIT_DB})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for analysis results (default: {DEFAULT_OUTPUT_DIR})",
    )

    res = parser.parse_args()
    return res


# ==== Analysis functions ====


def load_matched_similarities(
    evidence_conn: duckdb.DuckDBPyConnection,
    trait_conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """JOIN evidence and trait similarity views on PMID-model pairs.

    Matches studies on:
    - query_pmid, query_model
    - similar_pmid, similar_model

    Args:
        evidence_conn: Connection to evidence profile database
        trait_conn: Connection to trait profile database

    Returns:
        DataFrame with both similarity scores for matched pairs
    """
    logger.info("Loading evidence similarities...")
    evidence_query = """
    SELECT
        query_pmid,
        query_model,
        query_title,
        similar_pmid,
        similar_model,
        similar_title,
        matched_pairs,
        direction_concordance,
        effect_size_similarity,
        statistical_consistency,
        evidence_overlap,
        composite_similarity_equal as evidence_composite_equal,
        composite_similarity_direction as evidence_composite_direction
    FROM evidence_similarity_analysis
    """
    evidence_df = evidence_conn.execute(evidence_query).df()
    logger.info(f"Loaded {len(evidence_df)} evidence similarity pairs")

    logger.info("Loading trait similarities...")
    trait_query = """
    SELECT
        query_pmid,
        query_model,
        similar_pmid,
        similar_model,
        trait_profile_similarity as trait_semantic,
        trait_jaccard_similarity as trait_jaccard
    FROM trait_similarity_analysis
    """
    trait_df = trait_conn.execute(trait_query).df()
    logger.info(f"Loaded {len(trait_df)} trait similarity pairs")

    logger.info("Joining on PMID-model pairs...")
    merged_df = evidence_df.merge(
        trait_df,
        on=["query_pmid", "query_model", "similar_pmid", "similar_model"],
        how="inner",
    )
    logger.info(f"Matched {len(merged_df)} pairs across both databases")

    res = merged_df
    return res


def compute_correlation_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation stratified by model.

    For each model, compute:
    - Correlation between evidence and trait similarity
    - Sample size (number of matched pairs)
    - P-value for correlation

    Args:
        df: DataFrame with matched similarity scores

    Returns:
        DataFrame with correlation statistics by model
    """
    from scipy.stats import pearsonr

    results = []

    for model in df["query_model"].unique():
        model_df = df[df["query_model"] == model]

        # Correlation with evidence composite direction
        mask = (
            model_df[["evidence_composite_direction", "trait_semantic"]]
            .notna()
            .all(axis=1)
        )
        if mask.sum() >= 3:
            corr_dir, p_dir = pearsonr(
                model_df.loc[mask, "evidence_composite_direction"],
                model_df.loc[mask, "trait_semantic"],
            )
        else:
            corr_dir, p_dir = None, None

        # Correlation with evidence composite equal
        mask = (
            model_df[["evidence_composite_equal", "trait_semantic"]]
            .notna()
            .all(axis=1)
        )
        if mask.sum() >= 3:
            corr_eq, p_eq = pearsonr(
                model_df.loc[mask, "evidence_composite_equal"],
                model_df.loc[mask, "trait_semantic"],
            )
        else:
            corr_eq, p_eq = None, None

        # Correlation with trait Jaccard
        mask = (
            model_df[["evidence_composite_direction", "trait_jaccard"]]
            .notna()
            .all(axis=1)
        )
        if mask.sum() >= 3:
            corr_jac, p_jac = pearsonr(
                model_df.loc[mask, "evidence_composite_direction"],
                model_df.loc[mask, "trait_jaccard"],
            )
        else:
            corr_jac, p_jac = None, None

        results.append(
            {
                "model": model,
                "n_pairs": len(model_df),
                "corr_evidence_dir_trait_semantic": corr_dir,
                "p_value_dir_semantic": p_dir,
                "corr_evidence_equal_trait_semantic": corr_eq,
                "p_value_equal_semantic": p_eq,
                "corr_evidence_dir_trait_jaccard": corr_jac,
                "p_value_dir_jaccard": p_jac,
            }
        )

    res = pd.DataFrame(results)
    return res


def classify_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """Classify pairs into 4 quadrants using median splits.

    Quadrants:
    - high_trait_high_evidence: Both above median
    - high_trait_low_evidence: High trait, low evidence (divergent findings)
    - low_trait_high_evidence: Low trait, high evidence (analogous evidence)
    - low_trait_low_evidence: Both below median

    Args:
        df: DataFrame with matched similarity scores

    Returns:
        DataFrame with quadrant classification column added
    """
    logger.info("Classifying quadrants by model...")

    df = df.copy()
    df["quadrant"] = "unclassified"

    for model in df["query_model"].unique():
        model_mask = df["query_model"] == model
        model_df = df[model_mask]

        # Compute medians for this model
        trait_median = model_df["trait_semantic"].median()
        evidence_median = model_df["evidence_composite_direction"].median()

        # Classify quadrants
        high_trait = model_df["trait_semantic"] > trait_median
        high_evidence = (
            model_df["evidence_composite_direction"] > evidence_median
        )

        df.loc[model_mask & high_trait & high_evidence, "quadrant"] = (
            "high_trait_high_evidence"
        )
        df.loc[model_mask & high_trait & ~high_evidence, "quadrant"] = (
            "high_trait_low_evidence"
        )
        df.loc[model_mask & ~high_trait & high_evidence, "quadrant"] = (
            "low_trait_high_evidence"
        )
        df.loc[model_mask & ~high_trait & ~high_evidence, "quadrant"] = (
            "low_trait_low_evidence"
        )

    # Count quadrants by model
    quadrant_counts = (
        df.groupby(["query_model", "quadrant"])
        .size()
        .reset_index(name="count")
    )
    logger.info(f"\nQuadrant distribution:\n{quadrant_counts}")

    res = df
    return res


def identify_interesting_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cases in interesting quadrants.

    Focus on:
    - high_trait_low_evidence: Same traits, different evidence
    - low_trait_high_evidence: Different traits, similar evidence

    Args:
        df: DataFrame with quadrant classifications

    Returns:
        DataFrame containing only interesting cases
    """
    interesting_quadrants = [
        "high_trait_low_evidence",
        "low_trait_high_evidence",
    ]
    interesting_df = df[df["quadrant"].isin(interesting_quadrants)].copy()

    # Sort by extremeness of divergence
    interesting_df["divergence_score"] = abs(
        interesting_df["trait_semantic"]
        - interesting_df["evidence_composite_direction"]
    )
    interesting_df = interesting_df.sort_values(
        "divergence_score", ascending=False
    )

    logger.info(f"Identified {len(interesting_df)} interesting cases")
    logger.info(
        f"  - high_trait_low_evidence: {(interesting_df['quadrant'] == 'high_trait_low_evidence').sum()}"
    )
    logger.info(
        f"  - low_trait_high_evidence: {(interesting_df['quadrant'] == 'low_trait_high_evidence').sum()}"
    )

    res = interesting_df
    return res


# ==== Main execution ====


def main():
    """Execute trait vs evidence comparison analysis."""
    args = make_args()

    # ---- Validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating paths")
        if not args.evidence_db.exists():
            logger.error(f"Evidence database not found: {args.evidence_db}")
            return 1
        if not args.trait_db.exists():
            logger.error(f"Trait database not found: {args.trait_db}")
            return 1
        logger.info(f"Evidence database: {args.evidence_db}")
        logger.info(f"Trait database: {args.trait_db}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Dry run complete - paths validated")
        return 0

    # ---- Setup ----

    if not args.evidence_db.exists():
        logger.error(f"Evidence database not found: {args.evidence_db}")
        return 1

    if not args.trait_db.exists():
        logger.error(f"Trait database not found: {args.trait_db}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # ---- Connect to databases ----

    logger.info(f"Connecting to evidence database: {args.evidence_db}")
    try:
        evidence_conn = duckdb.connect(str(args.evidence_db), read_only=True)
    except Exception as e:
        logger.error(f"Failed to connect to evidence database: {e}")
        return 1

    logger.info(f"Connecting to trait database: {args.trait_db}")
    try:
        trait_conn = duckdb.connect(str(args.trait_db), read_only=True)
    except Exception as e:
        logger.error(f"Failed to connect to trait database: {e}")
        evidence_conn.close()
        return 1

    # ---- Load matched similarities ----

    logger.info("Loading matched similarity pairs...")
    matched_df = load_matched_similarities(evidence_conn, trait_conn)

    if len(matched_df) == 0:
        logger.warning("No matched pairs found between databases")
        logger.warning(
            "This is expected if evidence similarity computation has not been run yet"
        )
        evidence_conn.close()
        trait_conn.close()
        return 0

    # ---- Compute correlations ----

    logger.info("\nComputing correlations by model...")
    correlations = compute_correlation_by_model(matched_df)
    logger.info(f"Computed correlations for {len(correlations)} models")

    output_file = args.output_dir / "trait-vs-evidence-correlation.csv"
    correlations.to_csv(output_file, index=False)
    logger.info(f"Saved correlations: {output_file}")

    # Print summary
    logger.info("\nCorrelation Summary:")
    logger.info(f"\n{correlations.to_string()}")

    # ---- Classify quadrants ----

    logger.info("\nClassifying into quadrants...")
    classified_df = classify_quadrants(matched_df)

    output_file = args.output_dir / "quadrant-classification.csv"
    classified_df.to_csv(output_file, index=False)
    logger.info(f"Saved quadrant classifications: {output_file}")

    # ---- Identify interesting cases ----

    logger.info("\nIdentifying interesting cases...")
    interesting_df = identify_interesting_cases(classified_df)

    output_file = args.output_dir / "interesting-cases.csv"
    interesting_df.to_csv(output_file, index=False)
    logger.info(f"Saved interesting cases: {output_file}")

    # Print sample
    if len(interesting_df) > 0:
        logger.info("\nTop 5 Most Divergent Cases:")
        sample_cols = [
            "query_pmid",
            "similar_pmid",
            "quadrant",
            "trait_semantic",
            "evidence_composite_direction",
            "divergence_score",
        ]
        logger.info(f"\n{interesting_df[sample_cols].head(5).to_string()}")

    # ---- Cleanup ----

    evidence_conn.close()
    trait_conn.close()
    logger.info("\nAnalysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
