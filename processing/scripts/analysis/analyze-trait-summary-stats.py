"""Generate summary statistics and distributions for trait similarity.

This script computes summary statistics by model, similarity score distributions,
and correlations between semantic and Jaccard similarity metrics from the trait
profile similarity database.

Outputs:
- summary-stats-by-model.csv: Model-level statistics
- similarity-distributions.csv: Percentile distributions by metric
- metric-correlations.csv: Correlation between semantic and Jaccard metrics
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
DEFAULT_TRAIT_DB = DB_DIR / "trait_profile_db.db"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed" / "trait-profiles" / "analysis"


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


def compute_model_statistics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query model_similarity_stats view and compute summary metrics.

    Args:
        conn: DuckDB connection to trait profile database

    Returns:
        DataFrame with model-level statistics including:
        - total_combinations: Number of PMID-model combinations
        - avg_trait_count: Average number of traits per combination
        - min_trait_count: Minimum trait count
        - max_trait_count: Maximum trait count
        - sd_trait_count: Standard deviation of trait counts
        - total_similarity_pairs: Total number of similarity comparisons
    """
    query = """
    SELECT
        qc.model,
        COUNT(DISTINCT qc.id) as total_combinations,
        AVG(qc.trait_count) as avg_trait_count,
        MIN(qc.trait_count) as min_trait_count,
        MAX(qc.trait_count) as max_trait_count,
        STDDEV(qc.trait_count) as sd_trait_count,
        COUNT(ts.id) as total_similarity_pairs
    FROM query_combinations qc
    LEFT JOIN trait_similarities ts
        ON qc.id = ts.query_combination_id
    GROUP BY qc.model
    ORDER BY qc.model
    """
    res = conn.execute(query).df()
    return res


def compute_similarity_distributions(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Compute percentile distributions for each similarity metric.

    For each model, compute mean, median, SD, and percentiles for:
    - trait_profile_similarity: Semantic similarity from embeddings
    - trait_jaccard_similarity: Set overlap similarity

    Args:
        conn: DuckDB connection to trait profile database

    Returns:
        DataFrame with distribution statistics by model and metric
    """
    query = """
    SELECT
        qc.model,
        COUNT(*) as n_pairs,
        
        -- Trait profile similarity (semantic)
        AVG(ts.trait_profile_similarity) as mean_semantic_similarity,
        MEDIAN(ts.trait_profile_similarity) as median_semantic_similarity,
        STDDEV(ts.trait_profile_similarity) as sd_semantic_similarity,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY ts.trait_profile_similarity
        ) as p25_semantic_similarity,
        PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY ts.trait_profile_similarity
        ) as p50_semantic_similarity,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY ts.trait_profile_similarity
        ) as p75_semantic_similarity,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY ts.trait_profile_similarity
        ) as p95_semantic_similarity,
        
        -- Trait Jaccard similarity
        AVG(ts.trait_jaccard_similarity) as mean_jaccard_similarity,
        MEDIAN(ts.trait_jaccard_similarity) as median_jaccard_similarity,
        STDDEV(ts.trait_jaccard_similarity) as sd_jaccard_similarity,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY ts.trait_jaccard_similarity
        ) as p25_jaccard_similarity,
        PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY ts.trait_jaccard_similarity
        ) as p50_jaccard_similarity,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY ts.trait_jaccard_similarity
        ) as p75_jaccard_similarity,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY ts.trait_jaccard_similarity
        ) as p95_jaccard_similarity,
        
        -- Trait count statistics
        AVG(ts.query_trait_count) as avg_query_trait_count,
        AVG(ts.similar_trait_count) as avg_similar_trait_count
        
    FROM trait_similarities ts
    JOIN query_combinations qc ON ts.query_combination_id = qc.id
    GROUP BY qc.model
    ORDER BY qc.model
    """
    res = conn.execute(query).df()
    return res


def compute_metric_correlations(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Compute Pearson correlation between semantic and Jaccard similarity.

    Args:
        conn: DuckDB connection to trait profile database

    Returns:
        DataFrame with correlation coefficients by model
    """
    query = """
    SELECT
        qc.model,
        COUNT(*) as n_pairs,
        
        -- Correlation between semantic and Jaccard
        CORR(
            ts.trait_profile_similarity, ts.trait_jaccard_similarity
        ) as corr_semantic_jaccard,
        
        -- Correlations with trait counts
        CORR(
            ts.trait_profile_similarity, ts.query_trait_count
        ) as corr_semantic_query_count,
        CORR(
            ts.trait_profile_similarity, ts.similar_trait_count
        ) as corr_semantic_similar_count,
        CORR(
            ts.trait_jaccard_similarity, ts.query_trait_count
        ) as corr_jaccard_query_count,
        CORR(
            ts.trait_jaccard_similarity, ts.similar_trait_count
        ) as corr_jaccard_similar_count
        
    FROM trait_similarities ts
    JOIN query_combinations qc ON ts.query_combination_id = qc.id
    GROUP BY qc.model
    ORDER BY qc.model
    """
    res = conn.execute(query).df()
    return res


def compute_trait_count_distributions(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Compute distribution of trait counts per study.

    Args:
        conn: DuckDB connection to trait profile database

    Returns:
        DataFrame with trait count distribution statistics by model
    """
    query = """
    SELECT
        model,
        COUNT(*) as n_studies,
        AVG(trait_count) as mean_trait_count,
        MEDIAN(trait_count) as median_trait_count,
        STDDEV(trait_count) as sd_trait_count,
        MIN(trait_count) as min_trait_count,
        MAX(trait_count) as max_trait_count,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY trait_count
        ) as p25_trait_count,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY trait_count
        ) as p75_trait_count,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY trait_count
        ) as p95_trait_count
    FROM query_combinations
    GROUP BY model
    ORDER BY model
    """
    res = conn.execute(query).df()
    return res


# ==== Main execution ====


def main():
    """Execute summary statistics analysis."""
    args = make_args()

    # ---- Validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating paths")
        if not args.trait_db.exists():
            logger.error(f"Trait database not found: {args.trait_db}")
            return 1
        logger.info(f"Trait database: {args.trait_db}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Dry run complete - paths validated")
        return 0

    # ---- Setup ----

    if not args.trait_db.exists():
        logger.error(f"Trait database not found: {args.trait_db}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # ---- Connect to database ----

    logger.info(f"Connecting to trait database: {args.trait_db}")
    try:
        conn = duckdb.connect(str(args.trait_db), read_only=True)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    # ---- Compute model statistics ----

    logger.info("Computing model-level statistics...")
    model_stats = compute_model_statistics(conn)
    logger.info(f"Computed statistics for {len(model_stats)} models")

    output_file = args.output_dir / "summary-stats-by-model.csv"
    model_stats.to_csv(output_file, index=False)
    logger.info(f"Saved model statistics: {output_file}")

    # Print summary
    logger.info("\nModel Statistics Summary:")
    logger.info(f"\n{model_stats.to_string()}")

    # ---- Compute similarity distributions ----

    logger.info("\nComputing similarity metric distributions...")
    distributions = compute_similarity_distributions(conn)
    logger.info(f"Computed distributions for {len(distributions)} models")

    output_file = args.output_dir / "similarity-distributions.csv"
    distributions.to_csv(output_file, index=False)
    logger.info(f"Saved distributions: {output_file}")

    # Print sample
    logger.info("\nDistribution Sample (first model):")
    if len(distributions) > 0:
        sample_cols = [
            "model",
            "n_pairs",
            "mean_semantic_similarity",
            "median_semantic_similarity",
            "mean_jaccard_similarity",
            "median_jaccard_similarity",
        ]
        logger.info(f"\n{distributions[sample_cols].head(1).to_string()}")

    # ---- Compute metric correlations ----

    logger.info("\nComputing metric correlations...")
    correlations = compute_metric_correlations(conn)
    logger.info(f"Computed correlations for {len(correlations)} models")

    output_file = args.output_dir / "metric-correlations.csv"
    correlations.to_csv(output_file, index=False)
    logger.info(f"Saved correlations: {output_file}")

    # Print summary
    logger.info("\nMetric Correlations Summary:")
    logger.info(f"\n{correlations.to_string()}")

    # ---- Compute trait count distributions ----

    logger.info("\nComputing trait count distributions...")
    trait_counts = compute_trait_count_distributions(conn)
    logger.info(
        f"Computed trait count distributions for {len(trait_counts)} models"
    )

    output_file = args.output_dir / "trait-count-distributions.csv"
    trait_counts.to_csv(output_file, index=False)
    logger.info(f"Saved trait count distributions: {output_file}")

    # Print summary
    logger.info("\nTrait Count Distributions:")
    logger.info(f"\n{trait_counts.to_string()}")

    # ---- Cleanup ----

    conn.close()
    logger.info("\nAnalysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
