"""Generate summary statistics and distributions for evidence similarity.

This script computes summary statistics by model, similarity score distributions,
and correlations between metrics from the evidence profile similarity database.

Outputs:
- summary-stats-by-model.csv: Model-level statistics
- similarity-distributions.csv: Percentile distributions by metric
- metric-correlations.csv: Pairwise metric correlations
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
    """Query model_evidence_stats view and compute summary metrics.

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        DataFrame with model-level statistics including:
        - total_combinations: Number of PMID-model combinations
        - avg_result_count: Average number of results per combination
        - avg_completeness: Average data completeness score
        - min/max_result_count: Range of result counts
        - total_similarity_pairs: Total number of similarity comparisons
    """
    query = """
    SELECT
        qc.model,
        COUNT(DISTINCT qc.id) as total_combinations,
        AVG(qc.result_count) as avg_result_count,
        AVG(qc.data_completeness) as avg_completeness,
        MIN(qc.result_count) as min_result_count,
        MAX(qc.result_count) as max_result_count,
        STDDEV(qc.result_count) as sd_result_count,
        COUNT(es.id) as total_similarity_pairs
    FROM query_combinations qc
    LEFT JOIN evidence_similarities es
        ON qc.id = es.query_combination_id
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
    - direction_concordance
    - effect_size_similarity
    - statistical_consistency
    - evidence_overlap
    - composite_similarity_equal
    - composite_similarity_direction

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        DataFrame with distribution statistics by model and metric
    """
    query = """
    SELECT
        qc.model,
        COUNT(*) as n_pairs,
        
        -- Direction concordance
        AVG(es.direction_concordance) as mean_direction_concordance,
        MEDIAN(es.direction_concordance) as median_direction_concordance,
        STDDEV(es.direction_concordance) as sd_direction_concordance,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY es.direction_concordance
        ) as p25_direction_concordance,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY es.direction_concordance
        ) as p75_direction_concordance,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY es.direction_concordance
        ) as p95_direction_concordance,
        
        -- Effect size similarity
        COUNT(es.effect_size_similarity) as n_effect_size,
        AVG(es.effect_size_similarity) as mean_effect_size_similarity,
        MEDIAN(es.effect_size_similarity) as median_effect_size_similarity,
        STDDEV(es.effect_size_similarity) as sd_effect_size_similarity,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY es.effect_size_similarity
        ) as p25_effect_size_similarity,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY es.effect_size_similarity
        ) as p75_effect_size_similarity,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY es.effect_size_similarity
        ) as p95_effect_size_similarity,
        
        -- Statistical consistency
        COUNT(es.statistical_consistency) as n_statistical_consistency,
        AVG(es.statistical_consistency) as mean_statistical_consistency,
        MEDIAN(es.statistical_consistency) as median_statistical_consistency,
        STDDEV(es.statistical_consistency) as sd_statistical_consistency,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY es.statistical_consistency
        ) as p25_statistical_consistency,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY es.statistical_consistency
        ) as p75_statistical_consistency,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY es.statistical_consistency
        ) as p95_statistical_consistency,
        
        -- Evidence overlap
        AVG(es.evidence_overlap) as mean_evidence_overlap,
        MEDIAN(es.evidence_overlap) as median_evidence_overlap,
        STDDEV(es.evidence_overlap) as sd_evidence_overlap,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY es.evidence_overlap
        ) as p25_evidence_overlap,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY es.evidence_overlap
        ) as p75_evidence_overlap,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY es.evidence_overlap
        ) as p95_evidence_overlap,
        
        -- Composite similarity equal
        AVG(es.composite_similarity_equal) as mean_composite_equal,
        MEDIAN(es.composite_similarity_equal) as median_composite_equal,
        STDDEV(es.composite_similarity_equal) as sd_composite_equal,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY es.composite_similarity_equal
        ) as p25_composite_equal,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY es.composite_similarity_equal
        ) as p75_composite_equal,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY es.composite_similarity_equal
        ) as p95_composite_equal,
        
        -- Composite similarity direction
        AVG(es.composite_similarity_direction) as mean_composite_direction,
        MEDIAN(es.composite_similarity_direction) as median_composite_direction,
        STDDEV(es.composite_similarity_direction) as sd_composite_direction,
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY es.composite_similarity_direction
        ) as p25_composite_direction,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY es.composite_similarity_direction
        ) as p75_composite_direction,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY es.composite_similarity_direction
        ) as p95_composite_direction
        
    FROM evidence_similarities es
    JOIN query_combinations qc ON es.query_combination_id = qc.id
    GROUP BY qc.model
    ORDER BY qc.model
    """
    res = conn.execute(query).df()
    return res


def compute_metric_correlations(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlations between similarity metrics.

    Computes correlation matrix for:
    - direction_concordance
    - effect_size_similarity
    - statistical_consistency
    - evidence_overlap
    - composite_similarity_equal
    - composite_similarity_direction

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        DataFrame with correlation coefficients by model
    """
    query = """
    SELECT
        qc.model,
        COUNT(*) as n_pairs,
        
        -- Correlations with direction concordance
        CORR(
            es.direction_concordance, es.effect_size_similarity
        ) as corr_direction_effect,
        CORR(
            es.direction_concordance, es.statistical_consistency
        ) as corr_direction_statistical,
        CORR(
            es.direction_concordance, es.evidence_overlap
        ) as corr_direction_overlap,
        
        -- Correlations with effect size similarity
        CORR(
            es.effect_size_similarity, es.statistical_consistency
        ) as corr_effect_statistical,
        CORR(
            es.effect_size_similarity, es.evidence_overlap
        ) as corr_effect_overlap,
        
        -- Correlations with statistical consistency
        CORR(
            es.statistical_consistency, es.evidence_overlap
        ) as corr_statistical_overlap,
        
        -- Composite score correlations
        CORR(
            es.composite_similarity_equal, es.composite_similarity_direction
        ) as corr_composite_equal_direction,
        CORR(
            es.composite_similarity_equal, es.direction_concordance
        ) as corr_composite_equal_direction_metric,
        CORR(
            es.composite_similarity_direction, es.direction_concordance
        ) as corr_composite_direction_direction_metric
        
    FROM evidence_similarities es
    JOIN query_combinations qc ON es.query_combination_id = qc.id
    GROUP BY qc.model
    ORDER BY qc.model
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
        if not args.evidence_db.exists():
            logger.error(f"Evidence database not found: {args.evidence_db}")
            return 1
        logger.info(f"Evidence database: {args.evidence_db}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Dry run complete - paths validated")
        return 0

    # ---- Setup ----

    if not args.evidence_db.exists():
        logger.error(f"Evidence database not found: {args.evidence_db}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # ---- Connect to database ----

    logger.info(f"Connecting to evidence database: {args.evidence_db}")
    try:
        conn = duckdb.connect(str(args.evidence_db), read_only=True)
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
            "mean_direction_concordance",
            "median_composite_direction",
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

    # ---- Cleanup ----

    conn.close()
    logger.info("\nAnalysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
