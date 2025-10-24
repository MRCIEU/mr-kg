"""Assess data quality and completeness patterns.

This script analyzes data quality metrics from the evidence profile database:
- Data completeness distribution by model
- Matched pairs distribution
- Missing data patterns (NULL values in similarity metrics)

Outputs:
- data-quality-report.json: Summary report with key statistics
- completeness-by-model.csv: Completeness distribution metrics
- matched-pairs-distribution.csv: Distribution of matched pairs counts
"""

import argparse
import json
from pathlib import Path
from typing import Dict

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


def assess_completeness_by_model(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Analyze data_completeness distribution by model.

    For each model:
    - Mean, median, SD of data_completeness
    - Proportion with completeness <0.5, 0.5-0.75, >0.75
    - Min, max completeness values

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        DataFrame with completeness metrics by model
    """
    query = """
    SELECT
        model,
        COUNT(*) as n_combinations,
        AVG(data_completeness) as mean_completeness,
        MEDIAN(data_completeness) as median_completeness,
        STDDEV(data_completeness) as sd_completeness,
        MIN(data_completeness) as min_completeness,
        MAX(data_completeness) as max_completeness,
        
        -- Completeness categories
        SUM(CASE WHEN data_completeness < 0.5 THEN 1 ELSE 0 END) as n_low,
        SUM(CASE WHEN data_completeness >= 0.5 
            AND data_completeness < 0.75 THEN 1 ELSE 0 END) as n_medium,
        SUM(CASE WHEN data_completeness >= 0.75 THEN 1 ELSE 0 END) as n_high,
        
        -- Proportions
        SUM(CASE WHEN data_completeness < 0.5 THEN 1 ELSE 0 END)::FLOAT 
            / COUNT(*) as prop_low,
        SUM(CASE WHEN data_completeness >= 0.5 
            AND data_completeness < 0.75 THEN 1 ELSE 0 END)::FLOAT 
            / COUNT(*) as prop_medium,
        SUM(CASE WHEN data_completeness >= 0.75 THEN 1 ELSE 0 END)::FLOAT 
            / COUNT(*) as prop_high
        
    FROM query_combinations
    GROUP BY model
    ORDER BY model
    """
    res = conn.execute(query).df()
    return res


def analyze_matched_pairs_distribution(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Analyze distribution of matched_pairs in similarities.

    For each model:
    - Mean, median, SD of matched_pairs
    - Proportion with <3, 3-5, 6-10, >10 matched pairs
    - Distribution by percentiles

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        DataFrame with matched pairs distribution by model
    """
    query = """
    SELECT
        qc.model,
        COUNT(*) as n_pairs,
        AVG(es.matched_pairs) as mean_matched_pairs,
        MEDIAN(es.matched_pairs) as median_matched_pairs,
        STDDEV(es.matched_pairs) as sd_matched_pairs,
        MIN(es.matched_pairs) as min_matched_pairs,
        MAX(es.matched_pairs) as max_matched_pairs,
        
        -- Matched pairs categories
        SUM(CASE WHEN es.matched_pairs < 3 THEN 1 ELSE 0 END) as n_very_low,
        SUM(CASE WHEN es.matched_pairs >= 3 
            AND es.matched_pairs <= 5 THEN 1 ELSE 0 END) as n_low,
        SUM(CASE WHEN es.matched_pairs >= 6 
            AND es.matched_pairs <= 10 THEN 1 ELSE 0 END) as n_medium,
        SUM(CASE WHEN es.matched_pairs > 10 THEN 1 ELSE 0 END) as n_high,
        
        -- Proportions
        SUM(CASE WHEN es.matched_pairs < 3 THEN 1 ELSE 0 END)::FLOAT 
            / COUNT(*) as prop_very_low,
        SUM(CASE WHEN es.matched_pairs >= 3 
            AND es.matched_pairs <= 5 THEN 1 ELSE 0 END)::FLOAT 
            / COUNT(*) as prop_low,
        SUM(CASE WHEN es.matched_pairs >= 6 
            AND es.matched_pairs <= 10 THEN 1 ELSE 0 END)::FLOAT 
            / COUNT(*) as prop_medium,
        SUM(CASE WHEN es.matched_pairs > 10 THEN 1 ELSE 0 END)::FLOAT 
            / COUNT(*) as prop_high,
        
        -- Percentiles
        PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY es.matched_pairs
        ) as p25_matched_pairs,
        PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY es.matched_pairs
        ) as p50_matched_pairs,
        PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY es.matched_pairs
        ) as p75_matched_pairs,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY es.matched_pairs
        ) as p95_matched_pairs
        
    FROM evidence_similarities es
    JOIN query_combinations qc ON es.query_combination_id = qc.id
    GROUP BY qc.model
    ORDER BY qc.model
    """
    res = conn.execute(query).df()
    return res


def compute_missing_data_patterns(conn: duckdb.DuckDBPyConnection) -> Dict:
    """Analyze patterns of NULL values in similarity metrics.

    Count and proportion of NULL for:
    - effect_size_similarity (requires >=3 matched pairs with effect sizes)
    - statistical_consistency (requires variance in significance)

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        Dictionary with missing data statistics by model
    """
    query = """
    SELECT
        qc.model,
        COUNT(*) as total_pairs,
        
        -- Effect size similarity
        SUM(CASE WHEN es.effect_size_similarity IS NULL 
            THEN 1 ELSE 0 END) as n_missing_effect_size,
        SUM(CASE WHEN es.effect_size_similarity IS NULL 
            THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as prop_missing_effect_size,
        
        -- Statistical consistency
        SUM(CASE WHEN es.statistical_consistency IS NULL 
            THEN 1 ELSE 0 END) as n_missing_statistical,
        SUM(CASE WHEN es.statistical_consistency IS NULL 
            THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as prop_missing_statistical,
        
        -- Both metrics present
        SUM(CASE WHEN es.effect_size_similarity IS NOT NULL 
            AND es.statistical_consistency IS NOT NULL 
            THEN 1 ELSE 0 END) as n_both_present,
        SUM(CASE WHEN es.effect_size_similarity IS NOT NULL 
            AND es.statistical_consistency IS NOT NULL 
            THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as prop_both_present,
        
        -- At least one metric present
        SUM(CASE WHEN es.effect_size_similarity IS NOT NULL 
            OR es.statistical_consistency IS NOT NULL 
            THEN 1 ELSE 0 END) as n_at_least_one,
        SUM(CASE WHEN es.effect_size_similarity IS NOT NULL 
            OR es.statistical_consistency IS NOT NULL 
            THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as prop_at_least_one,
        
        -- Neither metric present
        SUM(CASE WHEN es.effect_size_similarity IS NULL 
            AND es.statistical_consistency IS NULL 
            THEN 1 ELSE 0 END) as n_neither,
        SUM(CASE WHEN es.effect_size_similarity IS NULL 
            AND es.statistical_consistency IS NULL 
            THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as prop_neither
        
    FROM evidence_similarities es
    JOIN query_combinations qc ON es.query_combination_id = qc.id
    GROUP BY qc.model
    ORDER BY qc.model
    """
    df = conn.execute(query).df()

    # Convert to dictionary format for JSON export
    res = {
        "missing_data_by_model": df.to_dict(orient="records"),
        "summary": {
            "total_models": len(df),
            "total_pairs": int(df["total_pairs"].sum()),
            "overall_prop_missing_effect_size": float(
                df["n_missing_effect_size"].sum() / df["total_pairs"].sum()
            ),
            "overall_prop_missing_statistical": float(
                df["n_missing_statistical"].sum() / df["total_pairs"].sum()
            ),
        },
    }
    return res


# ==== Main execution ====


def main():
    """Execute data quality assessment."""
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

    # ---- Assess completeness ----

    logger.info("Assessing data completeness by model...")
    completeness = assess_completeness_by_model(conn)
    logger.info(f"Analyzed completeness for {len(completeness)} models")

    output_file = args.output_dir / "completeness-by-model.csv"
    completeness.to_csv(output_file, index=False)
    logger.info(f"Saved completeness analysis: {output_file}")

    # Print summary
    logger.info("\nCompleteness Summary:")
    logger.info(f"\n{completeness.to_string()}")

    # ---- Analyze matched pairs ----

    logger.info("\nAnalyzing matched pairs distribution...")
    matched_pairs = analyze_matched_pairs_distribution(conn)
    logger.info(f"Analyzed matched pairs for {len(matched_pairs)} models")

    output_file = args.output_dir / "matched-pairs-distribution.csv"
    matched_pairs.to_csv(output_file, index=False)
    logger.info(f"Saved matched pairs analysis: {output_file}")

    # Print summary
    logger.info("\nMatched Pairs Summary:")
    if len(matched_pairs) > 0:
        summary_cols = [
            "model",
            "n_pairs",
            "mean_matched_pairs",
            "median_matched_pairs",
            "prop_very_low",
        ]
        logger.info(f"\n{matched_pairs[summary_cols].to_string()}")

    # ---- Compute missing data patterns ----

    logger.info("\nComputing missing data patterns...")
    missing_data = compute_missing_data_patterns(conn)
    logger.info(
        f"Analyzed missing data for {missing_data['summary']['total_models']} models"
    )

    output_file = args.output_dir / "data-quality-report.json"
    with open(output_file, "w") as f:
        json.dump(missing_data, f, indent=2)
    logger.info(f"Saved data quality report: {output_file}")

    # Print summary
    logger.info("\nMissing Data Summary:")
    logger.info(f"Total pairs: {missing_data['summary']['total_pairs']}")
    logger.info(
        f"Overall missing effect size: {missing_data['summary']['overall_prop_missing_effect_size']:.2%}"
    )
    logger.info(
        f"Overall missing statistical: {missing_data['summary']['overall_prop_missing_statistical']:.2%}"
    )

    # ---- Cleanup ----

    conn.close()
    logger.info("\nAnalysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
