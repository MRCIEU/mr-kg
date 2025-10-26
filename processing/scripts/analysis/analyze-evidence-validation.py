"""Validation analyses for evidence profile similarity.

This script performs validation checks:
- Extract high-concordance pairs for face validity inspection
- Extract discordant pairs (negative direction concordance)
- Validate metric agreement between composite scores
- Compute rank stability across different scoring methods

Outputs:
- validation-report.json: Summary validation metrics
- top-similar-pairs.csv: Highest similarity pairs for inspection
- discordant-pairs.csv: Pairs with contradictory evidence
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau
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

    # ---- --top-n ----
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top similar pairs to extract (default: 50)",
    )

    res = parser.parse_args()
    return res


# ==== Analysis functions ====


def extract_top_similar_pairs(
    conn: duckdb.DuckDBPyConnection, top_n: int = 50
) -> pd.DataFrame:
    """Query high_concordance_pairs view for top pairs.

    Extract top N pairs by composite_similarity_direction.
    Include all metrics for inspection.

    Args:
        conn: DuckDB connection to evidence profile database
        top_n: Number of top pairs to extract

    Returns:
        DataFrame with top similar pairs
    """
    query = f"""
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
        composite_similarity_equal,
        composite_similarity_direction,
        similarity_rank
    FROM evidence_similarity_analysis
    WHERE similarity_rank <= {top_n}
    ORDER BY query_model, similarity_rank
    """
    res = conn.execute(query).df()
    return res


def extract_discordant_pairs(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query discordant_evidence_pairs view.

    Extract all pairs with direction_concordance < 0.
    Flag potential contradictions.

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        DataFrame with discordant pairs
    """
    query = """
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
        composite_similarity_equal,
        composite_similarity_direction
    FROM evidence_similarity_analysis
    WHERE direction_concordance < 0
    ORDER BY direction_concordance ASC
    """
    res = conn.execute(query).df()
    return res


def validate_metric_agreement(conn: duckdb.DuckDBPyConnection) -> Dict:
    """Assess agreement between composite scores.

    Compute:
    - Pearson correlation between composite scores
    - Kendall tau for rank agreement
    - Proportion of pairs with >0.1 difference in composite scores

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        Dictionary with agreement statistics
    """
    from scipy.stats import pearsonr

    query = """
    SELECT
        qc.model,
        es.composite_similarity_equal,
        es.composite_similarity_direction
    FROM evidence_similarities es
    JOIN query_combinations qc ON es.query_combination_id = qc.id
    """
    df = conn.execute(query).df()

    results = {}

    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()

        # Pearson correlation
        corr, p_value = pearsonr(
            model_df["composite_similarity_equal"],
            model_df["composite_similarity_direction"],
        )

        # Kendall tau
        tau, tau_p = kendalltau(
            model_df["composite_similarity_equal"],
            model_df["composite_similarity_direction"],
        )

        # Difference analysis
        model_df["score_diff"] = abs(
            model_df["composite_similarity_equal"]
            - model_df["composite_similarity_direction"]
        )
        large_diff = (model_df["score_diff"] > 0.1).sum()
        prop_large_diff = large_diff / len(model_df)

        results[model] = {
            "n_pairs": len(model_df),
            "pearson_correlation": float(corr),
            "pearson_p_value": float(p_value),
            "kendall_tau": float(tau),
            "kendall_p_value": float(tau_p),
            "n_large_difference": int(large_diff),
            "prop_large_difference": float(prop_large_diff),
            "mean_difference": float(model_df["score_diff"].mean()),
            "max_difference": float(model_df["score_diff"].max()),
        }

    res = results
    return res


def compute_rank_stability(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compare top-10 rankings by different composite scores.

    For each query:
    - Jaccard overlap of top-10 by equal vs direction-prioritized
    - Average rank difference for shared pairs

    Args:
        conn: DuckDB connection to evidence profile database

    Returns:
        DataFrame with rank stability metrics by query
    """
    query = """
    WITH equal_ranks AS (
        SELECT
            es.query_combination_id,
            es.similar_pmid,
            es.similar_model,
            ROW_NUMBER() OVER (
                PARTITION BY es.query_combination_id
                ORDER BY es.composite_similarity_equal DESC
            ) as rank_equal
        FROM evidence_similarities es
    ),
    direction_ranks AS (
        SELECT
            es.query_combination_id,
            es.similar_pmid,
            es.similar_model,
            ROW_NUMBER() OVER (
                PARTITION BY es.query_combination_id
                ORDER BY es.composite_similarity_direction DESC
            ) as rank_direction
        FROM evidence_similarities es
    ),
    top10_equal AS (
        SELECT * FROM equal_ranks WHERE rank_equal <= 10
    ),
    top10_direction AS (
        SELECT * FROM direction_ranks WHERE rank_direction <= 10
    )
    SELECT
        qc.pmid as query_pmid,
        qc.model as query_model,
        qc.title as query_title,
        
        -- Count in each top-10
        COUNT(DISTINCT te.similar_pmid) as n_top10_equal,
        COUNT(DISTINCT td.similar_pmid) as n_top10_direction,
        
        -- Count in both
        COUNT(DISTINCT CASE 
            WHEN te.similar_pmid IS NOT NULL 
            AND td.similar_pmid IS NOT NULL 
            THEN te.similar_pmid 
        END) as n_in_both,
        
        -- Jaccard overlap
        COUNT(DISTINCT CASE 
            WHEN te.similar_pmid IS NOT NULL 
            AND td.similar_pmid IS NOT NULL 
            THEN te.similar_pmid 
        END)::FLOAT / 
        (COUNT(DISTINCT te.similar_pmid) + 
         COUNT(DISTINCT td.similar_pmid) - 
         COUNT(DISTINCT CASE 
            WHEN te.similar_pmid IS NOT NULL 
            AND td.similar_pmid IS NOT NULL 
            THEN te.similar_pmid 
         END)) as jaccard_overlap
        
    FROM query_combinations qc
    LEFT JOIN top10_equal te ON qc.id = te.query_combination_id
    LEFT JOIN top10_direction td ON qc.id = td.query_combination_id
    GROUP BY qc.id, qc.pmid, qc.model, qc.title
    ORDER BY qc.model, qc.pmid
    """
    res = conn.execute(query).df()
    return res


# ==== Main execution ====


def main():
    """Execute validation analyses."""
    args = make_args()

    # ---- Validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating paths")
        if not args.evidence_db.exists():
            logger.error(f"Evidence database not found: {args.evidence_db}")
            return 1
        logger.info(f"Evidence database: {args.evidence_db}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Top-N: {args.top_n}")
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

    # ---- Extract top similar pairs ----

    logger.info(f"\nExtracting top {args.top_n} similar pairs...")
    top_pairs = extract_top_similar_pairs(conn, top_n=args.top_n)
    logger.info(f"Extracted {len(top_pairs)} high-concordance pairs")

    output_file = args.output_dir / "top-similar-pairs.csv"
    top_pairs.to_csv(output_file, index=False)
    logger.info(f"Saved top pairs: {output_file}")

    # Print sample
    if len(top_pairs) > 0:
        logger.info("\nSample of Top Pairs:")
        sample_cols = [
            "query_pmid",
            "similar_pmid",
            "direction_concordance",
            "composite_similarity_direction",
            "similarity_rank",
        ]
        logger.info(f"\n{top_pairs[sample_cols].head(5).to_string()}")

    # ---- Extract discordant pairs ----

    logger.info("\nExtracting discordant evidence pairs...")
    discordant = extract_discordant_pairs(conn)
    logger.info(f"Extracted {len(discordant)} discordant pairs")

    output_file = args.output_dir / "discordant-pairs.csv"
    discordant.to_csv(output_file, index=False)
    logger.info(f"Saved discordant pairs: {output_file}")

    # Print sample
    if len(discordant) > 0:
        logger.info("\nSample of Discordant Pairs:")
        sample_cols = [
            "query_pmid",
            "similar_pmid",
            "direction_concordance",
            "matched_pairs",
        ]
        logger.info(f"\n{discordant[sample_cols].head(5).to_string()}")

    # ---- Validate metric agreement ----

    logger.info("\nValidating metric agreement...")
    agreement = validate_metric_agreement(conn)
    logger.info(f"Computed agreement for {len(agreement)} models")

    # ---- Compute rank stability ----

    logger.info("\nComputing rank stability...")
    rank_stability = compute_rank_stability(conn)
    logger.info(f"Computed rank stability for {len(rank_stability)} queries")

    # Build validation report
    validation_report = {
        "metric_agreement": agreement,
        "rank_stability_summary": {
            "n_queries": len(rank_stability),
            "mean_jaccard_overlap": float(
                rank_stability["jaccard_overlap"].mean()
            ),
            "median_jaccard_overlap": float(
                rank_stability["jaccard_overlap"].median()
            ),
            "min_jaccard_overlap": float(
                rank_stability["jaccard_overlap"].min()
            ),
            "max_jaccard_overlap": float(
                rank_stability["jaccard_overlap"].max()
            ),
        },
        "top_pairs_summary": {
            "n_pairs_extracted": len(top_pairs),
            "models_represented": top_pairs["query_model"].nunique()
            if len(top_pairs) > 0
            else 0,
        },
        "discordant_pairs_summary": {
            "n_discordant_pairs": len(discordant),
            "models_with_discordance": discordant["query_model"].nunique()
            if len(discordant) > 0
            else 0,
        },
    }

    output_file = args.output_dir / "validation-report.json"
    with open(output_file, "w") as f:
        json.dump(validation_report, f, indent=2)
    logger.info(f"Saved validation report: {output_file}")

    # Print summary
    logger.info("\nValidation Summary:")
    logger.info(
        f"Top pairs extracted: {validation_report['top_pairs_summary']['n_pairs_extracted']}"
    )
    logger.info(
        f"Discordant pairs found: {validation_report['discordant_pairs_summary']['n_discordant_pairs']}"
    )
    logger.info(
        f"Mean rank stability (Jaccard): {validation_report['rank_stability_summary']['mean_jaccard_overlap']:.3f}"
    )

    # Print metric agreement
    logger.info("\nMetric Agreement by Model:")
    for model, stats in agreement.items():
        logger.info(f"  {model}:")
        logger.info(f"    Pearson r: {stats['pearson_correlation']:.3f}")
        logger.info(f"    Kendall tau: {stats['kendall_tau']:.3f}")
        logger.info(
            f"    Prop large diff: {stats['prop_large_difference']:.2%}"
        )

    # ---- Cleanup ----

    conn.close()
    logger.info("\nAnalysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
