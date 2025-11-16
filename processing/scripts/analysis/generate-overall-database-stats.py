"""Generate comprehensive overall statistics from the MR-KG vector store.

This script extracts comprehensive database-wide statistics from the main
vector store, including:
- Total unique PMIDs and traits
- Per-model extraction counts and statistics
- Temporal coverage and publication year distributions
- Trait usage patterns (exposures/outcomes)
- Model result statistics

Outputs:
- database-summary.csv: Overall database statistics
- database-summary.json: Same data in JSON format
- model-statistics.csv: Per-model detailed statistics
- temporal-statistics.csv: Publication year distributions
- trait-usage-statistics.csv: Top traits by usage patterns
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import duckdb
import pandas as pd
from loguru import logger
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
DEFAULT_VECTOR_DB = DB_DIR / "vector_store.db"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed" / "overall-stats"


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

    # ---- --vector-db ----
    parser.add_argument(
        "--vector-db",
        type=Path,
        default=DEFAULT_VECTOR_DB,
        help=f"Path to vector store database (default: {DEFAULT_VECTOR_DB})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for statistics (default: {DEFAULT_OUTPUT_DIR})",
    )

    res = parser.parse_args()
    return res


# ==== Analysis functions ====


def compute_overall_statistics(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, Any]:
    """Compute overall database-wide statistics.

    Args:
        conn: DuckDB connection to vector store database

    Returns:
        Dictionary with overall statistics including:
        - total_unique_pmids: Total unique papers
        - total_unique_traits: Total unique traits
        - total_model_results: Total model extraction records
        - total_unique_models: Number of extraction models
        - temporal_range_start: Earliest publication year
        - temporal_range_end: Latest publication year
        - avg_results_per_pmid: Average model results per paper
    """
    query = """
    WITH overall_counts AS (
        SELECT
            COUNT(DISTINCT mr.pmid) as total_unique_pmids,
            COUNT(DISTINCT mr.id) as total_model_results,
            COUNT(DISTINCT mr.model) as total_unique_models
        FROM model_results mr
    ),
    trait_counts AS (
        SELECT COUNT(DISTINCT trait_index) as total_unique_traits
        FROM trait_embeddings
    ),
    temporal_range AS (
        SELECT
            MIN(CAST(SUBSTR(pub_date, 1, 4) AS INTEGER)) 
                as temporal_range_start,
            MAX(CAST(SUBSTR(pub_date, 1, 4) AS INTEGER)) 
                as temporal_range_end
        FROM mr_pubmed_data
    ),
    pmid_avg AS (
        SELECT AVG(results_per_pmid) as avg_results_per_pmid
        FROM (
            SELECT pmid, COUNT(*) as results_per_pmid
            FROM model_results
            GROUP BY pmid
        )
    )
    SELECT
        oc.total_unique_pmids,
        tc.total_unique_traits,
        oc.total_model_results,
        oc.total_unique_models,
        tr.temporal_range_start,
        tr.temporal_range_end,
        pa.avg_results_per_pmid
    FROM overall_counts oc, trait_counts tc, temporal_range tr, pmid_avg pa
    """
    result = conn.execute(query).fetchone()

    res = {
        "total_unique_pmids": result[0],
        "total_unique_traits": result[1],
        "total_model_results": result[2],
        "total_unique_models": result[3],
        "temporal_range_start": result[4],
        "temporal_range_end": result[5],
        "avg_results_per_pmid": round(result[6], 2) if result[6] else 0,
    }
    return res


def compute_model_statistics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Compute per-model extraction statistics.

    Args:
        conn: DuckDB connection to vector store database

    Returns:
        DataFrame with per-model statistics including:
        - model: Model identifier
        - extraction_count: Number of extraction records
        - unique_pmids: Unique papers processed
        - avg_traits_per_extraction: Average traits extracted
        - total_traits_extracted: Total trait mentions
        - avg_results_per_extraction: Average MR results per extraction
    """
    query = """
    SELECT
        mr.model,
        COUNT(DISTINCT mr.id) as extraction_count,
        COUNT(DISTINCT mr.pmid) as unique_pmids,
        AVG(trait_counts.trait_count) as avg_traits_per_extraction,
        SUM(trait_counts.trait_count) as total_traits_extracted,
        AVG(JSON_ARRAY_LENGTH(mr.results)) as avg_results_per_extraction,
        SUM(JSON_ARRAY_LENGTH(mr.results)) as total_results_extracted
    FROM model_results mr
    LEFT JOIN (
        SELECT model_result_id, COUNT(*) as trait_count
        FROM model_result_traits
        GROUP BY model_result_id
    ) trait_counts ON mr.id = trait_counts.model_result_id
    GROUP BY mr.model
    ORDER BY mr.model
    """
    res = conn.execute(query).df()
    return res


def compute_temporal_statistics(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Compute publication year distribution statistics.

    Args:
        conn: DuckDB connection to vector store database

    Returns:
        DataFrame with temporal statistics including:
        - publication_year: Year of publication
        - paper_count: Number of papers published
        - cumulative_papers: Cumulative paper count
        - model_results_count: Number of model extractions
        - unique_models_count: Number of different models applied
    """
    query = """
    SELECT
        CAST(SUBSTR(mpd.pub_date, 1, 4) AS INTEGER) as publication_year,
        COUNT(DISTINCT mpd.pmid) as paper_count,
        SUM(COUNT(DISTINCT mpd.pmid)) 
            OVER (ORDER BY SUBSTR(mpd.pub_date, 1, 4)) as cumulative_papers,
        COUNT(DISTINCT mr.id) as model_results_count,
        COUNT(DISTINCT mr.model) as unique_models_count
    FROM mr_pubmed_data mpd
    LEFT JOIN model_results mr ON mpd.pmid = mr.pmid
    WHERE CAST(SUBSTR(mpd.pub_date, 1, 4) AS INTEGER) IS NOT NULL
    GROUP BY SUBSTR(mpd.pub_date, 1, 4)
    ORDER BY publication_year
    """
    res = conn.execute(query).df()
    return res


def compute_trait_usage_statistics(
    conn: duckdb.DuckDBPyConnection,
    top_n: int = 50,
) -> pd.DataFrame:
    """Compute trait usage statistics for most common traits.

    Analyzes how traits are used as exposures vs outcomes by examining
    the role field in model results.

    Args:
        conn: DuckDB connection to vector store database
        top_n: Number of top traits to return

    Returns:
        DataFrame with trait usage statistics including:
        - trait_label: Trait name
        - total_mentions: Total times trait appears
        - exposure_count: Times used as exposure
        - outcome_count: Times used as outcome
        - unique_pmids: Papers mentioning this trait
        - unique_models: Models that extracted this trait
    """
    query = f"""
    WITH trait_basic_stats AS (
        SELECT
            trait_label,
            COUNT(*) as total_mentions,
            COUNT(DISTINCT mr.pmid) as unique_pmids,
            COUNT(DISTINCT mr.model) as unique_models
        FROM model_result_traits mrt
        JOIN model_results mr ON mrt.model_result_id = mr.id
        GROUP BY trait_label
    ),
    result_elements AS (
        SELECT
            mr.id as model_result_id,
            json_extract_string(
                json_array_element, '$.exposure'
            ) as exposure_trait,
            json_extract_string(
                json_array_element, '$.outcome'
            ) as outcome_trait
        FROM model_results mr,
        LATERAL (
            SELECT unnest(
                json_transform(mr.results, '["JSON"]')
            ) as json_array_element
        ) AS exploded_results
    ),
    trait_role_counts AS (
        SELECT
            mrt.trait_label,
            SUM(CASE 
                WHEN LOWER(re.exposure_trait) LIKE '%' 
                    || LOWER(mrt.trait_label) || '%'
                THEN 1 ELSE 0 
            END) as exposure_count,
            SUM(CASE 
                WHEN LOWER(re.outcome_trait) LIKE '%' 
                    || LOWER(mrt.trait_label) || '%'
                THEN 1 ELSE 0 
            END) as outcome_count
        FROM model_result_traits mrt
        JOIN result_elements re ON mrt.model_result_id = re.model_result_id
        GROUP BY mrt.trait_label
    )
    SELECT
        tbs.trait_label,
        tbs.total_mentions,
        COALESCE(trc.exposure_count, 0) as exposure_count,
        COALESCE(trc.outcome_count, 0) as outcome_count,
        tbs.unique_pmids,
        tbs.unique_models,
        ROUND(
            COALESCE(trc.exposure_count, 0)::DOUBLE / tbs.total_mentions * 100,
            2
        ) as exposure_percentage,
        ROUND(
            COALESCE(trc.outcome_count, 0)::DOUBLE / tbs.total_mentions * 100,
            2
        ) as outcome_percentage
    FROM trait_basic_stats tbs
    LEFT JOIN trait_role_counts trc ON tbs.trait_label = trc.trait_label
    ORDER BY tbs.total_mentions DESC
    LIMIT {top_n}
    """
    res = conn.execute(query).df()
    return res


def compute_journal_statistics(
    conn: duckdb.DuckDBPyConnection,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute journal publication statistics.

    Args:
        conn: DuckDB connection to vector store database
        top_n: Number of top journals to return

    Returns:
        DataFrame with journal statistics including:
        - journal: Journal name
        - paper_count: Number of papers published
        - percentage: Percentage of total papers
    """
    query = f"""
    SELECT
        journal,
        COUNT(DISTINCT pmid) as paper_count,
        ROUND(
            COUNT(DISTINCT pmid)::DOUBLE / 
            (SELECT COUNT(DISTINCT pmid) FROM mr_pubmed_data) * 100, 2
        ) as percentage
    FROM mr_pubmed_data
    WHERE journal IS NOT NULL AND journal != ''
    GROUP BY journal
    ORDER BY paper_count DESC
    LIMIT {top_n}
    """
    res = conn.execute(query).df()
    return res


# ==== Main execution ====


def main():
    """Execute overall database statistics generation."""
    args = make_args()

    # ---- Validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating paths")
        if not args.vector_db.exists():
            logger.error(f"Vector store database not found: {args.vector_db}")
            return 1
        logger.info(f"Vector store database: {args.vector_db}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Dry run complete - paths validated")
        return 0

    # ---- Setup ----

    if not args.vector_db.exists():
        logger.error(f"Vector store database not found: {args.vector_db}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # ---- Connect to database ----

    logger.info(f"Connecting to vector store database: {args.vector_db}")
    try:
        conn = duckdb.connect(str(args.vector_db), read_only=True)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    # ---- Compute overall statistics ----

    logger.info("Computing overall database statistics...")
    overall_stats = compute_overall_statistics(conn)
    logger.info(f"Overall statistics computed: {overall_stats}")

    overall_df = pd.DataFrame([overall_stats])
    output_file = args.output_dir / "database-summary.csv"
    overall_df.to_csv(output_file, index=False)
    logger.info(f"Saved overall statistics: {output_file}")

    output_json = args.output_dir / "database-summary.json"
    with open(output_json, "w") as f:
        json.dump(overall_stats, f, indent=2)
    logger.info(f"Saved overall statistics JSON: {output_json}")

    # ---- Compute model statistics ----

    logger.info("\nComputing per-model statistics...")
    model_stats = compute_model_statistics(conn)
    logger.info(f"Computed statistics for {len(model_stats)} models")

    output_file = args.output_dir / "model-statistics.csv"
    model_stats.to_csv(output_file, index=False)
    logger.info(f"Saved model statistics: {output_file}")

    logger.info("\nModel Statistics Summary:")
    logger.info(f"\n{model_stats.to_string()}")

    # ---- Compute temporal statistics ----

    logger.info("\nComputing temporal statistics...")
    temporal_stats = compute_temporal_statistics(conn)
    logger.info(
        f"Computed temporal statistics for {len(temporal_stats)} years"
    )

    output_file = args.output_dir / "temporal-statistics.csv"
    temporal_stats.to_csv(output_file, index=False)
    logger.info(f"Saved temporal statistics: {output_file}")

    logger.info("\nTemporal Coverage:")
    if len(temporal_stats) > 0:
        logger.info(
            f"Year range: {temporal_stats['publication_year'].min()} - "
            f"{temporal_stats['publication_year'].max()}"
        )
        logger.info(f"Total papers: {temporal_stats['paper_count'].sum()}")

    # ---- Compute trait usage statistics ----

    logger.info("\nComputing trait usage statistics...")
    trait_usage = compute_trait_usage_statistics(conn)
    logger.info(f"Computed usage for {len(trait_usage)} top traits")

    output_file = args.output_dir / "trait-usage-statistics.csv"
    trait_usage.to_csv(output_file, index=False)
    logger.info(f"Saved trait usage statistics: {output_file}")

    logger.info("\nTop 10 Most Mentioned Traits:")
    if len(trait_usage) > 0:
        display_cols = ["trait_label", "total_mentions", "unique_pmids"]
        logger.info(f"\n{trait_usage[display_cols].head(10).to_string()}")

    # ---- Compute journal statistics ----

    logger.info("\nComputing journal statistics...")
    journal_stats = compute_journal_statistics(conn)
    logger.info(f"Computed statistics for {len(journal_stats)} top journals")

    output_file = args.output_dir / "journal-statistics.csv"
    journal_stats.to_csv(output_file, index=False)
    logger.info(f"Saved journal statistics: {output_file}")

    logger.info("\nTop 10 Journals:")
    if len(journal_stats) > 0:
        logger.info(f"\n{journal_stats.head(10).to_string()}")

    # ---- Cleanup ----

    conn.close()
    logger.info("\nAnalysis complete!")
    logger.info(f"All outputs saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
