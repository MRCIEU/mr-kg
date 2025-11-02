"""Extract multi-study trait pairs for Case Study 1 reproducibility analysis.

This script performs a census of all trait pairs that appear in multiple
independent MR studies, extracting their direction concordance matrices
and metadata for downstream reproducibility analysis.

Outputs:
- multi_study_pairs.csv: Census of trait pairs with study counts
- multi_study_pairs_metadata.json: Metadata about extraction process
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd
import yaml
from loguru import logger
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


# ==== Data extraction functions ====


def extract_multi_study_pairs(
    conn: duckdb.DuckDBPyConnection,
    vector_store_db_path: Path,
    min_study_count: int,
) -> pd.DataFrame:
    """Extract trait pairs that appear across multiple independent studies.

    For each study in the database, identify other studies investigating
    the same trait pairs, compute summary statistics for direction
    concordance, and identify match types used. The study_count includes
    the focal study plus all comparison studies.

    Args:
        conn: DuckDB connection to evidence profile database
        vector_store_db_path: Path to vector store database for trait info
        min_study_count: Minimum number of studies for inclusion

    Returns:
        DataFrame with columns:
        - study1_pmid: PMID of focal study
        - study1_model: Model identifier of focal study
        - comparison_count: Number of comparison studies found
        - study_count: Total study count (comparison_count + 1)
        - mean_direction_concordance: Mean across all comparisons
        - median_direction_concordance: Median direction concordance
        - min_direction_concordance: Minimum direction concordance
        - max_direction_concordance: Maximum direction concordance
        - std_direction_concordance: Standard deviation
        - has_exact_match: Boolean for any exact matches
        - has_fuzzy_match: Boolean for any fuzzy matches
        - has_efo_match: Boolean for any EFO matches
        - total_matched_pairs: Sum of matched trait pairs
        - title: Study title
        - publication_year: Publication year
        - trait_pairs_json: JSON array of trait pair metadata
    """
    # ---- Attach vector store database ----
    conn.execute(f"ATTACH '{vector_store_db_path}' AS vector_db")

    query = """
    WITH study_pairs AS (
        SELECT
            qc1.pmid as study1_pmid,
            qc1.model as study1_model,
            qc2.pmid as study2_pmid,
            qc2.model as study2_model,
            es.matched_pairs,
            es.direction_concordance,
            es.match_type_exact,
            es.match_type_fuzzy,
            es.match_type_efo
        FROM evidence_similarities es
        JOIN query_combinations qc1
            ON es.query_combination_id = qc1.id
        JOIN query_combinations qc2
            ON es.similar_pmid = qc2.pmid
            AND es.similar_model = qc2.model
        WHERE es.matched_pairs >= 1
    ),
    study_aggregates AS (
        SELECT
            study1_pmid,
            study1_model,
            COUNT(DISTINCT study2_pmid || '|' || study2_model) as comparison_count,
            AVG(direction_concordance) as mean_direction_concordance,
            MEDIAN(direction_concordance) as median_direction_concordance,
            MIN(direction_concordance) as min_direction_concordance,
            MAX(direction_concordance) as max_direction_concordance,
            STDDEV(direction_concordance) as std_direction_concordance,
            BOOL_OR(match_type_exact) as has_exact_match,
            BOOL_OR(match_type_fuzzy) as has_fuzzy_match,
            BOOL_OR(match_type_efo) as has_efo_match,
            SUM(matched_pairs) as total_matched_pairs
        FROM study_pairs
        GROUP BY study1_pmid, study1_model
        HAVING comparison_count >= ?
    ),
    trait_pairs AS (
        SELECT
            mr.pmid,
            mr.model,
            json_group_array(
                json_object(
                    'exposure', json_extract(res.value, '$.exposure'),
                    'outcome', json_extract(res.value, '$.outcome')
                )
            ) as trait_pairs
        FROM vector_db.model_results mr,
        json_each(mr.results) res
        WHERE json_extract(res.value, '$.exposure') IS NOT NULL
          AND json_extract(res.value, '$.outcome') IS NOT NULL
        GROUP BY mr.pmid, mr.model
    )
    SELECT
        sa.*,
        sa.comparison_count + 1 as study_count,
        qc.title,
        qc.publication_year,
        COALESCE(tp.trait_pairs, '[]') as trait_pairs_json
    FROM study_aggregates sa
    JOIN query_combinations qc
        ON sa.study1_pmid = qc.pmid
        AND sa.study1_model = qc.model
    LEFT JOIN trait_pairs tp
        ON sa.study1_pmid = tp.pmid
        AND sa.study1_model = tp.model
    ORDER BY study_count DESC, mean_direction_concordance DESC
    """

    logger.info(f"Extracting studies with >= {min_study_count} total...")
    res = conn.execute(query, [min_study_count - 1]).df()

    logger.info(f"Extracted {len(res)} studies meeting criteria")
    return res


def compute_metadata(
    pairs_df: pd.DataFrame,
    config: Dict,
    extraction_params: Dict,
) -> Dict:
    """Compute metadata about the extraction process.

    Args:
        pairs_df: DataFrame of extracted trait pairs
        config: Configuration dictionary
        extraction_params: Parameters used for extraction

    Returns:
        Metadata dictionary with extraction statistics
    """
    metadata = {
        "extraction_params": extraction_params,
        "total_studies": len(pairs_df),
        "study_count_distribution": pairs_df["comparison_count"]
        .value_counts()
        .to_dict(),
        "direction_concordance_summary": {
            "mean": float(pairs_df["mean_direction_concordance"].mean()),
            "median": float(pairs_df["median_direction_concordance"].median()),
            "min": float(pairs_df["min_direction_concordance"].min()),
            "max": float(pairs_df["max_direction_concordance"].max()),
            "std": float(pairs_df["std_direction_concordance"].mean()),
        },
        "match_type_counts": {
            "exact": int(pairs_df["has_exact_match"].sum()),
            "fuzzy": int(pairs_df["has_fuzzy_match"].sum()),
            "efo": int(pairs_df["has_efo_match"].sum()),
        },
        "publication_year_range": {
            "min": (
                int(pairs_df["publication_year"].min())
                if len(pairs_df[pairs_df["publication_year"].notna()]) > 0
                else None
            ),
            "max": (
                int(pairs_df["publication_year"].max())
                if len(pairs_df[pairs_df["publication_year"].notna()]) > 0
                else None
            ),
        },
    }

    return metadata


# ==== Main execution ====


def main():
    """Execute multi-study trait pair extraction."""
    args = make_args()

    # ---- Load configuration ----

    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    cs1_config = config["case_study_1"]
    db_config = config["databases"]
    output_config = config["output"]["case_study_1"]

    # ---- Resolve paths ----

    evidence_db = PROJECT_ROOT / db_config["evidence_profile"]
    vector_store_db = PROJECT_ROOT / db_config["vector_store"]
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = PROJECT_ROOT / output_config["raw_pairs"]

    # ---- Validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating configuration and paths")
        logger.info(f"Evidence database: {evidence_db}")
        logger.info(f"Vector store database: {vector_store_db}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Min study count: {cs1_config['min_study_count']}")

        if not evidence_db.exists():
            logger.error(f"Evidence database not found: {evidence_db}")
            return 1

        if not vector_store_db.exists():
            logger.error(f"Vector store database not found: {vector_store_db}")
            return 1

        logger.info("Dry run complete - configuration validated")
        return 0

    # ---- Setup ----

    if not evidence_db.exists():
        logger.error(f"Evidence database not found: {evidence_db}")
        return 1

    if not vector_store_db.exists():
        logger.error(f"Vector store database not found: {vector_store_db}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ---- Connect to database ----

    logger.info(f"Connecting to evidence database: {evidence_db}")
    try:
        conn = duckdb.connect(str(evidence_db), read_only=True)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    # ---- Extract multi-study pairs ----

    min_study_count = cs1_config["min_study_count"]
    pairs_df = extract_multi_study_pairs(
        conn, vector_store_db, min_study_count
    )

    if len(pairs_df) == 0:
        logger.warning("No multi-study pairs found meeting criteria")
        conn.close()
        return 0

    # ---- Save results ----

    output_file = output_dir / "multi_study_pairs.csv"
    pairs_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(pairs_df)} studies: {output_file}")

    # ---- Compute and save metadata ----

    extraction_params = {
        "min_study_count": min_study_count,
        "evidence_database": str(evidence_db),
    }

    metadata = compute_metadata(pairs_df, config, extraction_params)
    metadata_file = output_dir / "multi_study_pairs_metadata.json"

    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata: {metadata_file}")

    # ---- Print summary ----

    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total studies extracted: {metadata['total_studies']}")
    logger.info(
        f"Direction concordance (mean): "
        f"{metadata['direction_concordance_summary']['mean']:.3f}"
    )
    logger.info(
        f"Direction concordance (median): "
        f"{metadata['direction_concordance_summary']['median']:.3f}"
    )
    logger.info("\nMatch type distribution:")
    logger.info(f"  Exact matches: {metadata['match_type_counts']['exact']}")
    logger.info(f"  Fuzzy matches: {metadata['match_type_counts']['fuzzy']}")
    logger.info(f"  EFO matches: {metadata['match_type_counts']['efo']}")

    if metadata["publication_year_range"]["min"]:
        logger.info(
            f"\nPublication year range: "
            f"{metadata['publication_year_range']['min']} - "
            f"{metadata['publication_year_range']['max']}"
        )

    logger.info("=" * 60)

    # ---- Cleanup ----

    conn.close()
    logger.info("\nExtraction complete!")

    return 0


if __name__ == "__main__":
    exit(main())
