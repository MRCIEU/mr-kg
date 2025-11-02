"""Temporal data preparation for Case Study 5.

This script prepares temporal metadata for all CS5 analyses by parsing
publication dates, assigning studies to methodological eras, and computing
temporal features. All analyses filter to gpt-5 model only.

Input:
    - data/db/vector_store.db (mr_pubmed_data, model_results)
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs5/temporal/
        temporal_metadata.csv
        era_statistics.csv
        temporal_metadata.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # ---- --config ----
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to configuration file",
    )

    # ---- --db ----
    parser.add_argument(
        "-d",
        "--db",
        type=Path,
        default=None,
        help="Path to vector_store database (overrides config)",
    )

    res = parser.parse_args()
    return res


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        res = yaml.safe_load(f)
    return res


def parse_pub_date(date_str: str) -> Tuple[int, bool]:
    """Parse publication date string to extract year.

    Handles formats: YYYY-MM-DD, YYYY-MM, YYYY

    Args:
        date_str: Publication date string

    Returns:
        Tuple of (year as int, success flag as bool)
    """
    if pd.isna(date_str) or date_str is None or str(date_str).strip() == "":
        return None, False

    date_str = str(date_str).strip()

    # ---- Try parsing YYYY-MM-DD or YYYY-MM ----
    if "-" in date_str:
        year_part = date_str.split("-")[0]
        try:
            year = int(year_part)
            return year, True
        except ValueError:
            return None, False

    # ---- Try parsing YYYY ----
    try:
        year = int(date_str)
        return year, True
    except ValueError:
        return None, False


def assign_era(year: int, era_defs: Dict[str, list]) -> str:
    """Assign study to temporal era based on publication year.

    Args:
        year: Publication year
        era_defs: Dictionary of era definitions {era_name: [start, end]}

    Returns:
        Era name string, or "unknown" if year not in defined eras
    """
    if pd.isna(year):
        return "unknown"

    for era_name, (start_year, end_year) in era_defs.items():
        if start_year <= year <= end_year:
            return era_name

    return "unknown"


def load_temporal_data(
    db_path: Path, model_filter: str = "gpt-5"
) -> pd.DataFrame:
    """Load publication dates and metadata from vector_store.

    Args:
        db_path: Path to vector_store.db
        model_filter: Model to filter results (default: gpt-5)

    Returns:
        DataFrame with pmid, model, pub_date
    """
    logger.info(f"Loading temporal data from {db_path}...")

    con = duckdb.connect(str(db_path), read_only=True)

    query = """
    SELECT
        mr.model,
        mr.pmid,
        mp.pub_date,
        mp.title
    FROM model_results mr
    LEFT JOIN mr_pubmed_data mp ON mr.pmid = mp.pmid
    WHERE mr.model = ?
    ORDER BY mr.pmid
    """

    df = con.execute(query, [model_filter]).fetchdf()
    con.close()

    logger.info(f"Loaded {len(df)} studies for model {model_filter}")

    # ---- Validate model filtering ----
    unique_models = df["model"].unique()
    if len(unique_models) != 1 or unique_models[0] != model_filter:
        logger.error(
            f"Model filtering failed: expected {model_filter}, "
            f"got {unique_models}"
        )
        sys.exit(1)

    logger.info(f"VALIDATION: All {len(df)} studies are model={model_filter}")

    res = df
    return res


def compute_temporal_features(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Compute temporal features from publication dates.

    Args:
        df: DataFrame with pub_date column
        config: Configuration dictionary with era definitions

    Returns:
        DataFrame with added temporal features
    """
    logger.info("Computing temporal features...")

    cs5_config = config["case_study_5"]
    era_defs = cs5_config["temporal_eras"]
    covid_years = cs5_config["covid_era_years"]

    # ---- Parse publication year ----
    results = df["pub_date"].apply(parse_pub_date)
    df["pub_year"] = results.apply(lambda x: x[0])
    df["parse_success"] = results.apply(lambda x: x[1])

    # ---- Count parsing failures ----
    n_failed = (~df["parse_success"]).sum()
    n_missing = df["pub_date"].isna().sum()
    logger.warning(f"Date parsing: {n_failed} failures, {n_missing} missing")

    # ---- Assign era ----
    df["era"] = df["pub_year"].apply(lambda y: assign_era(y, era_defs))

    # ---- Compute years since field inception ----
    inception_year = min(era_defs.values(), key=lambda x: x[0])[0]
    df["years_since_inception"] = df["pub_year"] - inception_year

    # ---- Flag COVID era ----
    covid_start, covid_end = covid_years
    df["covid_era_flag"] = (
        (df["pub_year"] >= covid_start) & (df["pub_year"] <= covid_end)
    ).astype(int)

    # ---- Summary statistics ----
    n_valid = df["pub_year"].notna().sum()
    n_unknown_era = (df["era"] == "unknown").sum()

    logger.info("Temporal features computed:")
    logger.info(f"  Valid years: {n_valid} / {len(df)}")
    logger.info(f"  Unknown era: {n_unknown_era}")
    logger.info(
        f"  Year range: {df['pub_year'].min()} - {df['pub_year'].max()}"
    )

    res = df
    return res


def compute_era_statistics(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Compute summary statistics by era.

    Args:
        df: DataFrame with temporal metadata
        config: Configuration dictionary with era definitions

    Returns:
        DataFrame with era-level statistics
    """
    logger.info("Computing era statistics...")

    cs5_config = config["case_study_5"]
    era_defs = cs5_config["temporal_eras"]

    # ---- Filter to valid years ----
    df_valid = df[df["pub_year"].notna()].copy()

    # ---- Group by era ----
    era_stats = df_valid.groupby("era").agg(
        n_studies=("pmid", "count"),
        min_year=("pub_year", "min"),
        max_year=("pub_year", "max"),
    )

    # ---- Add era year ranges from config ----
    era_stats["era_start"] = era_stats.index.map(
        lambda x: era_defs.get(x, [None, None])[0]
    )
    era_stats["era_end"] = era_stats.index.map(
        lambda x: era_defs.get(x, [None, None])[1]
    )

    # ---- Compute percentages ----
    total_studies = len(df_valid)
    era_stats["percent_total"] = era_stats["n_studies"] / total_studies * 100

    # ---- Format year range ----
    era_stats["year_range"] = (
        era_stats["min_year"].astype(int).astype(str)
        + "-"
        + era_stats["max_year"].astype(int).astype(str)
    )

    # ---- Reorder columns ----
    era_stats = era_stats[
        [
            "era_start",
            "era_end",
            "n_studies",
            "percent_total",
            "min_year",
            "max_year",
            "year_range",
        ]
    ]

    # ---- Sort by era start year ----
    era_stats = era_stats.sort_values("era_start")

    logger.info(f"Era statistics computed for {len(era_stats)} eras")

    res = era_stats
    return res


def create_metadata_summary(
    df: pd.DataFrame, era_stats: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create JSON metadata summary.

    Args:
        df: DataFrame with temporal metadata
        era_stats: DataFrame with era statistics
        config: Configuration dictionary

    Returns:
        Dictionary with metadata summary
    """
    logger.info("Creating metadata summary...")

    cs5_config = config["case_study_5"]

    # ---- Count missing and invalid dates ----
    n_missing = df["pub_date"].isna().sum()
    n_parse_failed = (~df["parse_success"]).sum()
    n_valid = df["pub_year"].notna().sum()

    # ---- Year range ----
    valid_years = df[df["pub_year"].notna()]["pub_year"]
    year_min = int(valid_years.min()) if len(valid_years) > 0 else None
    year_max = int(valid_years.max()) if len(valid_years) > 0 else None

    # ---- Era definitions ----
    era_defs = cs5_config["temporal_eras"]

    metadata = {
        "total_studies": len(df),
        "model": cs5_config["models_included"][0],
        "date_parsing": {
            "n_valid": int(n_valid),
            "n_missing": int(n_missing),
            "n_parse_failed": int(n_parse_failed),
            "percent_valid": round(n_valid / len(df) * 100, 2),
        },
        "year_range": {
            "min": year_min,
            "max": year_max,
        },
        "era_definitions": {
            era: {"start": years[0], "end": years[1]}
            for era, years in era_defs.items()
        },
        "era_statistics": era_stats.to_dict(orient="index"),
        "covid_era_years": cs5_config["covid_era_years"],
    }

    res = metadata
    return res


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # ---- Configure logger ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=" * 60)
    logger.info("Case Study 5: Temporal data preparation")
    logger.info("=" * 60)

    # ---- Dry run check ----
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    # ---- Load configuration ----
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    cs5_config = config["case_study_5"]
    output_config = config["output"]["case_study_5"]

    # ---- Get database path ----
    if args.db:
        db_path = args.db
    else:
        db_path = DATA_DIR / "db" / "vector_store.db"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # ---- Create output directories ----
    temporal_dir = PROJECT_ROOT / Path(output_config["temporal"])

    if not args.dry_run:
        temporal_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {temporal_dir}")
    else:
        logger.info(f"Would create output directory: {temporal_dir}")

    # ---- Load temporal data ----
    model_filter = cs5_config["models_included"][0]
    df = load_temporal_data(db_path, model_filter=model_filter)

    # ---- Compute temporal features ----
    df = compute_temporal_features(df, config)

    # ---- Compute era statistics ----
    era_stats = compute_era_statistics(df, config)

    # ---- Create metadata summary ----
    metadata = create_metadata_summary(df, era_stats, config)

    # ---- Select columns for output ----
    output_columns = [
        "pmid",
        "model",
        "pub_date",
        "pub_year",
        "era",
        "years_since_inception",
        "covid_era_flag",
        "parse_success",
    ]
    df_output = df[output_columns].copy()

    # ---- Write outputs ----
    if not args.dry_run:
        # ---- Write temporal metadata ----
        temporal_csv = temporal_dir / "temporal_metadata.csv"
        df_output.to_csv(temporal_csv, index=False)
        logger.info(f"Wrote temporal metadata: {temporal_csv}")

        # ---- Write era statistics ----
        era_csv = temporal_dir / "era_statistics.csv"
        era_stats.to_csv(era_csv)
        logger.info(f"Wrote era statistics: {era_csv}")

        # ---- Write metadata JSON ----
        metadata_json = temporal_dir / "temporal_metadata.json"
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Wrote metadata summary: {metadata_json}")

        logger.info("=" * 60)
        logger.info("Temporal preparation complete!")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("DRY RUN - Summary of what would be written:")
        logger.info(f"  {temporal_dir / 'temporal_metadata.csv'}")
        logger.info(
            f"    {len(df_output)} rows x {len(output_columns)} columns"
        )
        logger.info(f"  {temporal_dir / 'era_statistics.csv'}")
        logger.info(f"    {len(era_stats)} rows (eras)")
        logger.info(f"  {temporal_dir / 'temporal_metadata.json'}")
        logger.info("    Metadata summary with era definitions")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
