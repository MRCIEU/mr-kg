"""Analyze temporal trends in reporting completeness for Case Study 5 (RQ4).

This script analyzes how reporting quality and completeness of MR studies
has evolved over time, testing the hypothesis that completeness improved
following STROBE-MR guidelines in 2021.

Research Question 4:
Has reporting completeness improved over time, particularly after STROBE-MR?

Input:
    - data/db/vector_store.db (model_results table)
    - data/processed/case-study-cs5/temporal/temporal_metadata.csv
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs5/completeness/
        field_completeness_by_year.csv
        field_completeness_by_era.csv
        field_type_by_era.csv
        strobe_impact_on_reporting.csv
        completeness_metadata.json
    - data/processed/case-study-cs5/figures/
        completeness_over_time.png/svg
        strobe_reporting_impact.png/svg
        completeness_by_field_type.png/svg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats

from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"


# ==== Helper functions ====


def convert_to_native_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {
            key: convert_to_native_types(value) for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


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
        help="Path to vector_store database (overrides default)",
    )

    # ---- --temporal-metadata ----
    parser.add_argument(
        "-t",
        "--temporal-metadata",
        type=Path,
        default=None,
        help="Path to temporal_metadata.csv (overrides default)",
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


def load_model_results(
    db_path: Path, model_filter: str = "gpt-5"
) -> pd.DataFrame:
    """Load model results from vector_store database.

    Args:
        db_path: Path to vector_store.db
        model_filter: Model to filter results (default: gpt-5)

    Returns:
        DataFrame with model results including parsed results JSON
    """
    logger.info(f"Loading model results from {db_path}...")

    con = duckdb.connect(str(db_path), read_only=True)

    query = """
    SELECT
        id,
        model,
        pmid,
        metadata,
        results
    FROM model_results
    WHERE model = ?
    ORDER BY pmid
    """

    df = con.execute(query, [model_filter]).fetchdf()
    con.close()

    logger.info(
        f"Loaded {len(df)} model result records for model {model_filter}"
    )

    # ---- Parse results JSON ----
    df["results_parsed"] = df["results"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    res = df
    return res


def load_temporal_metadata(temporal_path: Path) -> pd.DataFrame:
    """Load temporal metadata from CSV.

    Args:
        temporal_path: Path to temporal_metadata.csv

    Returns:
        DataFrame with temporal features
    """
    logger.info(f"Loading temporal metadata from {temporal_path}...")

    df = pd.read_csv(temporal_path)
    df["pmid"] = df["pmid"].astype(str)
    logger.info(f"Loaded temporal metadata for {len(df)} studies")

    res = df
    return res


def compute_field_completeness(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute completeness scores for extracted fields.

    Args:
        results_df: DataFrame with parsed results

    Returns:
        DataFrame with field completeness indicators
    """
    logger.info("Computing field completeness from results JSON...")

    # ---- Define fields to check ----
    fields_to_check = {
        "effect_size_beta": "beta",
        "effect_size_or": "odds ratio",
        "effect_size_hr": "hazard ratio",
        "confidence_interval": "95% CI",
        "standard_error": "SE",
        "p_value": "P-value",
        "direction": "direction",
    }

    completeness_data = []

    for _, row in results_df.iterrows():
        pmid = row["pmid"]
        results_parsed = row["results_parsed"]

        if not isinstance(results_parsed, list):
            results_parsed = [results_parsed]

        # ---- Check field presence across all results for this PMID ----
        field_presence = {field: False for field in fields_to_check.keys()}

        for result in results_parsed:
            if not isinstance(result, dict):
                continue

            for field_key, field_name in fields_to_check.items():
                value = result.get(field_name)
                if value is not None and value != "" and value != "N/A":
                    field_presence[field_key] = True

        completeness_data.append(
            {
                "pmid": pmid,
                **field_presence,
            }
        )

    df = pd.DataFrame(completeness_data)
    logger.info(f"Computed completeness for {len(df)} studies")

    res = df
    return res


def merge_with_temporal_metadata(
    completeness_df: pd.DataFrame, temporal_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge completeness data with temporal metadata.

    Args:
        completeness_df: DataFrame with field completeness
        temporal_df: DataFrame with temporal metadata

    Returns:
        Merged DataFrame with completeness and temporal features
    """
    logger.info("Merging completeness data with temporal metadata...")

    completeness_df["pmid"] = completeness_df["pmid"].astype(str)
    temporal_df["pmid"] = temporal_df["pmid"].astype(str)

    merged = completeness_df.merge(
        temporal_df[["pmid", "pub_year", "era"]],
        on="pmid",
        how="left",
    )

    # ---- Filter valid eras ----
    initial_count = len(merged)
    merged = merged[
        (merged["era"].notna()) & (merged["era"] != "unknown")
    ].copy()
    final_count = len(merged)

    if initial_count > final_count:
        logger.warning(
            f"Dropped {initial_count - final_count} studies with "
            f"missing or unknown era"
        )

    logger.info(f"Merged data contains {len(merged)} studies with valid eras")

    res = merged
    return res


def compute_completeness_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute field completeness statistics by year.

    Args:
        df: DataFrame with completeness and year columns

    Returns:
        DataFrame with yearly completeness statistics
    """
    logger.info("Computing completeness statistics by year...")

    field_columns = [
        "effect_size_beta",
        "effect_size_or",
        "effect_size_hr",
        "confidence_interval",
        "standard_error",
        "p_value",
        "direction",
    ]

    yearly_stats = []

    for year in sorted(df["pub_year"].unique()):
        year_data = df[df["pub_year"] == year]
        n_studies = len(year_data)

        year_row = {
            "pub_year": int(year),
            "n_studies": n_studies,
        }

        for field in field_columns:
            n_reported = year_data[field].sum()
            pct_complete = (n_reported / n_studies) * 100
            year_row[f"{field}_n"] = int(n_reported)
            year_row[f"{field}_pct"] = float(pct_complete)

        yearly_stats.append(year_row)

    result_df = pd.DataFrame(yearly_stats)

    logger.info(f"Computed completeness statistics for {len(result_df)} years")

    res = result_df
    return res


def compute_completeness_by_era(
    df: pd.DataFrame, era_defs: Dict[str, list]
) -> pd.DataFrame:
    """Compute field completeness statistics by era.

    Args:
        df: DataFrame with completeness and era columns
        era_defs: Dictionary of era definitions

    Returns:
        DataFrame with era-level completeness statistics
    """
    logger.info("Computing completeness statistics by era...")

    field_columns = [
        "effect_size_beta",
        "effect_size_or",
        "effect_size_hr",
        "confidence_interval",
        "standard_error",
        "p_value",
        "direction",
    ]

    era_stats = []

    for era in df["era"].unique():
        era_data = df[df["era"] == era]
        n_studies = len(era_data)

        era_row = {
            "era": era,
            "n_studies": n_studies,
            "era_start": era_defs.get(era, [None, None])[0],
            "era_end": era_defs.get(era, [None, None])[1],
        }

        for field in field_columns:
            n_reported = era_data[field].sum()
            pct_complete = (n_reported / n_studies) * 100
            era_row[f"{field}_n"] = int(n_reported)
            era_row[f"{field}_pct"] = float(pct_complete)

        era_stats.append(era_row)

    result_df = pd.DataFrame(era_stats)
    result_df = result_df.sort_values("era_start")

    logger.info(f"Computed completeness statistics for {len(result_df)} eras")

    res = result_df
    return res


def compute_field_type_by_era(
    df: pd.DataFrame, era_defs: Dict[str, list]
) -> pd.DataFrame:
    """Compute completeness by field type category and era.

    Args:
        df: DataFrame with completeness and era columns
        era_defs: Dictionary of era definitions

    Returns:
        DataFrame with field type x era statistics
    """
    logger.info("Computing completeness by field type and era...")

    field_types = {
        "effect_size": [
            "effect_size_beta",
            "effect_size_or",
            "effect_size_hr",
        ],
        "statistical": ["p_value", "standard_error"],
        "confidence_interval": ["confidence_interval"],
        "direction": ["direction"],
    }

    type_era_stats = []

    for era in df["era"].unique():
        era_data = df[df["era"] == era]
        n_studies = len(era_data)

        for field_type, fields in field_types.items():
            # ---- At least one field in category present ----
            any_present = era_data[fields].any(axis=1).sum()
            pct_any = (any_present / n_studies) * 100

            # ---- All fields in category present ----
            all_present = era_data[fields].all(axis=1).sum()
            pct_all = (all_present / n_studies) * 100

            type_era_stats.append(
                {
                    "era": era,
                    "field_type": field_type,
                    "n_studies": n_studies,
                    "n_any_present": int(any_present),
                    "pct_any_present": float(pct_any),
                    "n_all_present": int(all_present),
                    "pct_all_present": float(pct_all),
                    "era_start": era_defs.get(era, [None, None])[0],
                }
            )

    result_df = pd.DataFrame(type_era_stats)
    result_df = result_df.sort_values(["era_start", "field_type"])
    result_df = result_df.drop(columns=["era_start"])

    logger.info(
        f"Computed field type statistics for "
        f"{len(result_df)} era x field type combinations"
    )

    res = result_df
    return res


def perform_strobe_impact_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform analysis of STROBE-MR impact on reporting completeness.

    Tests for changes at 2021 breakpoint.

    Args:
        df: DataFrame with completeness and year columns

    Returns:
        Dictionary with impact analysis results
    """
    logger.info("Performing STROBE-MR impact analysis (2021 breakpoint)...")

    field_columns = [
        "effect_size_beta",
        "effect_size_or",
        "effect_size_hr",
        "confidence_interval",
        "standard_error",
        "p_value",
        "direction",
    ]

    # ---- Compute overall completeness score ----
    df["overall_completeness"] = df[field_columns].mean(axis=1)

    # ---- Split data at STROBE-MR breakpoint (2021) ----
    pre_strobe = df[df["pub_year"] < 2021].copy()
    post_strobe = df[df["pub_year"] >= 2021].copy()

    logger.info(f"Pre-STROBE studies (2003-2020): {len(pre_strobe)}")
    logger.info(f"Post-STROBE studies (2021-2025): {len(post_strobe)}")

    results = {}

    # ---- Pre-STROBE statistics ----
    if len(pre_strobe) > 0:
        results["pre_strobe"] = {
            "period": "2003-2020",
            "n_studies": int(len(pre_strobe)),
            "mean_completeness": float(
                pre_strobe["overall_completeness"].mean()
            ),
            "std_completeness": float(
                pre_strobe["overall_completeness"].std()
            ),
        }

        # ---- Field-specific statistics ----
        for field in field_columns:
            n_reported = pre_strobe[field].sum()
            pct_complete = (n_reported / len(pre_strobe)) * 100
            results["pre_strobe"][f"{field}_pct"] = float(pct_complete)

        logger.info(
            f"Pre-STROBE mean completeness: "
            f"{results['pre_strobe']['mean_completeness']:.3f}"
        )
    else:
        results["pre_strobe"] = None

    # ---- Post-STROBE statistics ----
    if len(post_strobe) > 0:
        results["post_strobe"] = {
            "period": "2021-2025",
            "n_studies": int(len(post_strobe)),
            "mean_completeness": float(
                post_strobe["overall_completeness"].mean()
            ),
            "std_completeness": float(
                post_strobe["overall_completeness"].std()
            ),
        }

        # ---- Field-specific statistics ----
        for field in field_columns:
            n_reported = post_strobe[field].sum()
            pct_complete = (n_reported / len(post_strobe)) * 100
            results["post_strobe"][f"{field}_pct"] = float(pct_complete)

        logger.info(
            f"Post-STROBE mean completeness: "
            f"{results['post_strobe']['mean_completeness']:.3f}"
        )
    else:
        results["post_strobe"] = None

    # ---- Compute change in completeness ----
    if results["pre_strobe"] and results["post_strobe"]:
        change = (
            results["post_strobe"]["mean_completeness"]
            - results["pre_strobe"]["mean_completeness"]
        )
        results["completeness_change"] = float(change)

        logger.info(f"Completeness change at 2021: {change:+.3f}")

        # ---- Statistical test for mean difference ----
        t_stat, p_val = stats.ttest_ind(
            pre_strobe["overall_completeness"],
            post_strobe["overall_completeness"],
        )

        results["mean_difference_test"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
        }

        logger.info(f"Mean difference test: t={t_stat:.3f}, p={p_val:.4f}")

        # ---- Field-specific tests ----
        field_tests = []
        for field in field_columns:
            pre_mean = results["pre_strobe"][f"{field}_pct"]
            post_mean = results["post_strobe"][f"{field}_pct"]
            change_pct = post_mean - pre_mean

            # ---- Chi-square test for proportions ----
            pre_reported = pre_strobe[field].sum()
            post_reported = post_strobe[field].sum()

            contingency = [
                [pre_reported, len(pre_strobe) - pre_reported],
                [post_reported, len(post_strobe) - post_reported],
            ]

            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

            field_tests.append(
                {
                    "field": field,
                    "pre_pct": float(pre_mean),
                    "post_pct": float(post_mean),
                    "change_pct": float(change_pct),
                    "chi2_statistic": float(chi2),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                }
            )

        results["field_specific_tests"] = field_tests

    else:
        results["completeness_change"] = None

    res = results
    return res


def plot_completeness_over_time(
    yearly_stats: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate visualization of completeness over time.

    Args:
        yearly_stats: DataFrame with yearly completeness statistics
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating completeness over time plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate completeness over time plot")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    # ---- Plot key fields ----
    fields_to_plot = {
        "p_value_pct": "P-value",
        "confidence_interval_pct": "95% CI",
        "standard_error_pct": "SE",
    }

    for field, label in fields_to_plot.items():
        ax.plot(
            yearly_stats["pub_year"],
            yearly_stats[field],
            marker="o",
            label=label,
            linewidth=2,
            markersize=4,
        )

    # ---- Add STROBE-MR marker (2021) ----
    ax.axvline(2021, color="red", linestyle="--", linewidth=2, alpha=0.5)
    ax.text(
        2021,
        ax.get_ylim()[1] * 0.95,
        "STROBE-MR (2021)",
        rotation=90,
        verticalalignment="top",
        fontsize=9,
        color="red",
    )

    # ---- Formatting ----
    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Reporting Completeness (%)", fontsize=12)
    ax.set_title("Reporting Completeness Over Time (GPT-5 Model)", fontsize=14)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"completeness_over_time.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def plot_strobe_reporting_impact(
    strobe_results: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate before/after STROBE-MR comparison plot.

    Args:
        strobe_results: Dictionary with STROBE impact analysis results
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating STROBE-MR reporting impact plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate STROBE impact plot")
        return

    if not strobe_results.get("field_specific_tests"):
        logger.warning("No field-specific tests available for plotting")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    # ---- Prepare data for plotting ----
    tests = strobe_results["field_specific_tests"]
    field_labels = [t["field"].replace("_", " ").title() for t in tests]
    pre_pcts = [t["pre_pct"] for t in tests]
    post_pcts = [t["post_pct"] for t in tests]

    x = np.arange(len(field_labels))
    width = 0.35

    # ---- Plot bars ----
    ax.bar(x - width / 2, pre_pcts, width, label="Pre-STROBE (2003-2020)")
    ax.bar(x + width / 2, post_pcts, width, label="Post-STROBE (2021-2025)")

    # ---- Add significance markers ----
    for i, test in enumerate(tests):
        if test["significant"]:
            y = max(pre_pcts[i], post_pcts[i]) + 5
            ax.text(i, y, "*", ha="center", fontsize=16, color="red")

    # ---- Formatting ----
    ax.set_xlabel("Field", fontsize=12)
    ax.set_ylabel("Reporting Completeness (%)", fontsize=12)
    ax.set_title("STROBE-MR Impact on Reporting Completeness", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(field_labels, rotation=45, ha="right")
    ax.set_ylim([0, 105])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"strobe_reporting_impact.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def plot_completeness_by_field_type(
    field_type_stats: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate completeness plot by field type and era.

    Args:
        field_type_stats: DataFrame with field type x era statistics
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating completeness by field type plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate field type plot")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    # ---- Plot lines for each field type ----
    for field_type in field_type_stats["field_type"].unique():
        type_data = field_type_stats[
            field_type_stats["field_type"] == field_type
        ]
        ax.plot(
            type_data["era"],
            type_data["pct_any_present"],
            marker="o",
            label=field_type.replace("_", " ").title(),
            linewidth=2,
        )

    # ---- Formatting ----
    ax.set_xlabel("Methodological Era", fontsize=12)
    ax.set_ylabel("Reporting Completeness (%)", fontsize=12)
    ax.set_title("Reporting Completeness by Field Type and Era", fontsize=14)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.xticks(rotation=45, ha="right")

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"completeness_by_field_type.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def create_completeness_metadata(
    yearly_stats: pd.DataFrame,
    era_stats: pd.DataFrame,
    strobe_results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create JSON metadata summary for completeness analysis.

    Args:
        yearly_stats: DataFrame with yearly statistics
        era_stats: DataFrame with era statistics
        strobe_results: Dictionary with STROBE impact analysis results
        config: Configuration dictionary

    Returns:
        Dictionary with metadata summary
    """
    logger.info("Creating completeness analysis metadata...")

    cs5_config = config["case_study_5"]

    metadata = {
        "research_question": (
            "Has reporting completeness improved over time, "
            "particularly after STROBE-MR?"
        ),
        "hypothesis": (
            "Reporting completeness improved following STROBE-MR "
            "guidelines in 2021"
        ),
        "model": cs5_config["models_included"][0],
        "total_studies": int(yearly_stats["n_studies"].sum()),
        "strobe_impact_analysis": convert_to_native_types(strobe_results),
        "era_statistics_summary": {
            "n_eras": len(era_stats),
            "eras": era_stats["era"].tolist(),
        },
        "year_range": {
            "min_year": int(yearly_stats["pub_year"].min()),
            "max_year": int(yearly_stats["pub_year"].max()),
        },
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
    logger.info("Case Study 5: Reporting completeness temporal trends (RQ4)")
    logger.info("=" * 60)

    # ---- Dry run check ----
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    # ---- Load configuration ----
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    cs5_config = config["case_study_5"]
    output_config = config["output"]["case_study_5"]

    # ---- Validate model filtering ----
    models_included = cs5_config["models_included"]
    assert models_included == ["gpt-5"], "CS5 must use gpt-5 model only"
    logger.info("VALIDATION: Model filtering set to gpt-5 only")

    # ---- Get database path ----
    if args.db:
        db_path = args.db
    else:
        db_path = DATA_DIR / "db" / "vector_store.db"

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # ---- Get temporal metadata path ----
    if args.temporal_metadata:
        temporal_path = args.temporal_metadata
    else:
        temporal_path = (
            Path(output_config["temporal"]) / "temporal_metadata.csv"
        )

    if not temporal_path.exists():
        logger.error(f"Temporal metadata not found: {temporal_path}")
        logger.error(
            "Run case_study_5_temporal_preparation.py first to generate it"
        )
        sys.exit(1)

    # ---- Create output directories ----
    completeness_dir = Path(output_config["completeness"])
    figures_dir = Path(output_config["figures"])

    if not args.dry_run:
        completeness_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directories")
    else:
        logger.info("Would create output directories")

    # ---- Load data ----
    model_filter = cs5_config["models_included"][0]
    results_df = load_model_results(db_path, model_filter=model_filter)
    temporal_df = load_temporal_metadata(temporal_path)

    # ---- Compute field completeness ----
    completeness_df = compute_field_completeness(results_df)

    # ---- Merge with temporal metadata ----
    df = merge_with_temporal_metadata(completeness_df, temporal_df)

    # ---- Compute statistics ----
    yearly_stats = compute_completeness_by_year(df)
    era_stats = compute_completeness_by_era(df, cs5_config["temporal_eras"])
    field_type_stats = compute_field_type_by_era(
        df, cs5_config["temporal_eras"]
    )

    # ---- Perform STROBE-MR impact analysis ----
    strobe_results = perform_strobe_impact_analysis(df)

    # ---- Create metadata ----
    metadata = create_completeness_metadata(
        yearly_stats, era_stats, strobe_results, config
    )

    # ---- Generate plots ----
    plot_completeness_over_time(
        yearly_stats, figures_dir, config, dry_run=args.dry_run
    )
    plot_strobe_reporting_impact(
        strobe_results, figures_dir, config, dry_run=args.dry_run
    )
    plot_completeness_by_field_type(
        field_type_stats, figures_dir, config, dry_run=args.dry_run
    )

    # ---- Write outputs ----
    if not args.dry_run:
        # ---- Write yearly statistics ----
        yearly_csv = completeness_dir / "field_completeness_by_year.csv"
        yearly_stats.to_csv(yearly_csv, index=False)
        logger.info(f"Wrote yearly statistics: {yearly_csv}")

        # ---- Write era statistics ----
        era_csv = completeness_dir / "field_completeness_by_era.csv"
        era_stats.to_csv(era_csv, index=False)
        logger.info(f"Wrote era statistics: {era_csv}")

        # ---- Write field type x era statistics ----
        field_type_csv = completeness_dir / "field_type_by_era.csv"
        field_type_stats.to_csv(field_type_csv, index=False)
        logger.info(f"Wrote field type x era statistics: {field_type_csv}")

        # ---- Write STROBE impact analysis ----
        if strobe_results.get("field_specific_tests"):
            strobe_df = pd.DataFrame(strobe_results["field_specific_tests"])
            strobe_csv = completeness_dir / "strobe_impact_on_reporting.csv"
            strobe_df.to_csv(strobe_csv, index=False)
            logger.info(f"Wrote STROBE impact analysis: {strobe_csv}")

        # ---- Write metadata JSON ----
        metadata_json = completeness_dir / "completeness_metadata.json"
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Wrote metadata summary: {metadata_json}")

        logger.info("=" * 60)
        logger.info("Reporting completeness analysis complete!")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("DRY RUN - Summary of what would be written:")
        logger.info(f"  {completeness_dir / 'field_completeness_by_year.csv'}")
        logger.info(f"    {len(yearly_stats)} rows (years)")
        logger.info(f"  {completeness_dir / 'field_completeness_by_era.csv'}")
        logger.info(f"    {len(era_stats)} rows (eras)")
        logger.info(f"  {completeness_dir / 'field_type_by_era.csv'}")
        logger.info(f"    {len(field_type_stats)} rows")
        logger.info(f"  {completeness_dir / 'strobe_impact_on_reporting.csv'}")
        logger.info("    Field-specific impact tests")
        logger.info(f"  {completeness_dir / 'completeness_metadata.json'}")
        logger.info("    Metadata summary with all results")
        logger.info("  Figures:")
        logger.info("    completeness_over_time.png/svg")
        logger.info("    strobe_reporting_impact.png/svg")
        logger.info("    completeness_by_field_type.png/svg")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
