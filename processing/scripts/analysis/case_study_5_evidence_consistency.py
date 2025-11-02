"""Analyze temporal trends in evidence consistency for Case Study 5 (RQ2).

This script analyzes how direction concordance between similar trait pairs
has evolved over time, testing the hypothesis that consistency has improved
by 5-10 percentage points post-2018 following methodological advances.

Research Question 2:
Has evidence consistency (direction concordance) improved over time?

Input:
    - data/db/evidence_profile_db.db (evidence_similarities, query_combinations)
    - data/processed/case-study-cs5/temporal/temporal_metadata.csv
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs5/consistency/
        concordance_by_era.csv
        concordance_by_year.csv
        era_comparison_tests.csv
        strobe_impact_analysis.csv
        concordance_by_match_type_era.csv
        consistency_metadata.json
    - data/processed/case-study-cs5/figures/
        concordance_over_time.png/svg
        strobe_impact.png/svg
        concordance_by_match_type.png/svg
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
        help="Path to evidence_profile_db database (overrides config)",
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


def load_evidence_similarities(
    db_path: Path, model_filter: str = "gpt-5"
) -> pd.DataFrame:
    """Load evidence similarities from evidence_profile_db.

    Args:
        db_path: Path to evidence_profile_db.db
        model_filter: Model to filter results (default: gpt-5)

    Returns:
        DataFrame with evidence similarity data
    """
    logger.info(f"Loading evidence similarities from {db_path}...")

    con = duckdb.connect(str(db_path), read_only=True)

    query = """
    SELECT
        qc1.pmid as pmid1,
        es.similar_pmid as pmid2,
        qc1.model,
        es.direction_concordance,
        CASE
            WHEN es.match_type_exact THEN 'exact'
            WHEN es.match_type_fuzzy THEN 'fuzzy'
            WHEN es.match_type_efo THEN 'efo'
            ELSE 'unknown'
        END as match_type,
        qc1.publication_year as pmid1_year,
        es.similar_publication_year as pmid2_year
    FROM evidence_similarities es
    INNER JOIN query_combinations qc1
        ON es.query_combination_id = qc1.id
    WHERE qc1.model = ? AND es.similar_model = ?
    ORDER BY qc1.pmid, es.similar_pmid
    """

    df = con.execute(query, [model_filter, model_filter]).fetchdf()
    con.close()

    logger.info(
        f"Loaded {len(df)} evidence similarity records "
        f"for model {model_filter}"
    )

    # ---- Validate model filtering ----
    unique_models = df["model"].unique()
    if len(unique_models) != 1 or unique_models[0] != model_filter:
        logger.error(
            f"Model filtering failed: expected {model_filter}, "
            f"got {unique_models}"
        )
        sys.exit(1)

    logger.info(f"VALIDATION: All {len(df)} records are model={model_filter}")

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


def assign_era_to_similarities(
    sim_df: pd.DataFrame, temporal_df: pd.DataFrame
) -> pd.DataFrame:
    """Assign eras to both studies in each similarity pair.

    Args:
        sim_df: DataFrame with evidence similarities
        temporal_df: DataFrame with temporal metadata

    Returns:
        DataFrame with era assignments
    """
    logger.info("Assigning eras to similarity pairs...")

    sim_df = sim_df.copy()
    sim_df["pmid1"] = sim_df["pmid1"].astype(str)
    sim_df["pmid2"] = sim_df["pmid2"].astype(str)

    # ---- Merge era information for both studies ----
    sim_df = sim_df.merge(
        temporal_df[["pmid", "era", "pub_year"]],
        left_on="pmid1",
        right_on="pmid",
        how="left",
        suffixes=("", "_pmid1"),
    )
    sim_df.rename(
        columns={"era": "era_pmid1", "pub_year": "year_pmid1"}, inplace=True
    )
    sim_df.drop(columns=["pmid"], inplace=True)

    sim_df = sim_df.merge(
        temporal_df[["pmid", "era", "pub_year"]],
        left_on="pmid2",
        right_on="pmid",
        how="left",
        suffixes=("", "_pmid2"),
    )
    sim_df.rename(
        columns={"era": "era_pmid2", "pub_year": "year_pmid2"}, inplace=True
    )
    sim_df.drop(columns=["pmid"], inplace=True)

    # ---- Filter out pairs with missing era information ----
    initial_count = len(sim_df)
    sim_df = sim_df[
        (sim_df["era_pmid1"].notna())
        & (sim_df["era_pmid2"].notna())
        & (sim_df["era_pmid1"] != "unknown")
        & (sim_df["era_pmid2"] != "unknown")
    ].copy()
    final_count = len(sim_df)

    if initial_count > final_count:
        logger.warning(
            f"Dropped {initial_count - final_count} pairs with "
            f"missing or unknown era"
        )

    logger.info(
        f"Assigned eras to {len(sim_df)} similarity pairs with valid eras"
    )

    res = sim_df
    return res


def compute_concordance_by_era(
    df: pd.DataFrame, era_defs: Dict[str, list]
) -> pd.DataFrame:
    """Compute concordance statistics by era.

    Uses the later study's era for each pair.

    Args:
        df: DataFrame with concordance and era columns
        era_defs: Dictionary of era definitions

    Returns:
        DataFrame with era-level concordance statistics
    """
    logger.info("Computing concordance statistics by era...")

    # ---- Use later study's era for each pair ----
    df = df.copy()
    df["pair_era"] = df.apply(
        lambda row: (
            row["era_pmid2"]
            if row["year_pmid2"] >= row["year_pmid1"]
            else row["era_pmid1"]
        ),
        axis=1,
    )

    era_stats = (
        df.groupby("pair_era")["direction_concordance"]
        .agg(
            [
                ("n_pairs", "count"),
                ("mean_concordance", "mean"),
                ("median_concordance", "median"),
                ("std_concordance", "std"),
                ("q25_concordance", lambda x: x.quantile(0.25)),
                ("q75_concordance", lambda x: x.quantile(0.75)),
                ("min_concordance", "min"),
                ("max_concordance", "max"),
            ]
        )
        .reset_index()
    )

    # ---- Add era start/end years ----
    era_stats["era_start"] = era_stats["pair_era"].map(
        lambda x: era_defs.get(x, [None, None])[0]
    )
    era_stats["era_end"] = era_stats["pair_era"].map(
        lambda x: era_defs.get(x, [None, None])[1]
    )

    # ---- Sort by era start ----
    era_stats = era_stats.sort_values("era_start")
    era_stats.rename(columns={"pair_era": "era"}, inplace=True)

    logger.info(f"Computed concordance statistics for {len(era_stats)} eras")

    res = era_stats
    return res


def compute_concordance_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute concordance statistics by publication year.

    Uses the later study's year for each pair.

    Args:
        df: DataFrame with concordance and year columns

    Returns:
        DataFrame with yearly concordance statistics
    """
    logger.info("Computing concordance statistics by year...")

    df = df.copy()
    df["pair_year"] = df[["year_pmid1", "year_pmid2"]].max(axis=1)

    yearly_stats = (
        df.groupby("pair_year")["direction_concordance"]
        .agg(
            [
                ("n_pairs", "count"),
                ("mean_concordance", "mean"),
                ("median_concordance", "median"),
                ("std_concordance", "std"),
            ]
        )
        .reset_index()
    )

    yearly_stats["pair_year"] = yearly_stats["pair_year"].astype(int)
    yearly_stats.rename(columns={"pair_year": "pub_year"}, inplace=True)

    logger.info(
        f"Computed concordance statistics for {len(yearly_stats)} years"
    )

    res = yearly_stats
    return res


def compute_concordance_by_match_type_era(
    df: pd.DataFrame, era_defs: Dict[str, list]
) -> pd.DataFrame:
    """Compute concordance statistics stratified by match type and era.

    Args:
        df: DataFrame with concordance, match_type, and era columns
        era_defs: Dictionary of era definitions

    Returns:
        DataFrame with match type x era concordance statistics
    """
    logger.info("Computing concordance by match type and era...")

    df = df.copy()
    df["pair_era"] = df.apply(
        lambda row: (
            row["era_pmid2"]
            if row["year_pmid2"] >= row["year_pmid1"]
            else row["era_pmid1"]
        ),
        axis=1,
    )

    match_era_stats = (
        df.groupby(["pair_era", "match_type"])["direction_concordance"]
        .agg(
            [
                ("n_pairs", "count"),
                ("mean_concordance", "mean"),
                ("std_concordance", "std"),
            ]
        )
        .reset_index()
    )

    # ---- Add era start years for sorting ----
    match_era_stats["era_start"] = match_era_stats["pair_era"].map(
        lambda x: era_defs.get(x, [None, None])[0]
    )

    match_era_stats = match_era_stats.sort_values(["era_start", "match_type"])
    match_era_stats.rename(columns={"pair_era": "era"}, inplace=True)
    match_era_stats.drop(columns=["era_start"], inplace=True)

    logger.info(
        f"Computed concordance for {len(match_era_stats)} "
        f"era x match type combinations"
    )

    res = match_era_stats
    return res


def perform_era_comparison_tests(
    df: pd.DataFrame, era_stats: pd.DataFrame
) -> Dict[str, Any]:
    """Perform statistical tests comparing concordance across eras.

    Tests:
    - ANOVA: Overall difference across eras
    - Linear regression: concordance ~ era (ordinal)
    - Post-hoc pairwise t-tests

    Args:
        df: DataFrame with concordance and era columns
        era_stats: DataFrame with era statistics

    Returns:
        Dictionary with test results
    """
    logger.info("Performing era comparison tests...")

    df = df.copy()
    df["pair_era"] = df.apply(
        lambda row: (
            row["era_pmid2"]
            if row["year_pmid2"] >= row["year_pmid1"]
            else row["era_pmid1"]
        ),
        axis=1,
    )

    # ---- Prepare groups for ANOVA ----
    era_names = era_stats["era"].tolist()
    groups = [
        df[df["pair_era"] == era]["direction_concordance"].values
        for era in era_names
    ]

    # ---- Perform ANOVA ----
    f_stat, anova_p = stats.f_oneway(*groups)

    logger.info("ANOVA results:")
    logger.info(f"  F-statistic: {f_stat:.4f}")
    logger.info(f"  P-value: {anova_p:.4e}")

    # ---- Linear regression with ordinal era encoding ----
    era_order = {era: i for i, era in enumerate(era_names)}
    df["era_ordinal"] = df["pair_era"].map(era_order)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["era_ordinal"], df["direction_concordance"]
    )

    logger.info("Linear regression (concordance ~ era ordinal):")
    logger.info(f"  Slope: {slope:.4f}")
    logger.info(f"  R-squared: {r_value**2:.4f}")
    logger.info(f"  P-value: {p_value:.4e}")

    # ---- Pairwise t-tests ----
    pairwise_tests = []
    n_comparisons = 0

    for i, era1 in enumerate(era_names):
        for era2 in era_names[i + 1 :]:
            group1 = df[df["pair_era"] == era1]["direction_concordance"].values
            group2 = df[df["pair_era"] == era2]["direction_concordance"].values

            t_stat, p_val = stats.ttest_ind(group1, group2)
            n_comparisons += 1

            pairwise_tests.append(
                {
                    "era1": era1,
                    "era2": era2,
                    "era1_mean": float(np.mean(group1)),
                    "era2_mean": float(np.mean(group2)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "n1": int(len(group1)),
                    "n2": int(len(group2)),
                }
            )

    # ---- Bonferroni correction ----
    bonferroni_alpha = 0.05 / n_comparisons

    for test in pairwise_tests:
        test["p_value_bonferroni_corrected"] = test["p_value"] * n_comparisons
        test["significant_bonferroni"] = test["p_value"] < bonferroni_alpha

    logger.info(
        f"Performed {n_comparisons} pairwise comparisons "
        f"(Bonferroni alpha: {bonferroni_alpha:.4f})"
    )

    test_results = {
        "anova": {
            "f_statistic": float(f_stat),
            "p_value": float(anova_p),
            "n_groups": len(era_names),
        },
        "linear_regression": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_err": float(std_err),
        },
        "pairwise_tests": pairwise_tests,
        "bonferroni_alpha": float(bonferroni_alpha),
        "n_comparisons": n_comparisons,
    }

    res = test_results
    return res


def perform_strobe_impact_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform interrupted time series analysis for STROBE-MR impact.

    Tests for level and slope changes at 2021 breakpoint.

    Args:
        df: DataFrame with concordance and year columns

    Returns:
        Dictionary with impact analysis results
    """
    logger.info("Performing STROBE-MR impact analysis (2021 breakpoint)...")

    df = df.copy()
    df["pair_year"] = df[["year_pmid1", "year_pmid2"]].max(axis=1)

    # ---- Split data at STROBE-MR breakpoint (2021) ----
    pre_strobe = df[df["pair_year"] < 2021].copy()
    post_strobe = df[df["pair_year"] >= 2021].copy()

    logger.info(f"Pre-STROBE pairs (2015-2020): {len(pre_strobe)}")
    logger.info(f"Post-STROBE pairs (2021-2024): {len(post_strobe)}")

    results = {}

    # ---- Pre-STROBE trend (2015-2020) ----
    if len(pre_strobe) >= 10 and pre_strobe["pair_year"].nunique() > 1:
        slope_pre, intercept_pre, r_val_pre, p_val_pre, stderr_pre = (
            stats.linregress(
                pre_strobe["pair_year"], pre_strobe["direction_concordance"]
            )
        )

        results["pre_strobe"] = {
            "period": "2015-2020",
            "n_pairs": int(len(pre_strobe)),
            "mean_concordance": float(
                pre_strobe["direction_concordance"].mean()
            ),
            "trend_slope": float(slope_pre),
            "intercept": float(intercept_pre),
            "r_squared": float(r_val_pre**2),
            "p_value": float(p_val_pre),
        }

        logger.info("Pre-STROBE trend:")
        logger.info(f"  Slope: {slope_pre:.4f} per year")
        logger.info(
            f"  Mean concordance: {results['pre_strobe']['mean_concordance']:.3f}"
        )
    else:
        reason = (
            "Insufficient data"
            if len(pre_strobe) < 10
            else "All pairs from same year"
        )
        logger.warning(f"{reason} for pre-STROBE trend analysis")
        results["pre_strobe"] = {
            "period": "2015-2020",
            "n_pairs": int(len(pre_strobe)),
            "mean_concordance": (
                float(pre_strobe["direction_concordance"].mean())
                if len(pre_strobe) > 0
                else None
            ),
            "trend_slope": None,
            "intercept": None,
            "r_squared": None,
            "p_value": None,
        }

    # ---- Post-STROBE trend (2021-2024) ----
    if len(post_strobe) >= 10 and post_strobe["pair_year"].nunique() > 1:
        slope_post, intercept_post, r_val_post, p_val_post, stderr_post = (
            stats.linregress(
                post_strobe["pair_year"], post_strobe["direction_concordance"]
            )
        )

        results["post_strobe"] = {
            "period": "2021-2024",
            "n_pairs": int(len(post_strobe)),
            "mean_concordance": float(
                post_strobe["direction_concordance"].mean()
            ),
            "trend_slope": float(slope_post),
            "intercept": float(intercept_post),
            "r_squared": float(r_val_post**2),
            "p_value": float(p_val_post),
        }

        logger.info("Post-STROBE trend:")
        logger.info(f"  Slope: {slope_post:.4f} per year")
        logger.info(
            f"  Mean concordance: {results['post_strobe']['mean_concordance']:.3f}"
        )
    else:
        reason = (
            "Insufficient data"
            if len(post_strobe) < 10
            else "All pairs from same year"
        )
        logger.warning(f"{reason} for post-STROBE trend analysis")
        results["post_strobe"] = {
            "period": "2021-2024",
            "n_pairs": int(len(post_strobe)),
            "mean_concordance": (
                float(post_strobe["direction_concordance"].mean())
                if len(post_strobe) > 0
                else None
            ),
            "trend_slope": None,
            "intercept": None,
            "r_squared": None,
            "p_value": None,
        }

    # ---- Compute level change and slope change ----
    if (
        results["pre_strobe"]["mean_concordance"] is not None
        and results["post_strobe"]["mean_concordance"] is not None
    ):
        level_change = (
            results["post_strobe"]["mean_concordance"]
            - results["pre_strobe"]["mean_concordance"]
        )
        results["level_change"] = float(level_change)

        logger.info(f"Level change at 2021: {level_change:+.3f}")
    else:
        results["level_change"] = None

    if (
        results["pre_strobe"]["trend_slope"] is not None
        and results["post_strobe"]["trend_slope"] is not None
    ):
        slope_change = (
            results["post_strobe"]["trend_slope"]
            - results["pre_strobe"]["trend_slope"]
        )
        results["slope_change"] = float(slope_change)

        logger.info(f"Slope change at 2021: {slope_change:+.4f}")
    else:
        results["slope_change"] = None

    # ---- Statistical test for mean difference ----
    if len(pre_strobe) > 0 and len(post_strobe) > 0:
        t_stat, p_val = stats.ttest_ind(
            pre_strobe["direction_concordance"],
            post_strobe["direction_concordance"],
        )

        results["mean_difference_test"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
        }

        logger.info(f"Mean difference test: t={t_stat:.3f}, p={p_val:.4f}")

    res = results
    return res


def plot_concordance_over_time(
    yearly_stats: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate visualization of concordance over time with era boundaries.

    Args:
        yearly_stats: DataFrame with yearly concordance statistics
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating concordance over time plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate concordance over time plot")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    # ---- Plot mean concordance with error bars ----
    x = yearly_stats["pub_year"]
    y = yearly_stats["mean_concordance"]
    yerr = yearly_stats["std_concordance"] / np.sqrt(yearly_stats["n_pairs"])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        capsize=3,
        label="Mean concordance (± SE)",
        alpha=0.7,
    )

    # ---- Add era boundaries ----
    era_defs = config["case_study_5"]["temporal_eras"]
    era_colors = {
        "early_mr": "#e0e0e0",
        "mr_egger": "#c0c0c0",
        "mr_presso": "#a0a0a0",
        "within_family": "#808080",
        "strobe_mr": "#606060",
    }

    for era_name, (start_year, end_year) in era_defs.items():
        ax.axvspan(
            start_year,
            end_year,
            alpha=0.1,
            color=era_colors.get(era_name, "#d0d0d0"),
            label=era_name.replace("_", " ").title(),
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
    ax.set_ylabel("Direction Concordance", fontsize=12)
    ax.set_title("Evidence Consistency Over Time (GPT-5 Model)", fontsize=14)
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"concordance_over_time.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def plot_strobe_impact(
    df: pd.DataFrame,
    strobe_results: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate before/after STROBE-MR comparison plot.

    Args:
        df: DataFrame with concordance and year columns
        strobe_results: Dictionary with STROBE impact analysis results
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating STROBE-MR impact plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate STROBE impact plot")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    df = df.copy()
    df["pair_year"] = df[["year_pmid1", "year_pmid2"]].max(axis=1)

    # ---- Split data ----
    pre_strobe = df[df["pair_year"] < 2021].copy()
    post_strobe = df[df["pair_year"] >= 2021].copy()

    # ---- Plot scatter points ----
    ax.scatter(
        pre_strobe["pair_year"],
        pre_strobe["direction_concordance"],
        alpha=0.3,
        s=20,
        label=f"Pre-STROBE (n={len(pre_strobe)})",
        color="steelblue",
    )
    ax.scatter(
        post_strobe["pair_year"],
        post_strobe["direction_concordance"],
        alpha=0.3,
        s=20,
        label=f"Post-STROBE (n={len(post_strobe)})",
        color="coral",
    )

    # ---- Plot trend lines ----
    if strobe_results["pre_strobe"]["trend_slope"] is not None:
        pre_years = np.array([2015, 2020])
        pre_trend = (
            strobe_results["pre_strobe"]["intercept"]
            + strobe_results["pre_strobe"]["trend_slope"] * pre_years
        )
        ax.plot(
            pre_years,
            pre_trend,
            "b--",
            linewidth=2,
            label="Pre-STROBE trend",
        )

    if strobe_results["post_strobe"]["trend_slope"] is not None:
        post_years = np.array([2021, 2024])
        post_trend = (
            strobe_results["post_strobe"]["intercept"]
            + strobe_results["post_strobe"]["trend_slope"] * post_years
        )
        ax.plot(
            post_years,
            post_trend,
            "r--",
            linewidth=2,
            label="Post-STROBE trend",
        )

    # ---- Add STROBE-MR marker ----
    ax.axvline(2021, color="black", linestyle="-", linewidth=2, alpha=0.7)

    # ---- Formatting ----
    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Direction Concordance", fontsize=12)
    ax.set_title("STROBE-MR Impact on Evidence Consistency", fontsize=14)
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"strobe_impact.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def plot_concordance_by_match_type(
    match_era_stats: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate stratified concordance plot by match type.

    Args:
        match_era_stats: DataFrame with match type x era statistics
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating concordance by match type plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate match type plot")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    # ---- Pivot data for plotting ----
    pivot_data = match_era_stats.pivot(
        index="era", columns="match_type", values="mean_concordance"
    )

    # ---- Plot lines for each match type ----
    for match_type in pivot_data.columns:
        ax.plot(
            pivot_data.index,
            pivot_data[match_type],
            marker="o",
            label=match_type.upper(),
            linewidth=2,
        )

    # ---- Formatting ----
    ax.set_xlabel("Methodological Era", fontsize=12)
    ax.set_ylabel("Mean Direction Concordance", fontsize=12)
    ax.set_title("Evidence Consistency by Match Type and Era", fontsize=14)
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.xticks(rotation=45, ha="right")

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"concordance_by_match_type.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def create_consistency_metadata(
    era_stats: pd.DataFrame,
    test_results: Dict[str, Any],
    strobe_results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create JSON metadata summary for consistency analysis.

    Args:
        era_stats: DataFrame with era statistics
        test_results: Dictionary with era comparison test results
        strobe_results: Dictionary with STROBE impact analysis results
        config: Configuration dictionary

    Returns:
        Dictionary with metadata summary
    """
    logger.info("Creating consistency analysis metadata...")

    cs5_config = config["case_study_5"]

    metadata = {
        "research_question": (
            "Has evidence consistency (direction concordance) "
            "improved over time?"
        ),
        "hypothesis": (
            "5-10 percentage point improvement in direction concordance "
            "post-2018"
        ),
        "model": cs5_config["models_included"][0],
        "total_pairs": int(era_stats["n_pairs"].sum()),
        "era_comparison_tests": convert_to_native_types(test_results),
        "strobe_impact_analysis": convert_to_native_types(strobe_results),
        "era_statistics_summary": {
            "n_eras": len(era_stats),
            "eras": era_stats["era"].tolist(),
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
    logger.info("Case Study 5: Evidence consistency temporal trends (RQ2)")
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
        db_path = DATA_DIR / "db" / "evidence_profile_db.db"

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
    consistency_dir = Path(output_config["consistency"])
    figures_dir = Path(output_config["figures"])

    if not args.dry_run:
        consistency_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directories")
    else:
        logger.info("Would create output directories")

    # ---- Load data ----
    model_filter = cs5_config["models_included"][0]
    sim_df = load_evidence_similarities(db_path, model_filter=model_filter)
    temporal_df = load_temporal_metadata(temporal_path)

    # ---- Assign eras to similarity pairs ----
    df = assign_era_to_similarities(sim_df, temporal_df)

    # ---- Compute statistics ----
    era_stats = compute_concordance_by_era(df, cs5_config["temporal_eras"])
    yearly_stats = compute_concordance_by_year(df)
    match_era_stats = compute_concordance_by_match_type_era(
        df, cs5_config["temporal_eras"]
    )

    # ---- Perform statistical tests ----
    test_results = perform_era_comparison_tests(df, era_stats)
    strobe_results = perform_strobe_impact_analysis(df)

    # ---- Create metadata ----
    metadata = create_consistency_metadata(
        era_stats, test_results, strobe_results, config
    )

    # ---- Generate plots ----
    plot_concordance_over_time(
        yearly_stats, figures_dir, config, dry_run=args.dry_run
    )
    plot_strobe_impact(
        df, strobe_results, figures_dir, config, dry_run=args.dry_run
    )
    plot_concordance_by_match_type(
        match_era_stats, figures_dir, config, dry_run=args.dry_run
    )

    # ---- Write outputs ----
    if not args.dry_run:
        # ---- Write era statistics ----
        era_csv = consistency_dir / "concordance_by_era.csv"
        era_stats.to_csv(era_csv, index=False)
        logger.info(f"Wrote era statistics: {era_csv}")

        # ---- Write yearly statistics ----
        yearly_csv = consistency_dir / "concordance_by_year.csv"
        yearly_stats.to_csv(yearly_csv, index=False)
        logger.info(f"Wrote yearly statistics: {yearly_csv}")

        # ---- Write era comparison tests ----
        if test_results["pairwise_tests"]:
            tests_df = pd.DataFrame(test_results["pairwise_tests"])
            tests_csv = consistency_dir / "era_comparison_tests.csv"
            tests_df.to_csv(tests_csv, index=False)
            logger.info(f"Wrote era comparison tests: {tests_csv}")

        # ---- Write STROBE impact analysis ----
        strobe_df = pd.DataFrame(
            [
                strobe_results.get("pre_strobe", {}),
                strobe_results.get("post_strobe", {}),
            ]
        )
        strobe_csv = consistency_dir / "strobe_impact_analysis.csv"
        strobe_df.to_csv(strobe_csv, index=False)
        logger.info(f"Wrote STROBE impact analysis: {strobe_csv}")

        # ---- Write match type x era statistics ----
        match_csv = consistency_dir / "concordance_by_match_type_era.csv"
        match_era_stats.to_csv(match_csv, index=False)
        logger.info(f"Wrote match type x era statistics: {match_csv}")

        # ---- Write metadata JSON ----
        metadata_json = consistency_dir / "consistency_metadata.json"
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Wrote metadata summary: {metadata_json}")

        logger.info("=" * 60)
        logger.info("Evidence consistency analysis complete!")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("DRY RUN - Summary of what would be written:")
        logger.info(f"  {consistency_dir / 'concordance_by_era.csv'}")
        logger.info(f"    {len(era_stats)} rows (eras)")
        logger.info(f"  {consistency_dir / 'concordance_by_year.csv'}")
        logger.info(f"    {len(yearly_stats)} rows (years)")
        logger.info(f"  {consistency_dir / 'era_comparison_tests.csv'}")
        logger.info(f"    {len(test_results['pairwise_tests'])} rows")
        logger.info(f"  {consistency_dir / 'strobe_impact_analysis.csv'}")
        logger.info("    2 rows (pre/post STROBE)")
        logger.info(
            f"  {consistency_dir / 'concordance_by_match_type_era.csv'}"
        )
        logger.info(f"    {len(match_era_stats)} rows")
        logger.info(f"  {consistency_dir / 'consistency_metadata.json'}")
        logger.info("    Metadata summary with all results")
        logger.info("  Figures:")
        logger.info("    concordance_over_time.png/svg")
        logger.info("    strobe_impact.png/svg")
        logger.info("    concordance_by_match_type.png/svg")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
