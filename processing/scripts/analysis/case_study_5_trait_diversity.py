"""Analyze temporal trends in trait diversity for Case Study 5 (RQ1).

This script analyzes how trait diversity in MR studies has evolved over time,
testing the hypothesis that trait counts per study have increased 20-30%
from 2015 to 2024 as the field has matured.

Research Question 1:
Has trait diversity in MR studies increased over time?

Input:
    - data/db/trait_profile_db.db (query_combinations)
    - data/processed/case-study-cs5/temporal/temporal_metadata.csv
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs5/diversity/
        trait_counts_by_year.csv
        trait_counts_by_era.csv
        temporal_trend_model.csv
        era_comparison_tests.csv
        diversity_metadata.json
    - data/processed/case-study-cs5/figures/
        trait_diversity_over_time.png
        trait_diversity_over_time.svg
        trait_diversity_by_era.png
        trait_diversity_by_era.svg
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
        help="Path to trait_profile_db database (overrides config)",
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


def load_trait_profiles(
    db_path: Path, model_filter: str = "gpt-5"
) -> pd.DataFrame:
    """Load trait profile data from trait_profile_db.

    Args:
        db_path: Path to trait_profile_db.db
        model_filter: Model to filter results (default: gpt-5)

    Returns:
        DataFrame with pmid, model, trait_count
    """
    logger.info(f"Loading trait profiles from {db_path}...")

    con = duckdb.connect(str(db_path), read_only=True)

    query = """
    SELECT
        pmid,
        model,
        trait_count
    FROM query_combinations
    WHERE model = ?
    ORDER BY pmid
    """

    df = con.execute(query, [model_filter]).fetchdf()
    con.close()

    logger.info(f"Loaded {len(df)} trait profiles for model {model_filter}")

    # ---- Validate model filtering ----
    unique_models = df["model"].unique()
    if len(unique_models) != 1 or unique_models[0] != model_filter:
        logger.error(
            f"Model filtering failed: expected {model_filter}, "
            f"got {unique_models}"
        )
        sys.exit(1)

    logger.info(f"VALIDATION: All {len(df)} profiles are model={model_filter}")

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
    logger.info(f"Loaded temporal metadata for {len(df)} studies")

    res = df
    return res


def merge_trait_temporal_data(
    trait_df: pd.DataFrame, temporal_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge trait profiles with temporal metadata.

    Args:
        trait_df: DataFrame with trait profile data
        temporal_df: DataFrame with temporal metadata

    Returns:
        Merged DataFrame
    """
    logger.info("Merging trait profiles with temporal metadata...")

    # ---- Convert pmid to string for consistent merge ----
    trait_df = trait_df.copy()
    temporal_df = temporal_df.copy()
    trait_df["pmid"] = trait_df["pmid"].astype(str)
    temporal_df["pmid"] = temporal_df["pmid"].astype(str)

    # ---- Merge on pmid (both are gpt-5 only) ----
    merged = trait_df.merge(
        temporal_df[["pmid", "pub_year", "era", "years_since_inception"]],
        on="pmid",
        how="inner",
    )

    # ---- Filter out rows with missing years ----
    initial_count = len(merged)
    merged = merged[merged["pub_year"].notna()].copy()
    final_count = len(merged)

    if initial_count > final_count:
        logger.warning(
            f"Dropped {initial_count - final_count} studies with missing "
            f"publication years"
        )

    logger.info(f"Merged dataset: {len(merged)} studies with valid years")

    res = merged
    return res


def compute_trait_counts_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trait count statistics by publication year.

    Args:
        df: DataFrame with trait_count and pub_year columns

    Returns:
        DataFrame with yearly statistics
    """
    logger.info("Computing trait count statistics by year...")

    yearly_stats = (
        df.groupby("pub_year")["trait_count"]
        .agg(
            [
                ("n_studies", "count"),
                ("mean_trait_count", "mean"),
                ("median_trait_count", "median"),
                ("std_trait_count", "std"),
                ("min_trait_count", "min"),
                ("q25_trait_count", lambda x: x.quantile(0.25)),
                ("q75_trait_count", lambda x: x.quantile(0.75)),
                ("max_trait_count", "max"),
            ]
        )
        .reset_index()
    )

    # ---- Convert pub_year to int ----
    yearly_stats["pub_year"] = yearly_stats["pub_year"].astype(int)

    logger.info(f"Computed statistics for {len(yearly_stats)} years")

    res = yearly_stats
    return res


def compute_trait_counts_by_era(
    df: pd.DataFrame, era_defs: Dict[str, list]
) -> pd.DataFrame:
    """Compute trait count statistics by methodological era.

    Args:
        df: DataFrame with trait_count and era columns
        era_defs: Dictionary of era definitions

    Returns:
        DataFrame with era statistics
    """
    logger.info("Computing trait count statistics by era...")

    # ---- Filter out unknown era ----
    df_known_eras = df[df["era"] != "unknown"].copy()

    era_stats = (
        df_known_eras.groupby("era")["trait_count"]
        .agg(
            [
                ("n_studies", "count"),
                ("mean_trait_count", "mean"),
                ("median_trait_count", "median"),
                ("std_trait_count", "std"),
                ("min_trait_count", "min"),
                ("q25_trait_count", lambda x: x.quantile(0.25)),
                ("q75_trait_count", lambda x: x.quantile(0.75)),
                ("max_trait_count", "max"),
            ]
        )
        .reset_index()
    )

    # ---- Add era start/end years from config ----
    era_stats["era_start"] = era_stats["era"].map(
        lambda x: era_defs.get(x, [None, None])[0]
    )
    era_stats["era_end"] = era_stats["era"].map(
        lambda x: era_defs.get(x, [None, None])[1]
    )

    # ---- Sort by era start year ----
    era_stats = era_stats.sort_values("era_start")

    logger.info(f"Computed statistics for {len(era_stats)} eras")

    res = era_stats
    return res


def fit_temporal_trend_model(yearly_stats: pd.DataFrame) -> Dict[str, Any]:
    """Fit linear regression model to test temporal trend.

    Tests hypothesis: trait_count ~ publication_year

    Args:
        yearly_stats: DataFrame with pub_year and mean_trait_count

    Returns:
        Dictionary with model results
    """
    logger.info("Fitting temporal trend model...")

    # ---- Prepare data for regression ----
    x = yearly_stats["pub_year"].values
    y = yearly_stats["mean_trait_count"].values

    # ---- Fit weighted linear regression ----
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # ---- Compute additional statistics ----
    y_pred = intercept + slope * x
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # ---- Compute percent change from 2015 to 2024 ----
    year_2015_idx = np.where(x == 2015)[0]
    year_2024_idx = np.where(x == 2024)[0]

    if len(year_2015_idx) > 0 and len(year_2024_idx) > 0:
        y_2015 = y[year_2015_idx[0]]
        y_2024 = y[year_2024_idx[0]]
        percent_change = ((y_2024 - y_2015) / y_2015) * 100
    else:
        y_2015 = None
        y_2024 = None
        percent_change = None

    model_results = {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "r_squared": float(r_squared),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_years": int(len(x)),
        "year_range": [int(x.min()), int(x.max())],
        "mean_trait_count_2015": float(y_2015) if y_2015 is not None else None,
        "mean_trait_count_2024": float(y_2024) if y_2024 is not None else None,
        "percent_change_2015_2024": (
            float(percent_change) if percent_change is not None else None
        ),
    }

    logger.info("Temporal trend model fitted:")
    logger.info(f"  Slope: {slope:.4f} traits/year")
    logger.info(f"  R-squared: {r_squared:.4f}")
    logger.info(f"  P-value: {p_value:.4e}")

    if percent_change is not None:
        logger.info(
            f"  Change 2015-2024: {percent_change:.1f}% "
            f"({y_2015:.2f} -> {y_2024:.2f} traits)"
        )

    res = model_results
    return res


def perform_era_comparison_tests(
    df: pd.DataFrame, era_stats: pd.DataFrame
) -> Dict[str, Any]:
    """Perform statistical tests comparing trait diversity across eras.

    Tests:
    - ANOVA: Overall difference across eras
    - Post-hoc pairwise t-tests with Bonferroni correction

    Args:
        df: DataFrame with trait_count and era columns
        era_stats: DataFrame with era statistics

    Returns:
        Dictionary with test results
    """
    logger.info("Performing era comparison tests...")

    # ---- Filter to known eras ----
    df_known = df[df["era"] != "unknown"].copy()

    # ---- Prepare groups for ANOVA ----
    era_names = era_stats["era"].tolist()
    groups = [
        df_known[df_known["era"] == era]["trait_count"].values
        for era in era_names
    ]

    # ---- Perform ANOVA ----
    f_stat, anova_p = stats.f_oneway(*groups)

    logger.info("ANOVA results:")
    logger.info(f"  F-statistic: {f_stat:.4f}")
    logger.info(f"  P-value: {anova_p:.4e}")

    # ---- Perform pairwise t-tests ----
    pairwise_tests = []
    n_comparisons = 0

    for i, era1 in enumerate(era_names):
        for era2 in era_names[i + 1 :]:
            group1 = df_known[df_known["era"] == era1]["trait_count"].values
            group2 = df_known[df_known["era"] == era2]["trait_count"].values

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

    # ---- Apply Bonferroni correction ----
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
        "pairwise_tests": pairwise_tests,
        "bonferroni_alpha": float(bonferroni_alpha),
        "n_comparisons": n_comparisons,
    }

    res = test_results
    return res


def plot_diversity_over_time(
    yearly_stats: pd.DataFrame,
    model_results: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate visualization of trait diversity over time.

    Args:
        yearly_stats: DataFrame with yearly statistics
        model_results: Dictionary with temporal trend model results
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating trait diversity over time plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate diversity over time plot")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    # ---- Plot mean trait count with error bars ----
    x = yearly_stats["pub_year"]
    y = yearly_stats["mean_trait_count"]
    yerr = yearly_stats["std_trait_count"] / np.sqrt(yearly_stats["n_studies"])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        capsize=3,
        label="Mean trait count (± SE)",
        alpha=0.7,
    )

    # ---- Plot regression line ----
    x_line = np.array([x.min(), x.max()])
    y_line = model_results["intercept"] + model_results["slope"] * x_line
    ax.plot(
        x_line,
        y_line,
        "r--",
        label=f"Linear trend (R² = {model_results['r_squared']:.3f})",
        linewidth=2,
    )

    # ---- Formatting ----
    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Mean Trait Count per Study", fontsize=12)
    ax.set_title(
        "Temporal Trends in Trait Diversity (GPT-5 Model)", fontsize=14
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"trait_diversity_over_time.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def plot_diversity_by_era(
    era_stats: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Generate visualization of trait diversity by methodological era.

    Args:
        era_stats: DataFrame with era statistics
        output_dir: Output directory for figures
        config: Configuration dictionary
        dry_run: If True, show what would be done without executing
    """
    logger.info("Generating trait diversity by era plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate diversity by era plot")
        return

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(figsize=tuple(fig_config["figsize"]["single"]))

    # ---- Create bar plot with error bars ----
    x = np.arange(len(era_stats))
    y = era_stats["mean_trait_count"]
    yerr = era_stats["std_trait_count"]

    bars = ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7, color="steelblue")

    # ---- Formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels(era_stats["era"], rotation=45, ha="right")
    ax.set_xlabel("Methodological Era", fontsize=12)
    ax.set_ylabel("Mean Trait Count per Study", fontsize=12)
    ax.set_title(
        "Trait Diversity by Methodological Era (GPT-5 Model)", fontsize=14
    )
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Add sample size annotations ----
    for i, (bar, n_studies) in enumerate(zip(bars, era_stats["n_studies"])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + yerr.iloc[i] + 0.5,
            f"n={n_studies}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # ---- Save figures ----
    for fmt in fig_config["format"]:
        output_path = output_dir / f"trait_diversity_by_era.{fmt}"
        fig.savefig(output_path, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close(fig)


def create_diversity_metadata(
    yearly_stats: pd.DataFrame,
    era_stats: pd.DataFrame,
    model_results: Dict[str, Any],
    test_results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create JSON metadata summary for diversity analysis.

    Args:
        yearly_stats: DataFrame with yearly statistics
        era_stats: DataFrame with era statistics
        model_results: Dictionary with temporal trend model results
        test_results: Dictionary with era comparison test results
        config: Configuration dictionary

    Returns:
        Dictionary with metadata summary
    """
    logger.info("Creating diversity analysis metadata...")

    cs5_config = config["case_study_5"]

    metadata = {
        "research_question": (
            "Has trait diversity in MR studies increased over time?"
        ),
        "hypothesis": (
            "20-30% increase in trait counts per study from 2015 to 2024"
        ),
        "model": cs5_config["models_included"][0],
        "total_studies": int(yearly_stats["n_studies"].sum()),
        "year_range": {
            "min": int(yearly_stats["pub_year"].min()),
            "max": int(yearly_stats["pub_year"].max()),
            "n_years": len(yearly_stats),
        },
        "temporal_trend_model": model_results,
        "era_comparison_tests": test_results,
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
    logger.info("Case Study 5: Trait diversity temporal trends (RQ1)")
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
        db_path = DATA_DIR / "db" / "trait_profile_db.db"

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
    diversity_dir = Path(output_config["diversity"])
    figures_dir = Path(output_config["figures"])

    if not args.dry_run:
        diversity_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directories")
    else:
        logger.info("Would create output directories")

    # ---- Load data ----
    model_filter = cs5_config["models_included"][0]
    trait_df = load_trait_profiles(db_path, model_filter=model_filter)
    temporal_df = load_temporal_metadata(temporal_path)

    # ---- Merge data ----
    df = merge_trait_temporal_data(trait_df, temporal_df)

    # ---- Filter by minimum studies per year ----
    min_studies = cs5_config["trait_diversity"]["min_studies_per_year"]
    yearly_counts = df["pub_year"].value_counts()
    valid_years = yearly_counts[yearly_counts >= min_studies].index
    df_filtered = df[df["pub_year"].isin(valid_years)].copy()

    logger.info(
        f"Filtered to {len(df_filtered)} studies "
        f"(years with >= {min_studies} studies)"
    )

    # ---- Compute statistics ----
    yearly_stats = compute_trait_counts_by_year(df_filtered)
    era_stats = compute_trait_counts_by_era(
        df_filtered, cs5_config["temporal_eras"]
    )

    # ---- Fit temporal trend model ----
    model_results = fit_temporal_trend_model(yearly_stats)

    # ---- Perform era comparison tests ----
    test_results = perform_era_comparison_tests(df_filtered, era_stats)

    # ---- Create metadata ----
    metadata = create_diversity_metadata(
        yearly_stats, era_stats, model_results, test_results, config
    )

    # ---- Generate plots ----
    plot_diversity_over_time(
        yearly_stats, model_results, figures_dir, config, dry_run=args.dry_run
    )
    plot_diversity_by_era(era_stats, figures_dir, config, dry_run=args.dry_run)

    # ---- Write outputs ----
    if not args.dry_run:
        # ---- Write yearly statistics ----
        yearly_csv = diversity_dir / "trait_counts_by_year.csv"
        yearly_stats.to_csv(yearly_csv, index=False)
        logger.info(f"Wrote yearly statistics: {yearly_csv}")

        # ---- Write era statistics ----
        era_csv = diversity_dir / "trait_counts_by_era.csv"
        era_stats.to_csv(era_csv, index=False)
        logger.info(f"Wrote era statistics: {era_csv}")

        # ---- Write temporal trend model ----
        model_df = pd.DataFrame([model_results])
        model_csv = diversity_dir / "temporal_trend_model.csv"
        model_df.to_csv(model_csv, index=False)
        logger.info(f"Wrote temporal trend model: {model_csv}")

        # ---- Write era comparison tests ----
        if test_results["pairwise_tests"]:
            tests_df = pd.DataFrame(test_results["pairwise_tests"])
            tests_csv = diversity_dir / "era_comparison_tests.csv"
            tests_df.to_csv(tests_csv, index=False)
            logger.info(f"Wrote era comparison tests: {tests_csv}")

        # ---- Write metadata JSON ----
        metadata_json = diversity_dir / "diversity_metadata.json"
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Wrote metadata summary: {metadata_json}")

        logger.info("=" * 60)
        logger.info("Trait diversity analysis complete!")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("DRY RUN - Summary of what would be written:")
        logger.info(f"  {diversity_dir / 'trait_counts_by_year.csv'}")
        logger.info(f"    {len(yearly_stats)} rows (years)")
        logger.info(f"  {diversity_dir / 'trait_counts_by_era.csv'}")
        logger.info(f"    {len(era_stats)} rows (eras)")
        logger.info(f"  {diversity_dir / 'temporal_trend_model.csv'}")
        logger.info("    1 row (model results)")
        logger.info(f"  {diversity_dir / 'era_comparison_tests.csv'}")
        logger.info(f"    {len(test_results['pairwise_tests'])} rows")
        logger.info(f"  {diversity_dir / 'diversity_metadata.json'}")
        logger.info("    Metadata summary with all results")
        logger.info("  Figures:")
        logger.info("    trait_diversity_over_time.png/svg")
        logger.info("    trait_diversity_by_era.png/svg")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
