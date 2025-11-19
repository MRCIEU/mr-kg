"""Compute reproducibility metrics for Case Study 1.

This script processes multi-study trait pairs to compute pair-level
reproducibility metrics including direction concordance statistics
and reproducibility tier assignments. Results are stratified by
study frequency, temporal era, match type, and outcome category.

Outputs:
- pair_reproducibility_metrics.csv: Pair-level metrics with categories
- tier_distribution.csv: Distribution across reproducibility tiers
- stratified_by_study_count.csv: Metrics by study count bands
- stratified_by_temporal_era.csv: Metrics by time period
- stratified_by_match_type.csv: Metrics by match type
- stratified_by_category.csv: Metrics by outcome category
- stratified_by_category_era.csv: Category x era interaction
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

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
        help="Perform dry run without executing analysis",
    )

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration file (default: {DEFAULT_CONFIG})",
    )

    # ---- --input-csv ----
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Override input CSV path from config",
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


# ==== Metric computation functions ====


def assign_reproducibility_tier(
    mean_concordance: float,
    tiers: Dict[str, float],
) -> str:
    """Assign reproducibility tier based on mean direction concordance.

    Args:
        mean_concordance: Mean direction concordance value
        tiers: Dictionary of tier thresholds from config

    Returns:
        Tier label: high, moderate, low, or discordant
    """
    if mean_concordance >= tiers["high"]:
        return "high"
    elif mean_concordance >= tiers["moderate"]:
        return "moderate"
    elif mean_concordance >= tiers["low"]:
        return "low"
    else:
        return "discordant"


def assign_study_count_band(
    count: int,
    bands: List[List[int]],
) -> str:
    """Assign study count band label.

    Args:
        count: Number of studies
        bands: List of [min, max] study count ranges

    Returns:
        Band label like "2-3" or "11+"

    Raises:
        ValueError: If count falls outside all defined bands
    """
    for band in bands:
        if band[0] <= count <= band[1]:
            if band[1] >= 999:
                return f"{band[0]}+"
            return f"{band[0]}-{band[1]}"

    raise ValueError(
        f"Study count {count} falls outside all defined bands: {bands}"
    )


def assign_temporal_era(
    year: float,
    eras: Dict[str, List[int]],
) -> str:
    """Assign temporal era based on publication year.

    Args:
        year: Publication year
        eras: Dictionary of era definitions from config

    Returns:
        Era label (early_mr, mr_egger, mr_presso, within_family, strobe_mr, or other)
    """
    if pd.isna(year):
        return "unknown"

    year_int = int(year)

    for era_name, era_range in eras.items():
        if era_range[0] <= year_int <= era_range[1]:
            return era_name

    return "other"


def assign_outcome_category(
    trait_pairs_json: str,
    category_mapping: Dict[str, List[str]],
) -> str:
    """Assign outcome category based on trait pairs in study.

    Uses outcome-only categorization: extracts all outcome traits
    from trait pairs and assigns category based on keyword matching.

    Args:
        trait_pairs_json: JSON string of trait pairs
        category_mapping: Dictionary mapping categories to keywords

    Returns:
        Category label or "uncategorized"
    """
    try:
        import json as json_module

        if isinstance(trait_pairs_json, str):
            pairs = json_module.loads(trait_pairs_json)
        else:
            pairs = trait_pairs_json

        outcomes = [
            pair.get("outcome", "").lower()
            for pair in pairs
            if "outcome" in pair
        ]

        if not outcomes:
            return "uncategorized"

        for category, keywords in category_mapping.items():
            for outcome in outcomes:
                for keyword in keywords:
                    if keyword.lower() in outcome:
                        return category

        return "uncategorized"

    except Exception as e:
        logger.warning(f"Error parsing trait pairs: {e}")
        return "uncategorized"


def compute_pair_metrics(
    pairs_df: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    """Compute pair-level reproducibility metrics.

    Args:
        pairs_df: DataFrame from extract_pairs script
        config: Configuration dictionary

    Returns:
        DataFrame with added metrics columns:
        - reproducibility_tier
        - study_count_band
        - temporal_era
        - concordance_variance
        - outcome_category (if category analysis enabled)
    """
    tiers = config["case_study_1"]["reproducibility_tiers"]
    bands = config["case_study_1"]["study_count_bands"]
    eras = config["case_study_1"]["temporal_eras"]

    logger.info("Computing pair-level metrics...")

    result_df = pairs_df.copy()

    result_df["reproducibility_tier"] = result_df[
        "mean_direction_concordance"
    ].apply(lambda x: assign_reproducibility_tier(x, tiers))

    result_df["study_count_band"] = result_df["study_count"].apply(
        lambda x: assign_study_count_band(x, bands)
    )

    result_df["temporal_era"] = result_df["publication_year"].apply(
        lambda x: assign_temporal_era(x, eras)
    )

    result_df["concordance_variance"] = (
        result_df["std_direction_concordance"] ** 2
    )

    category_config = config["case_study_1"].get("category_analysis", {})
    if category_config.get("enabled", False):
        logger.info("Computing outcome categories...")
        category_mapping = category_config["category_mapping"]
        result_df["outcome_category"] = result_df["trait_pairs_json"].apply(
            lambda x: assign_outcome_category(x, category_mapping)
        )

        n_categorized = (
            result_df["outcome_category"] != "uncategorized"
        ).sum()
        coverage = n_categorized / len(result_df)
        logger.info(
            f"Category coverage: {n_categorized}/{len(result_df)} "
            f"({coverage:.1%})"
        )

        min_threshold = category_config.get("min_coverage_threshold", 0.40)
        if coverage < min_threshold:
            logger.warning(
                f"Category coverage ({coverage:.1%}) below minimum "
                f"threshold ({min_threshold:.0%})"
            )

    logger.info(f"Computed metrics for {len(result_df)} pairs")

    return result_df


def compute_tier_distribution(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution of pairs across reproducibility tiers.

    Args:
        metrics_df: DataFrame with reproducibility tier assignments

    Returns:
        DataFrame with tier counts and percentages
    """
    logger.info("Computing tier distribution...")

    tier_order = ["high", "moderate", "low", "discordant"]

    counts = (
        metrics_df["reproducibility_tier"]
        .value_counts()
        .reindex(tier_order, fill_value=0)
    )

    total = len(metrics_df)

    res = pd.DataFrame(
        {
            "tier": tier_order,
            "count": [counts[tier] for tier in tier_order],
            "percentage": [100 * counts[tier] / total for tier in tier_order],
        }
    )

    logger.info(f"Computed distribution for {total} pairs")

    return res


def compute_stratified_metrics(
    metrics_df: pd.DataFrame,
    stratify_by: str,
) -> pd.DataFrame:
    """Compute reproducibility metrics stratified by a grouping variable.

    Args:
        metrics_df: DataFrame with reproducibility metrics
        stratify_by: Column name to stratify by

    Returns:
        DataFrame with aggregated metrics by group
    """
    logger.info(f"Computing metrics stratified by {stratify_by}...")

    res = (
        metrics_df.groupby(stratify_by)
        .agg(
            n_pairs=("study1_pmid", "count"),
            mean_concordance=("mean_direction_concordance", "mean"),
            median_concordance=("median_direction_concordance", "median"),
            std_concordance=("std_direction_concordance", "mean"),
            mean_variance=("concordance_variance", "mean"),
            n_high=(
                "reproducibility_tier",
                lambda x: (x == "high").sum(),
            ),
            n_moderate=(
                "reproducibility_tier",
                lambda x: (x == "moderate").sum(),
            ),
            n_low=(
                "reproducibility_tier",
                lambda x: (x == "low").sum(),
            ),
            n_discordant=(
                "reproducibility_tier",
                lambda x: (x == "discordant").sum(),
            ),
        )
        .reset_index()
    )

    res["pct_high"] = 100 * res["n_high"] / res["n_pairs"]
    res["pct_moderate"] = 100 * res["n_moderate"] / res["n_pairs"]
    res["pct_low"] = 100 * res["n_low"] / res["n_pairs"]
    res["pct_discordant"] = 100 * res["n_discordant"] / res["n_pairs"]

    logger.info(f"Computed stratified metrics for {len(res)} groups")

    return res


# ==== Main execution ====


def main():
    """Execute reproducibility metrics computation."""
    args = make_args()

    # ---- Load configuration ----

    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    # ---- Resolve paths ----

    if args.input_csv:
        input_csv = args.input_csv
    else:
        raw_pairs_dir = PROJECT_ROOT / output_config["raw_pairs"]
        input_csv = raw_pairs_dir / "multi_study_pairs.csv"

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = PROJECT_ROOT / output_config["metrics"]

    # ---- Validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating configuration and paths")
        logger.info(f"Input CSV: {input_csv}")
        logger.info(f"Output directory: {output_dir}")

        if not input_csv.exists():
            logger.error(f"Input CSV not found: {input_csv}")
            return 1

        logger.info("Dry run complete - configuration validated")
        return 0

    # ---- Setup ----

    if not input_csv.exists():
        logger.error(f"Input CSV not found: {input_csv}")
        logger.error(
            "Please run case_study_1_extract_pairs.py first to generate input"
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ---- Load input data ----

    logger.info(f"Loading input data from: {input_csv}")
    pairs_df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(pairs_df)} study pairs")

    # ---- Compute pair-level metrics ----

    metrics_df = compute_pair_metrics(pairs_df, config)

    output_file = output_dir / "pair_reproducibility_metrics.csv"
    metrics_df.to_csv(output_file, index=False)
    logger.info(f"Saved pair metrics: {output_file}")

    # ---- Compute tier distribution ----

    tier_dist = compute_tier_distribution(metrics_df)

    output_file = output_dir / "tier_distribution.csv"
    tier_dist.to_csv(output_file, index=False)
    logger.info(f"Saved tier distribution: {output_file}")

    # ---- Stratify by study count ----

    study_count_strat = compute_stratified_metrics(
        metrics_df, "study_count_band"
    )

    output_file = output_dir / "stratified_by_study_count.csv"
    study_count_strat.to_csv(output_file, index=False)
    logger.info(f"Saved study count stratification: {output_file}")

    # ---- Stratify by temporal era ----

    temporal_strat = compute_stratified_metrics(metrics_df, "temporal_era")

    output_file = output_dir / "stratified_by_temporal_era.csv"
    temporal_strat.to_csv(output_file, index=False)
    logger.info(f"Saved temporal era stratification: {output_file}")

    # ---- Stratify by match type ----

    match_type_metrics = []
    for match_col, match_label in [
        ("has_exact_match", "exact"),
        ("has_fuzzy_match", "fuzzy"),
        ("has_efo_match", "efo"),
    ]:
        subset = metrics_df[metrics_df[match_col]]
        if len(subset) > 0:
            stats = {
                "match_type": match_label,
                "n_pairs": len(subset),
                "mean_concordance": subset[
                    "mean_direction_concordance"
                ].mean(),
                "median_concordance": subset[
                    "median_direction_concordance"
                ].median(),
                "std_concordance": subset["std_direction_concordance"].mean(),
                "mean_variance": subset["concordance_variance"].mean(),
                "n_high": (subset["reproducibility_tier"] == "high").sum(),
                "n_moderate": (
                    subset["reproducibility_tier"] == "moderate"
                ).sum(),
                "n_low": (subset["reproducibility_tier"] == "low").sum(),
                "n_discordant": (
                    subset["reproducibility_tier"] == "discordant"
                ).sum(),
            }
            stats["pct_high"] = 100 * stats["n_high"] / stats["n_pairs"]
            stats["pct_moderate"] = (
                100 * stats["n_moderate"] / stats["n_pairs"]
            )
            stats["pct_low"] = 100 * stats["n_low"] / stats["n_pairs"]
            stats["pct_discordant"] = (
                100 * stats["n_discordant"] / stats["n_pairs"]
            )
            match_type_metrics.append(stats)

    match_type_strat = pd.DataFrame(match_type_metrics)

    output_file = output_dir / "stratified_by_match_type.csv"
    match_type_strat.to_csv(output_file, index=False)
    logger.info(f"Saved match type stratification: {output_file}")

    # ---- Stratify by outcome category ----

    category_config = config["case_study_1"].get("category_analysis", {})
    if (
        category_config.get("enabled", False)
        and "outcome_category" in metrics_df.columns
    ):
        logger.info("Computing category stratifications...")

        category_strat = compute_stratified_metrics(
            metrics_df[metrics_df["outcome_category"] != "uncategorized"],
            "outcome_category",
        )

        output_file = output_dir / "stratified_by_category.csv"
        category_strat.to_csv(output_file, index=False)
        logger.info(f"Saved category stratification: {output_file}")

        category_era_df = metrics_df[
            (metrics_df["outcome_category"] != "uncategorized")
            & (metrics_df["temporal_era"].isin(["early", "recent"]))
        ]

        if len(category_era_df) > 0:
            category_era_strat = (
                category_era_df.groupby(["outcome_category", "temporal_era"])
                .agg(
                    n_pairs=("study1_pmid", "count"),
                    mean_concordance=("mean_direction_concordance", "mean"),
                    median_concordance=(
                        "median_direction_concordance",
                        "median",
                    ),
                    n_high=(
                        "reproducibility_tier",
                        lambda x: (x == "high").sum(),
                    ),
                )
                .reset_index()
            )

            category_era_strat["pct_high"] = (
                100
                * category_era_strat["n_high"]
                / category_era_strat["n_pairs"]
            )

            output_file = output_dir / "stratified_by_category_era.csv"
            category_era_strat.to_csv(output_file, index=False)
            logger.info(f"Saved category×era stratification: {output_file}")

    # ---- Print summary ----

    logger.info("\n" + "=" * 60)
    logger.info("REPRODUCIBILITY METRICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pairs analyzed: {len(metrics_df)}")
    logger.info("\nTier distribution:")
    for _, row in tier_dist.iterrows():
        logger.info(
            f"  {row['tier']:12s}: {row['count']:4d} ({row['percentage']:5.1f}%)"
        )

    logger.info(
        f"\nOverall mean concordance: "
        f"{metrics_df['mean_direction_concordance'].mean():.3f}"
    )
    logger.info(
        f"Overall median concordance: "
        f"{metrics_df['median_direction_concordance'].median():.3f}"
    )
    logger.info(
        f"Mean variance: {metrics_df['concordance_variance'].mean():.3f}"
    )

    logger.info("\nStudy count stratification:")
    for _, row in study_count_strat.iterrows():
        logger.info(
            f"  {row['study_count_band']:8s}: "
            f"{row['n_pairs']:4d} pairs, "
            f"mean concordance = {row['mean_concordance']:.3f}"
        )

    logger.info("=" * 60)

    logger.info("\nMetrics computation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
