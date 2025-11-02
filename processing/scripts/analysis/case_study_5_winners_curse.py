"""Analyze winner's curse phenomenon for Case Study 5 (RQ6).

This script implements a two-stage analysis of winner's curse in MR studies:
Stage 1 assesses effect size availability to determine feasibility;
Stage 2 (conditional) quantifies temporal decline in effect sizes.

Research Question 6:
Do early studies report systematically larger effect sizes than later studies?

Hypothesis:
First quartile studies report 20-40% larger effects than last quartile.

Stage 1 (Mandatory):
    - Assess overall effect size availability in model_results
    - Calculate subgroup-specific availability rates
    - Apply feasibility threshold (>=10% availability required)
    - Output feasibility report

Stage 2 (Conditional on Stage 1):
    - Extract and normalize effect sizes for trait pairs with >=5 studies
    - Test temporal decline via linear regression
    - Compare first vs last quartile effect sizes
    - Identify dramatic examples of effect size shrinkage

Input:
    - data/db/vector_store.db (model_results table)
    - data/processed/case-study-cs5/temporal/temporal_metadata.csv
    - config/case_studies.yml

Output (Stage 1 - Always):
    - data/processed/case-study-cs5/winners_curse/
        effect_size_availability.csv
        availability_by_subgroup.csv
        winners_curse_feasibility.md
        winners_curse_metadata.json

Output (Stage 2 - If feasible):
    - data/processed/case-study-cs5/winners_curse/
        trait_pair_temporal_effects.csv
        winners_curse_summary.csv
        example_pairs.csv
    - data/processed/case-study-cs5/figures/
        effect_size_temporal_trends.png/svg
        first_vs_last_quartile.png/svg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    # ---- --skip-stage2 ----
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 analysis even if feasible",
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


def extract_effect_sizes(results_df: pd.DataFrame) -> pd.DataFrame:
    """Extract effect sizes from parsed results JSON.

    Args:
        results_df: DataFrame with parsed results

    Returns:
        DataFrame with effect size indicators
    """
    logger.info("Extracting effect sizes from results JSON...")

    effect_size_data = []

    for _, row in results_df.iterrows():
        pmid = row["pmid"]
        results_parsed = row["results_parsed"]

        if not isinstance(results_parsed, list):
            results_parsed = [results_parsed]

        for result in results_parsed:
            if not isinstance(result, dict):
                continue

            # ---- Extract effect size fields ----
            beta = result.get("beta")
            odds_ratio = result.get("odds ratio")
            hazard_ratio = result.get("hazard ratio")

            # ---- Extract additional metadata ----
            exposure = result.get("exposure", "")
            outcome = result.get("outcome", "")

            effect_size_data.append(
                {
                    "pmid": pmid,
                    "exposure": exposure,
                    "outcome": outcome,
                    "beta": beta,
                    "odds_ratio": odds_ratio,
                    "hazard_ratio": hazard_ratio,
                    "has_beta": beta is not None
                    and beta != ""
                    and beta != "N/A",
                    "has_or": odds_ratio is not None
                    and odds_ratio != ""
                    and odds_ratio != "N/A",
                    "has_hr": hazard_ratio is not None
                    and hazard_ratio != ""
                    and hazard_ratio != "N/A",
                }
            )

    df = pd.DataFrame(effect_size_data)
    df["has_any_effect_size"] = df["has_beta"] | df["has_or"] | df["has_hr"]

    logger.info(f"Extracted effect sizes for {len(df)} result records")

    res = df
    return res


def assess_overall_availability(effect_sizes_df: pd.DataFrame) -> pd.DataFrame:
    """Assess overall effect size availability.

    Args:
        effect_sizes_df: DataFrame with effect size indicators

    Returns:
        DataFrame with availability statistics
    """
    logger.info("Assessing overall effect size availability...")

    n_total = len(effect_sizes_df)
    n_with_beta = effect_sizes_df["has_beta"].sum()
    n_with_or = effect_sizes_df["has_or"].sum()
    n_with_hr = effect_sizes_df["has_hr"].sum()
    n_with_any = effect_sizes_df["has_any_effect_size"].sum()

    availability_data = [
        {
            "field_name": "beta",
            "n_total_studies": n_total,
            "n_with_effect_size": n_with_beta,
            "percent_available": (
                100 * n_with_beta / n_total if n_total > 0 else 0.0
            ),
        },
        {
            "field_name": "odds_ratio",
            "n_total_studies": n_total,
            "n_with_effect_size": n_with_or,
            "percent_available": (
                100 * n_with_or / n_total if n_total > 0 else 0.0
            ),
        },
        {
            "field_name": "hazard_ratio",
            "n_total_studies": n_total,
            "n_with_effect_size": n_with_hr,
            "percent_available": (
                100 * n_with_hr / n_total if n_total > 0 else 0.0
            ),
        },
        {
            "field_name": "any_effect_size",
            "n_total_studies": n_total,
            "n_with_effect_size": n_with_any,
            "percent_available": (
                100 * n_with_any / n_total if n_total > 0 else 0.0
            ),
        },
    ]

    df = pd.DataFrame(availability_data)

    logger.info(
        "Overall availability: "
        f"{df[df['field_name'] == 'any_effect_size']['percent_available'].values[0]:.2f}%"
    )

    res = df
    return res


def assess_subgroup_availability(
    effect_sizes_df: pd.DataFrame, temporal_df: pd.DataFrame
) -> pd.DataFrame:
    """Assess effect size availability by subgroups.

    Args:
        effect_sizes_df: DataFrame with effect size indicators
        temporal_df: DataFrame with temporal metadata

    Returns:
        DataFrame with subgroup-specific availability
    """
    logger.info("Assessing subgroup-specific availability...")

    # ---- Merge with temporal metadata ----
    merged_df = effect_sizes_df.merge(temporal_df, on="pmid", how="left")

    subgroup_data = []

    # ---- Availability by era ----
    if "era" in merged_df.columns:
        for era in merged_df["era"].dropna().unique():
            era_df = merged_df[merged_df["era"] == era]
            n_total = len(era_df)
            n_with_any = era_df["has_any_effect_size"].sum()
            percent = 100 * n_with_any / n_total if n_total > 0 else 0.0

            subgroup_data.append(
                {
                    "subgroup_type": "era",
                    "subgroup_name": era,
                    "n_studies": n_total,
                    "n_with_effect_size": n_with_any,
                    "percent_available": percent,
                    "analysis_feasible_flag": percent >= 10.0,
                }
            )

    # ---- Availability by year bins (early vs recent) ----
    if "pub_year" in merged_df.columns:
        year_bins = [
            ("2010-2014", 2010, 2014),
            ("2015-2017", 2015, 2017),
            ("2018-2019", 2018, 2019),
            ("2020", 2020, 2020),
            ("2021-2024", 2021, 2024),
        ]

        for bin_name, year_start, year_end in year_bins:
            bin_df = merged_df[
                (merged_df["pub_year"] >= year_start)
                & (merged_df["pub_year"] <= year_end)
            ]
            n_total = len(bin_df)
            n_with_any = bin_df["has_any_effect_size"].sum()
            percent = 100 * n_with_any / n_total if n_total > 0 else 0.0

            subgroup_data.append(
                {
                    "subgroup_type": "year_bin",
                    "subgroup_name": bin_name,
                    "n_studies": n_total,
                    "n_with_effect_size": n_with_any,
                    "percent_available": percent,
                    "analysis_feasible_flag": percent >= 10.0,
                }
            )

    df = pd.DataFrame(subgroup_data)

    logger.info(
        f"Assessed availability for {len(df)} subgroups "
        f"({df['analysis_feasible_flag'].sum()} feasible)"
    )

    res = df
    return res


def generate_feasibility_report(
    overall_availability: pd.DataFrame,
    subgroup_availability: pd.DataFrame,
    config: Dict[str, Any],
) -> str:
    """Generate feasibility report for winner's curse analysis.

    Args:
        overall_availability: DataFrame with overall availability stats
        subgroup_availability: DataFrame with subgroup availability
        config: Configuration dictionary

    Returns:
        Markdown-formatted feasibility report
    """
    logger.info("Generating feasibility report...")

    min_threshold = config["case_study_5"]["winners_curse"][
        "min_effect_size_availability"
    ]
    min_threshold_pct = min_threshold * 100

    # ---- Overall availability ----
    overall_pct = overall_availability[
        overall_availability["field_name"] == "any_effect_size"
    ]["percent_available"].values[0]

    # ---- Feasible subgroups ----
    feasible_subgroups = subgroup_availability[
        subgroup_availability["analysis_feasible_flag"]
    ]

    # ---- Build report ----
    report_lines = [
        "# Winner's Curse Analysis - Feasibility Report",
        "",
        "## Stage 1: Effect Size Availability Assessment",
        "",
        "### Overall Availability",
        "",
        f"- **Total result records**: "
        f"{overall_availability['n_total_studies'].values[0]:,}",
        f"- **Records with any effect size**: "
        f"{overall_availability[overall_availability['field_name'] == 'any_effect_size']['n_with_effect_size'].values[0]:,} "
        f"({overall_pct:.2f}%)",
        "",
        "### Effect Size Field Breakdown",
        "",
    ]

    for _, row in overall_availability[
        overall_availability["field_name"] != "any_effect_size"
    ].iterrows():
        report_lines.append(
            f"- **{row['field_name']}**: "
            f"{row['n_with_effect_size']:,} / {row['n_total_studies']:,} "
            f"({row['percent_available']:.2f}%)"
        )

    report_lines.extend(
        [
            "",
            "### Subgroup-Specific Availability",
            "",
            f"**Feasibility Threshold**: >={min_threshold_pct:.0f}% availability",
            "",
        ]
    )

    if len(feasible_subgroups) > 0:
        report_lines.append(
            f"**Feasible Subgroups**: "
            f"{len(feasible_subgroups)} / "
            f"{len(subgroup_availability)}"
        )
        report_lines.append("")

        for subgroup_type in feasible_subgroups["subgroup_type"].unique():
            report_lines.append(
                f"#### {subgroup_type.replace('_', ' ').title()}"
            )
            report_lines.append("")

            type_subgroups = feasible_subgroups[
                feasible_subgroups["subgroup_type"] == subgroup_type
            ].sort_values("percent_available", ascending=False)

            for _, row in type_subgroups.iterrows():
                report_lines.append(
                    f"- **{row['subgroup_name']}**: "
                    f"{row['n_with_effect_size']:,} / {row['n_studies']:,} "
                    f"({row['percent_available']:.2f}%) ✓"
                )

            report_lines.append("")
    else:
        report_lines.extend(
            [
                "**No feasible subgroups identified**",
                "",
                "All subgroups fall below the minimum feasibility threshold.",
                "",
            ]
        )

    # ---- Stage 2 recommendation ----
    report_lines.extend(
        [
            "## Stage 2: Analysis Feasibility Decision",
            "",
        ]
    )

    if len(feasible_subgroups) > 0:
        report_lines.extend(
            [
                "**Decision**: PROCEED to Stage 2 winner's curse analysis",
                "",
                "Sufficient effect size data available in feasible subgroups to test temporal decline hypothesis.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "**Decision**: DO NOT PROCEED to Stage 2 analysis",
                "",
                f"Effect size availability ({overall_pct:.2f}%) is below the "
                f"feasibility threshold ({min_threshold_pct:.0f}%). No subgroups "
                "meet the minimum data requirement for winner's curse analysis.",
                "",
                "### Recommendations",
                "",
                "1. Improve LLM extraction prompts for effect size fields",
                "2. Focus on recent publications (2021-2024) with better reporting",
                "3. Target high-impact journals with stricter reporting standards",
                "4. Consider manual extraction for high-priority trait pairs",
                "",
            ]
        )

    res = "\n".join(report_lines)
    return res


def create_output_directories(config: Dict[str, Any]) -> Tuple[Path, Path]:
    """Create output directories for CS5 winners_curse analysis.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (winners_curse_dir, figures_dir)
    """
    base_path = (
        PROJECT_ROOT / config["output"]["case_study_5"]["winners_curse"]
    )
    figures_path = PROJECT_ROOT / config["output"]["case_study_5"]["figures"]

    base_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directories created: {base_path}, {figures_path}")

    res = (base_path, figures_path)
    return res


def save_stage1_outputs(
    overall_availability: pd.DataFrame,
    subgroup_availability: pd.DataFrame,
    feasibility_report: str,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save Stage 1 analysis outputs.

    Args:
        overall_availability: DataFrame with overall availability stats
        subgroup_availability: DataFrame with subgroup availability
        feasibility_report: Markdown feasibility report
        config: Configuration dictionary
        output_dir: Output directory path
    """
    logger.info("Saving Stage 1 outputs...")

    # ---- Save availability CSVs ----
    overall_availability.to_csv(
        output_dir / "effect_size_availability.csv", index=False
    )
    logger.info(f"Saved: {output_dir / 'effect_size_availability.csv'}")

    subgroup_availability.to_csv(
        output_dir / "availability_by_subgroup.csv", index=False
    )
    logger.info(f"Saved: {output_dir / 'availability_by_subgroup.csv'}")

    # ---- Save feasibility report ----
    with open(output_dir / "winners_curse_feasibility.md", "w") as f:
        f.write(feasibility_report)
    logger.info(f"Saved: {output_dir / 'winners_curse_feasibility.md'}")

    # ---- Save metadata ----
    metadata = {
        "script": "case_study_5_winners_curse.py",
        "stage": "stage1_availability_assessment",
        "model_filter": config["case_study_5"]["models_included"][0],
        "overall_availability_pct": float(
            overall_availability[
                overall_availability["field_name"] == "any_effect_size"
            ]["percent_available"].values[0]
        ),
        "n_feasible_subgroups": int(
            subgroup_availability["analysis_feasible_flag"].sum()
        ),
        "feasibility_threshold_pct": (
            config["case_study_5"]["winners_curse"][
                "min_effect_size_availability"
            ]
            * 100
        ),
        "outputs": [
            "effect_size_availability.csv",
            "availability_by_subgroup.csv",
            "winners_curse_feasibility.md",
        ],
    }

    metadata_converted = convert_to_native_types(metadata)

    with open(output_dir / "winners_curse_metadata.json", "w") as f:
        json.dump(metadata_converted, f, indent=2)
    logger.info(f"Saved: {output_dir / 'winners_curse_metadata.json'}")


# ==== Stage 2: Winner's Curse Analysis Functions ====


def parse_effect_size_value(value: Any) -> Optional[float]:
    """Parse effect size value from various formats.

    Args:
        value: Raw effect size value (may be string, float, or None)

    Returns:
        Parsed float value or None if invalid
    """
    if value is None or value == "" or value == "N/A":
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()
        if value in ["", "N/A", "NA", "null"]:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    return None


def normalize_effect_size(
    beta: Optional[float],
    odds_ratio: Optional[float],
    hazard_ratio: Optional[float],
) -> Optional[float]:
    """Normalize effect sizes to a common scale.

    Priority: beta > log(OR) > log(HR)
    All converted to log scale for OR and HR.

    Args:
        beta: Beta coefficient
        odds_ratio: Odds ratio
        hazard_ratio: Hazard ratio

    Returns:
        Normalized effect size on log scale or None
    """
    beta_parsed = parse_effect_size_value(beta)
    or_parsed = parse_effect_size_value(odds_ratio)
    hr_parsed = parse_effect_size_value(hazard_ratio)

    # ---- Priority: beta ----
    if beta_parsed is not None:
        return beta_parsed

    # ---- Priority: log(OR) ----
    if or_parsed is not None and or_parsed > 0:
        return np.log(or_parsed)

    # ---- Priority: log(HR) ----
    if hr_parsed is not None and hr_parsed > 0:
        return np.log(hr_parsed)

    return None


def extract_trait_pairs_with_temporal_data(
    effect_sizes_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Extract trait pairs with temporal ordering and effect sizes.

    Args:
        effect_sizes_df: DataFrame with effect size data
        temporal_df: DataFrame with temporal metadata
        config: Configuration dictionary

    Returns:
        DataFrame with trait pairs, temporal order, and normalized effect sizes
    """
    logger.info("Extracting trait pairs with temporal data...")

    # ---- Merge with temporal metadata ----
    merged_df = effect_sizes_df.merge(temporal_df, on="pmid", how="inner")

    # ---- Filter to records with effect sizes ----
    with_effects = merged_df[merged_df["has_any_effect_size"]].copy()

    logger.info(
        f"Found {len(with_effects)} records with effect sizes "
        f"and temporal metadata"
    )

    # ---- Normalize effect sizes ----
    with_effects["effect_size_normalized"] = with_effects.apply(
        lambda row: normalize_effect_size(
            row["beta"], row["odds_ratio"], row["hazard_ratio"]
        ),
        axis=1,
    )

    # ---- Filter out invalid normalized values ----
    valid_effects = with_effects[
        with_effects["effect_size_normalized"].notna()
    ].copy()

    logger.info(f"Successfully normalized {len(valid_effects)} effect sizes")

    # ---- Create trait pair identifier ----
    valid_effects["trait_pair"] = (
        valid_effects["exposure"] + " -> " + valid_effects["outcome"]
    )

    # ---- Sort by trait pair and publication year ----
    valid_effects = valid_effects.sort_values(["trait_pair", "pub_year"])

    # ---- Assign study order within each trait pair ----
    valid_effects["study_order"] = (
        valid_effects.groupby("trait_pair").cumcount() + 1
    )
    valid_effects["n_studies_in_pair"] = valid_effects.groupby("trait_pair")[
        "pmid"
    ].transform("count")

    # ---- Filter to pairs with minimum number of studies ----
    min_studies = config["case_study_5"]["winners_curse"][
        "min_studies_per_pair"
    ]
    valid_pairs = valid_effects[
        valid_effects["n_studies_in_pair"] >= min_studies
    ].copy()

    logger.info(
        f"Found {valid_pairs['trait_pair'].nunique()} trait pairs "
        f"with >={min_studies} studies"
    )

    res = valid_pairs
    return res


def compute_temporal_decline(trait_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute temporal decline statistics for each trait pair.

    Uses linear regression: effect_size ~ study_order

    Args:
        trait_pairs_df: DataFrame with trait pairs and temporal data

    Returns:
        DataFrame with decline statistics per trait pair
    """
    logger.info("Computing temporal decline via linear regression...")

    decline_data = []

    for trait_pair, group_df in trait_pairs_df.groupby("trait_pair"):
        if len(group_df) < 3:
            continue

        X = group_df["study_order"].values.reshape(-1, 1)
        y = group_df["effect_size_normalized"].values

        # ---- Fit linear regression ----
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            X.flatten(), y
        )

        # ---- Calculate first and last effect sizes ----
        first_effect = group_df[group_df["study_order"] == 1][
            "effect_size_normalized"
        ].mean()
        last_effect = group_df[
            group_df["study_order"] == group_df["study_order"].max()
        ]["effect_size_normalized"].mean()

        # ---- Calculate percent decline ----
        if first_effect != 0:
            percent_decline = (
                100 * (first_effect - last_effect) / abs(first_effect)
            )
        else:
            percent_decline = 0.0

        decline_data.append(
            {
                "trait_pair": trait_pair,
                "n_studies": len(group_df),
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "std_err": std_err,
                "first_effect_size": first_effect,
                "last_effect_size": last_effect,
                "absolute_decline": first_effect - last_effect,
                "percent_decline": percent_decline,
                "mean_effect_size": y.mean(),
                "sd_effect_size": y.std(),
            }
        )

    df = pd.DataFrame(decline_data)

    # ---- Flag significant declines ----
    df["significant_decline"] = (df["p_value"] < 0.05) & (df["slope"] < 0)

    logger.info(
        f"Computed decline statistics for {len(df)} trait pairs "
        f"({df['significant_decline'].sum()} with significant decline)"
    )

    res = df
    return res


def compare_quartiles(trait_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Compare first vs last quartile effect sizes for each trait pair.

    Args:
        trait_pairs_df: DataFrame with trait pairs and temporal data

    Returns:
        DataFrame with quartile comparison statistics
    """
    logger.info("Comparing first vs last quartile effect sizes...")

    quartile_data = []

    for trait_pair, group_df in trait_pairs_df.groupby("trait_pair"):
        if len(group_df) < 8:
            continue

        # ---- Calculate quartiles based on study order ----
        n_studies = len(group_df)
        q1_cutoff = np.ceil(n_studies * 0.25)
        q4_cutoff = np.floor(n_studies * 0.75)

        q1_studies = group_df[group_df["study_order"] <= q1_cutoff]
        q4_studies = group_df[group_df["study_order"] > q4_cutoff]

        # ---- Calculate mean effect sizes ----
        q1_mean = q1_studies["effect_size_normalized"].mean()
        q4_mean = q4_studies["effect_size_normalized"].mean()

        # ---- Calculate effect size difference ----
        absolute_diff = q1_mean - q4_mean
        if q1_mean != 0:
            percent_diff = 100 * absolute_diff / abs(q1_mean)
        else:
            percent_diff = 0.0

        # ---- Perform t-test ----
        t_stat, t_pvalue = stats.ttest_ind(
            q1_studies["effect_size_normalized"],
            q4_studies["effect_size_normalized"],
        )

        quartile_data.append(
            {
                "trait_pair": trait_pair,
                "n_studies": n_studies,
                "n_q1_studies": len(q1_studies),
                "n_q4_studies": len(q4_studies),
                "q1_mean_effect": q1_mean,
                "q4_mean_effect": q4_mean,
                "absolute_difference": absolute_diff,
                "percent_difference": percent_diff,
                "t_statistic": t_stat,
                "t_pvalue": t_pvalue,
            }
        )

    df = pd.DataFrame(quartile_data)

    # ---- Flag significant differences ----
    df["significant_difference"] = (df["t_pvalue"] < 0.05) & (
        df["absolute_difference"] > 0
    )

    logger.info(
        f"Computed quartile comparisons for {len(df)} trait pairs "
        f"({df['significant_difference'].sum()} with significant difference)"
    )

    res = df
    return res


def identify_example_pairs(
    decline_df: pd.DataFrame, quartile_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Identify top examples of winner's curse effect.

    Args:
        decline_df: DataFrame with temporal decline statistics
        quartile_df: DataFrame with quartile comparisons
        top_n: Number of top examples to return

    Returns:
        DataFrame with top example pairs
    """
    logger.info(f"Identifying top {top_n} winner's curse examples...")

    # ---- Merge decline and quartile data ----
    merged_df = decline_df.merge(
        quartile_df[
            [
                "trait_pair",
                "q1_mean_effect",
                "q4_mean_effect",
                "absolute_difference",
                "percent_difference",
                "t_pvalue",
            ]
        ],
        on="trait_pair",
        how="left",
    )

    # ---- Score examples ----
    merged_df["winner_curse_score"] = merged_df["percent_decline"] * (
        1 - merged_df["p_value"]
    )

    # ---- Filter to significant declines ----
    significant_df = merged_df[
        (merged_df["significant_decline"])
        & (merged_df["percent_decline"] > 15)
    ].copy()

    # ---- Sort by winner's curse score ----
    top_examples = significant_df.nlargest(top_n, "winner_curse_score")

    logger.info(
        f"Identified {len(top_examples)} top winner's curse examples "
        f"(mean decline: {top_examples['percent_decline'].mean():.1f}%)"
    )

    res = top_examples
    return res


def generate_temporal_figures(
    trait_pairs_df: pd.DataFrame,
    example_pairs_df: pd.DataFrame,
    quartile_df: pd.DataFrame,
    config: Dict[str, Any],
    figures_dir: Path,
) -> None:
    """Generate figures for winner's curse analysis.

    Args:
        trait_pairs_df: DataFrame with trait pairs and temporal data
        example_pairs_df: DataFrame with top example pairs
        quartile_df: DataFrame with quartile comparisons
        config: Configuration dictionary
        figures_dir: Output directory for figures
    """
    logger.info("Generating temporal decline figures...")

    sns.set_style("whitegrid")
    fig_formats = config["figures"]["format"]
    dpi_setting = config["figures"]["dpi"]

    # ---- Figure 1: Temporal trends for top examples ----
    n_examples = min(6, len(example_pairs_df))
    if n_examples > 0:
        fig, axes = plt.subplots(2, 3, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (_, row) in enumerate(
            example_pairs_df.head(n_examples).iterrows()
        ):
            trait_pair = row["trait_pair"]
            pair_data = trait_pairs_df[
                trait_pairs_df["trait_pair"] == trait_pair
            ]

            ax = axes[idx]
            ax.scatter(
                pair_data["study_order"],
                pair_data["effect_size_normalized"],
                alpha=0.6,
                s=50,
            )

            # ---- Add regression line ----
            X = pair_data["study_order"].values
            y_pred = row["slope"] * X + row["intercept"]
            ax.plot(X, y_pred, "r--", linewidth=2, label="Trend")

            ax.set_xlabel("Study Order")
            ax.set_ylabel("Effect Size (log scale)")
            ax.set_title(
                f"{trait_pair[:40]}...\n"
                f"Decline: {row['percent_decline']:.1f}% (p={row['p_value']:.3f})",
                fontsize=9,
            )
            ax.legend()

        plt.tight_layout()

        for fmt in fig_formats:
            output_path = figures_dir / f"effect_size_temporal_trends.{fmt}"
            plt.savefig(output_path, dpi=dpi_setting, bbox_inches="tight")
            logger.info(f"Saved: {output_path}")

        plt.close()

    # ---- Figure 2: First vs last quartile comparison ----
    if len(quartile_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))

        significant_pairs = quartile_df[quartile_df["significant_difference"]]
        non_significant_pairs = quartile_df[
            ~quartile_df["significant_difference"]
        ]

        ax.scatter(
            non_significant_pairs["q1_mean_effect"],
            non_significant_pairs["q4_mean_effect"],
            alpha=0.4,
            s=30,
            c="gray",
            label="Non-significant",
        )

        ax.scatter(
            significant_pairs["q1_mean_effect"],
            significant_pairs["q4_mean_effect"],
            alpha=0.6,
            s=50,
            c="red",
            label="Significant decline",
        )

        # ---- Add identity line ----
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, zorder=0, label="No change")

        ax.set_xlabel("First Quartile Mean Effect Size")
        ax.set_ylabel("Last Quartile Mean Effect Size")
        ax.set_title(
            "Winner's Curse: First vs Last Quartile Effect Sizes\n"
            f"({len(significant_pairs)} significant declines / "
            f"{len(quartile_df)} trait pairs)"
        )
        ax.legend()

        plt.tight_layout()

        for fmt in fig_formats:
            output_path = figures_dir / f"first_vs_last_quartile.{fmt}"
            plt.savefig(output_path, dpi=dpi_setting, bbox_inches="tight")
            logger.info(f"Saved: {output_path}")

        plt.close()


def save_stage2_outputs(
    trait_pairs_df: pd.DataFrame,
    decline_df: pd.DataFrame,
    example_pairs_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save Stage 2 analysis outputs.

    Args:
        trait_pairs_df: DataFrame with trait pairs and temporal data
        decline_df: DataFrame with temporal decline statistics
        example_pairs_df: DataFrame with top example pairs
        config: Configuration dictionary
        output_dir: Output directory path
    """
    logger.info("Saving Stage 2 outputs...")

    # ---- Save trait pair temporal effects ----
    trait_pairs_df.to_csv(
        output_dir / "trait_pair_temporal_effects.csv", index=False
    )
    logger.info(f"Saved: {output_dir / 'trait_pair_temporal_effects.csv'}")

    # ---- Save decline summary ----
    decline_df.to_csv(output_dir / "winners_curse_summary.csv", index=False)
    logger.info(f"Saved: {output_dir / 'winners_curse_summary.csv'}")

    # ---- Save example pairs ----
    example_pairs_df.to_csv(output_dir / "example_pairs.csv", index=False)
    logger.info(f"Saved: {output_dir / 'example_pairs.csv'}")

    # ---- Update metadata ----
    with open(output_dir / "winners_curse_metadata.json", "r") as f:
        metadata = json.load(f)

    metadata["stage"] = "stage2_winners_curse_analysis"
    metadata["stage2_outputs"] = [
        "trait_pair_temporal_effects.csv",
        "winners_curse_summary.csv",
        "example_pairs.csv",
    ]
    metadata["stage2_figures"] = [
        "effect_size_temporal_trends.png",
        "effect_size_temporal_trends.svg",
        "first_vs_last_quartile.png",
        "first_vs_last_quartile.svg",
    ]
    metadata["n_trait_pairs_analyzed"] = int(len(decline_df))
    metadata["n_pairs_with_significant_decline"] = int(
        decline_df["significant_decline"].sum()
    )
    metadata["mean_percent_decline"] = float(
        decline_df["percent_decline"].mean()
    )

    metadata_converted = convert_to_native_types(metadata)

    with open(output_dir / "winners_curse_metadata.json", "w") as f:
        json.dump(metadata_converted, f, indent=2)
    logger.info(f"Updated: {output_dir / 'winners_curse_metadata.json'}")


def main() -> None:
    """Main execution function for winner's curse analysis."""
    args = parse_args()

    # ---- Load configuration ----
    config = load_config(args.config)

    # ---- Resolve database path ----
    if args.db:
        db_path = args.db
    else:
        db_path = PROJECT_ROOT / config["databases"]["vector_store"]

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    # ---- Resolve temporal metadata path ----
    if args.temporal_metadata:
        temporal_path = args.temporal_metadata
    else:
        temporal_path = (
            PROJECT_ROOT
            / config["output"]["case_study_5"]["temporal"]
            / "temporal_metadata.csv"
        )

    if not temporal_path.exists():
        logger.error(f"Temporal metadata not found: {temporal_path}")
        logger.error("Run case_study_5_temporal_preparation.py first")
        sys.exit(1)

    # ---- Create output directories ----
    output_dir, figures_dir = create_output_directories(config)

    if args.dry_run:
        logger.info("Dry run mode - no files will be written")
        logger.info(f"Would process: {db_path}")
        logger.info(f"Would use temporal data: {temporal_path}")
        logger.info(f"Would output to: {output_dir}")
        return

    # ---- Load data ----
    model_filter = config["case_study_5"]["models_included"][0]
    results_df = load_model_results(db_path, model_filter)
    temporal_df = load_temporal_metadata(temporal_path)

    # ==== STAGE 1: Availability Assessment ====
    logger.info("=" * 60)
    logger.info("STAGE 1: Effect Size Availability Assessment")
    logger.info("=" * 60)

    # ---- Extract effect sizes ----
    effect_sizes_df = extract_effect_sizes(results_df)

    # ---- Assess overall availability ----
    overall_availability = assess_overall_availability(effect_sizes_df)

    # ---- Assess subgroup availability ----
    subgroup_availability = assess_subgroup_availability(
        effect_sizes_df, temporal_df
    )

    # ---- Generate feasibility report ----
    feasibility_report = generate_feasibility_report(
        overall_availability, subgroup_availability, config
    )

    # ---- Save Stage 1 outputs ----
    save_stage1_outputs(
        overall_availability,
        subgroup_availability,
        feasibility_report,
        config,
        output_dir,
    )

    # ---- Check feasibility for Stage 2 ----
    n_feasible = subgroup_availability["analysis_feasible_flag"].sum()
    overall_pct = overall_availability[
        overall_availability["field_name"] == "any_effect_size"
    ]["percent_available"].values[0]
    min_threshold_pct = (
        config["case_study_5"]["winners_curse"]["min_effect_size_availability"]
        * 100
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 1 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Overall availability: {overall_pct:.2f}%")
    logger.info(
        f"Feasible subgroups: {n_feasible} / {len(subgroup_availability)}"
    )

    proceed_to_stage2 = n_feasible > 0 and not args.skip_stage2

    if not proceed_to_stage2:
        if args.skip_stage2:
            logger.info("Skipping Stage 2 (--skip-stage2 flag)")
        else:
            logger.warning(
                f"Effect size availability ({overall_pct:.2f}%) is below "
                f"feasibility threshold ({min_threshold_pct:.0f}%)"
            )
            logger.warning("Stage 2 winner's curse analysis NOT FEASIBLE")
            logger.warning(
                "See winners_curse_feasibility.md for recommendations"
            )
        logger.info("")
        logger.info("Analysis complete (Stage 1 only)")
        return

    # ==== STAGE 2: Winner's Curse Analysis ====
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Winner's Curse Analysis")
    logger.info("=" * 60)
    logger.info(
        "Sufficient data available - proceeding to temporal decline analysis"
    )

    # ---- Extract trait pairs with temporal ordering ----
    trait_pairs_df = extract_trait_pairs_with_temporal_data(
        effect_sizes_df, temporal_df, config
    )

    if len(trait_pairs_df) == 0:
        logger.error("No trait pairs found with sufficient studies")
        logger.error("Cannot proceed with Stage 2 analysis")
        return

    # ---- Compute temporal decline statistics ----
    decline_df = compute_temporal_decline(trait_pairs_df)

    if len(decline_df) == 0:
        logger.error("No trait pairs had sufficient data for decline analysis")
        logger.error("Cannot proceed with Stage 2 analysis")
        return

    # ---- Compare quartiles ----
    quartile_df = compare_quartiles(trait_pairs_df)

    # ---- Identify example pairs ----
    example_pairs_df = identify_example_pairs(
        decline_df, quartile_df, top_n=10
    )

    # ---- Generate figures ----
    generate_temporal_figures(
        trait_pairs_df, example_pairs_df, quartile_df, config, figures_dir
    )

    # ---- Save Stage 2 outputs ----
    save_stage2_outputs(
        trait_pairs_df, decline_df, example_pairs_df, config, output_dir
    )

    # ---- Summary statistics ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2 COMPLETE - Winner's Curse Analysis")
    logger.info("=" * 60)
    logger.info(f"Trait pairs analyzed: {len(decline_df)}")
    logger.info(
        f"Pairs with significant decline: {decline_df['significant_decline'].sum()}"
    )
    logger.info(
        f"Mean percent decline: {decline_df['percent_decline'].mean():.2f}%"
    )
    logger.info(f"Top examples identified: {len(example_pairs_df)}")
    logger.info("")
    logger.info("Analysis complete (Stages 1 and 2)")


if __name__ == "__main__":
    main()
