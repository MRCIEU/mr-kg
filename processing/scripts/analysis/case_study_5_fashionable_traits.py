"""Analyze fashionable trait trends over time for Case Study 5 (RQ4).

This script identifies traits that have become fashionable or experienced hype
cycles over time, analyzing how research focus has shifted across temporal
eras. All analyses filter to gpt-5 model only.

Research Question 4:
Have fashionable trait pairs changed over time?

Input:
    - data/db/vector_store.db (model_result_traits, model_results)
    - data/processed/case-study-cs5/temporal/temporal_metadata.csv
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs5/fashionable/
        top_traits_by_year.csv
        trait_growth_rates.csv
        era_dominant_traits.csv
        trait_temporal_profiles.csv
        fashionable_traits_summary.md
        fashionable_traits_metadata.json
    - data/processed/case-study-cs5/figures/
        trait_popularity_heatmap.png/svg
        top_traits_trajectories.png/svg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def load_trait_data(
    db_path: Path, model_filter: str = "gpt-5"
) -> pd.DataFrame:
    """Load trait occurrences from vector_store.

    Args:
        db_path: Path to vector_store.db
        model_filter: Model to filter results (default: gpt-5)

    Returns:
        DataFrame with trait occurrences
    """
    logger.info(f"Loading trait data from {db_path}...")

    con = duckdb.connect(str(db_path), read_only=True)

    query = """
    SELECT
        mrt.trait_label,
        mrt.trait_index,
        mr.pmid,
        mr.model
    FROM model_result_traits mrt
    JOIN model_results mr ON mrt.model_result_id = mr.id
    WHERE mr.model = ?
    ORDER BY mr.pmid, mrt.trait_index
    """

    df = con.execute(query, [model_filter]).fetchdf()
    con.close()

    logger.info(f"Loaded {len(df)} trait occurrences for model {model_filter}")

    # ---- Validate model filtering ----
    unique_models = df["model"].unique()
    if len(unique_models) != 1 or unique_models[0] != model_filter:
        logger.error(
            f"Model filtering failed: expected {model_filter}, "
            f"got {unique_models}"
        )
        sys.exit(1)

    logger.info(
        f"VALIDATION: All {len(df)} trait occurrences are model={model_filter}"
    )

    res = df
    return res


def load_temporal_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load temporal metadata from Phase 1.

    Args:
        metadata_path: Path to temporal_metadata.csv

    Returns:
        DataFrame with temporal metadata
    """
    logger.info(f"Loading temporal metadata from {metadata_path}...")

    df = pd.read_csv(metadata_path)
    
    # ---- Ensure PMID is string type to match trait data ----
    df["pmid"] = df["pmid"].astype(str)

    logger.info(f"Loaded temporal metadata for {len(df)} studies")

    res = df
    return res


def compute_top_traits_by_year(
    df: pd.DataFrame, top_k: int = 10
) -> pd.DataFrame:
    """Compute top K most studied traits per year.

    Args:
        df: DataFrame with trait occurrences and temporal metadata
        top_k: Number of top traits to return per year

    Returns:
        DataFrame with top traits by year
    """
    logger.info(f"Computing top {top_k} traits by year...")

    # ---- Filter to valid years ----
    df_valid = df[df["pub_year"].notna()].copy()

    # ---- Count occurrences per trait per year ----
    year_trait_counts = (
        df_valid.groupby(["pub_year", "trait_label"])
        .agg(occurrence_count=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Compute total studies per year ----
    year_totals = (
        df_valid.groupby("pub_year")
        .agg(total_studies=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Merge and compute percentage ----
    year_trait_counts = year_trait_counts.merge(
        year_totals, on="pub_year", how="left"
    )
    year_trait_counts["percent_of_studies"] = (
        year_trait_counts["occurrence_count"]
        / year_trait_counts["total_studies"]
        * 100
    )

    # ---- Rank within year ----
    year_trait_counts["rank"] = year_trait_counts.groupby("pub_year")[
        "occurrence_count"
    ].rank(method="dense", ascending=False)

    # ---- Filter to top K per year ----
    top_traits = year_trait_counts[year_trait_counts["rank"] <= top_k].copy()

    # ---- Sort by year and rank ----
    top_traits = top_traits.sort_values(["pub_year", "rank"])

    logger.info(
        f"Computed top traits for {top_traits['pub_year'].nunique()} years"
    )

    res = top_traits
    return res


def compute_trait_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year growth rates for traits.

    Args:
        df: DataFrame with trait occurrences and temporal metadata

    Returns:
        DataFrame with growth rates and hype cycle flags
    """
    logger.info("Computing trait growth rates...")

    # ---- Filter to valid years ----
    df_valid = df[df["pub_year"].notna()].copy()

    # ---- Count occurrences per trait per year ----
    trait_year_counts = (
        df_valid.groupby(["trait_label", "pub_year"])
        .agg(occurrence_count=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Sort by trait and year ----
    trait_year_counts = trait_year_counts.sort_values(
        ["trait_label", "pub_year"]
    )

    # ---- Compute year-over-year growth rates ----
    trait_year_counts["yoy_growth_rate"] = trait_year_counts.groupby(
        "trait_label"
    )["occurrence_count"].pct_change()

    logger.info(f"Computed growth rates for {len(trait_year_counts)} records")

    res = trait_year_counts
    return res


def detect_hype_cycles(
    growth_df: pd.DataFrame,
    growth_threshold: float,
    decline_threshold: float,
) -> pd.DataFrame:
    """Detect hype cycles in trait popularity.

    A hype cycle is defined as:
    - At least one year with >growth_threshold YoY growth
    - Followed by at least one year with >decline_threshold YoY decline

    Args:
        growth_df: DataFrame with growth rates
        growth_threshold: Minimum growth rate for rapid rise (e.g., 1.0 = 100%)
        decline_threshold: Minimum decline rate for fall (e.g., 0.5 = 50%)

    Returns:
        DataFrame with hype cycle flags
    """
    logger.info("Detecting hype cycles...")

    # ---- Flag rapid growth and decline ----
    growth_df["rapid_growth"] = (
        growth_df["yoy_growth_rate"] > growth_threshold
    ).astype(int)
    growth_df["rapid_decline"] = (
        growth_df["yoy_growth_rate"] < -decline_threshold
    ).astype(int)

    # ---- Detect hype cycles per trait ----
    hype_traits = []

    for trait_label, group in growth_df.groupby("trait_label"):
        group = group.sort_values("pub_year")

        # ---- Check if both growth and decline present ----
        has_growth = group["rapid_growth"].sum() > 0
        has_decline = group["rapid_decline"].sum() > 0

        if has_growth and has_decline:
            # ---- Check if decline comes after growth ----
            growth_years = group[group["rapid_growth"] == 1]["pub_year"].values
            decline_years = group[group["rapid_decline"] == 1][
                "pub_year"
            ].values

            if len(growth_years) > 0 and len(decline_years) > 0:
                first_growth = growth_years.min()
                first_decline = decline_years.min()

                if first_decline > first_growth:
                    hype_traits.append(trait_label)

    # ---- Flag hype cycle traits ----
    growth_df["hype_cycle_flag"] = (
        growth_df["trait_label"].isin(hype_traits).astype(int)
    )

    logger.info(f"Detected {len(hype_traits)} traits with hype cycles")

    res = growth_df
    return res


def compute_era_dominant_traits(
    df: pd.DataFrame, config: Dict[str, Any], top_k: int = 10
) -> pd.DataFrame:
    """Identify dominant traits per era.

    Args:
        df: DataFrame with trait occurrences and temporal metadata
        config: Configuration dictionary
        top_k: Number of top traits to return per era

    Returns:
        DataFrame with era-dominant traits
    """
    logger.info(f"Computing top {top_k} traits per era...")

    # ---- Filter to valid eras ----
    df_valid = df[df["era"] != "unknown"].copy()

    # ---- Count occurrences per trait per era ----
    era_trait_counts = (
        df_valid.groupby(["era", "trait_label"])
        .agg(occurrence_count=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Compute total studies per era ----
    era_totals = (
        df_valid.groupby("era")
        .agg(total_studies=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Merge and compute percentage ----
    era_trait_counts = era_trait_counts.merge(era_totals, on="era", how="left")
    era_trait_counts["percent_of_era_studies"] = (
        era_trait_counts["occurrence_count"]
        / era_trait_counts["total_studies"]
        * 100
    )

    # ---- Rank within era ----
    era_trait_counts["rank_within_era"] = era_trait_counts.groupby("era")[
        "occurrence_count"
    ].rank(method="dense", ascending=False)

    # ---- Filter to top K per era ----
    dominant_traits = era_trait_counts[
        era_trait_counts["rank_within_era"] <= top_k
    ].copy()

    # ---- Sort by era and rank ----
    era_order = list(config["case_study_5"]["temporal_eras"].keys())
    dominant_traits["era_order"] = dominant_traits["era"].map(
        {era: i for i, era in enumerate(era_order)}
    )
    dominant_traits = dominant_traits.sort_values(
        ["era_order", "rank_within_era"]
    ).drop(columns=["era_order"])

    logger.info(
        f"Computed dominant traits for {dominant_traits['era'].nunique()} eras"
    )

    res = dominant_traits
    return res


def compute_trait_temporal_profiles(
    df: pd.DataFrame, min_occurrences: int = 3
) -> pd.DataFrame:
    """Compute temporal profiles for traits.

    Args:
        df: DataFrame with trait occurrences and temporal metadata
        min_occurrences: Minimum total studies for inclusion

    Returns:
        DataFrame with trait temporal profiles
    """
    logger.info("Computing trait temporal profiles...")

    # ---- Filter to valid years ----
    df_valid = df[df["pub_year"].notna()].copy()

    # ---- Count occurrences per trait per year ----
    trait_year_counts = (
        df_valid.groupby(["trait_label", "pub_year"])
        .agg(occurrence_count=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Compute profile statistics per trait ----
    profiles = []

    for trait_label, group in trait_year_counts.groupby("trait_label"):
        total_studies = group["occurrence_count"].sum()

        if total_studies < min_occurrences:
            continue

        profile = {
            "trait_label": trait_label,
            "first_year": int(group["pub_year"].min()),
            "peak_year": int(
                group.loc[group["occurrence_count"].idxmax(), "pub_year"]
            ),
            "last_year": int(group["pub_year"].max()),
            "peak_count": int(group["occurrence_count"].max()),
            "total_studies": int(total_studies),
            "n_years_active": int(group["pub_year"].nunique()),
            "stability_score": float(
                group["occurrence_count"].std()
                / group["occurrence_count"].mean()
                if group["occurrence_count"].mean() > 0
                else 0
            ),
        }

        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)

    # ---- Sort by total studies descending ----
    profiles_df = profiles_df.sort_values("total_studies", ascending=False)

    logger.info(f"Computed temporal profiles for {len(profiles_df)} traits")

    res = profiles_df
    return res


def generate_popularity_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 30,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """Generate trait popularity heatmap.

    Args:
        df: DataFrame with trait occurrences and temporal metadata
        output_dir: Directory to save figures
        top_n: Number of top traits to include
        figsize: Figure size
    """
    logger.info(f"Generating popularity heatmap for top {top_n} traits...")

    # ---- Filter to valid years ----
    df_valid = df[df["pub_year"].notna()].copy()

    # ---- Count occurrences per trait per year ----
    trait_year_counts = (
        df_valid.groupby(["trait_label", "pub_year"])
        .agg(occurrence_count=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Get top N traits by total occurrences ----
    top_traits = (
        trait_year_counts.groupby("trait_label")["occurrence_count"]
        .sum()
        .nlargest(top_n)
        .index
    )

    # ---- Filter to top traits ----
    trait_year_counts = trait_year_counts[
        trait_year_counts["trait_label"].isin(top_traits)
    ]

    # ---- Pivot to matrix ----
    heatmap_data = trait_year_counts.pivot(
        index="trait_label", columns="pub_year", values="occurrence_count"
    )
    heatmap_data = heatmap_data.fillna(0)

    # ---- Sort by peak year ----
    peak_years = heatmap_data.idxmax(axis=1)
    heatmap_data = heatmap_data.loc[peak_years.sort_values().index]

    # ---- Create heatmap ----
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        annot=False,
        fmt="d",
        cbar_kws={"label": "Number of Studies"},
        ax=ax,
        linewidths=0.1,
        linecolor="white",
    )

    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Trait", fontsize=12)
    ax.set_title(
        f"Trait Popularity Over Time (Top {top_n} Traits)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # ---- Save figures ----
    for ext in ["png", "svg"]:
        output_path = output_dir / f"trait_popularity_heatmap.{ext}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved heatmap: {output_path}")

    plt.close(fig)


def generate_trajectory_plot(
    df: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """Generate temporal trajectory plot for top traits.

    Args:
        df: DataFrame with trait occurrences and temporal metadata
        output_dir: Directory to save figures
        config: Configuration dictionary
        top_n: Number of top traits to include
        figsize: Figure size
    """
    logger.info(f"Generating trajectory plot for top {top_n} traits...")

    # ---- Filter to valid years ----
    df_valid = df[df["pub_year"].notna()].copy()

    # ---- Count occurrences per trait per year ----
    trait_year_counts = (
        df_valid.groupby(["trait_label", "pub_year"])
        .agg(occurrence_count=("pmid", "nunique"))
        .reset_index()
    )

    # ---- Get top N traits by total occurrences ----
    top_traits = (
        trait_year_counts.groupby("trait_label")["occurrence_count"]
        .sum()
        .nlargest(top_n)
        .index
    )

    # ---- Filter to top traits ----
    trait_year_counts = trait_year_counts[
        trait_year_counts["trait_label"].isin(top_traits)
    ]

    # ---- Create plot ----
    fig, ax = plt.subplots(figsize=figsize)

    # ---- Plot trajectories ----
    for trait_label, group in trait_year_counts.groupby("trait_label"):
        group = group.sort_values("pub_year")
        ax.plot(
            group["pub_year"],
            group["occurrence_count"],
            marker="o",
            markersize=4,
            alpha=0.7,
            label=trait_label,
            linewidth=1.5,
        )

    # ---- Add era boundaries ----
    era_defs = config["case_study_5"]["temporal_eras"]
    era_colors = ["red", "blue", "green", "orange", "purple"]

    for idx, (era_name, (start_year, end_year)) in enumerate(era_defs.items()):
        if idx > 0:
            ax.axvline(
                x=start_year,
                color=era_colors[idx % len(era_colors)],
                linestyle="--",
                alpha=0.3,
                linewidth=1,
            )

    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Number of Studies", fontsize=12)
    ax.set_title(
        f"Temporal Trajectories for Top {top_n} Traits",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
        ncol=1,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ---- Save figures ----
    for ext in ["png", "svg"]:
        output_path = output_dir / f"top_traits_trajectories.{ext}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved trajectory plot: {output_path}")

    plt.close(fig)


def generate_summary_narrative(
    top_traits_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    era_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate Markdown summary narrative.

    Args:
        top_traits_df: Top traits by year
        growth_df: Trait growth rates
        era_df: Era-dominant traits
        profiles_df: Trait temporal profiles
        output_dir: Directory to save summary
    """
    logger.info("Generating summary narrative...")

    # ---- Identify hype cycle traits ----
    hype_traits = growth_df[growth_df["hype_cycle_flag"] == 1][
        "trait_label"
    ].unique()

    # ---- Identify COVID-era traits ----
    covid_traits = top_traits_df[
        (top_traits_df["pub_year"] >= 2020)
        & (top_traits_df["pub_year"] <= 2021)
        & (top_traits_df["rank"] <= 5)
    ]["trait_label"].unique()

    # ---- Identify recent trends ----
    recent_traits = top_traits_df[
        (top_traits_df["pub_year"] >= 2022) & (top_traits_df["rank"] <= 5)
    ]["trait_label"].unique()

    # ---- Generate narrative ----
    lines = [
        "# Fashionable Traits Analysis Summary",
        "",
        "## Overview",
        "",
        f"This analysis identified temporal patterns in trait popularity "
        f"across {top_traits_df['pub_year'].nunique()} years of MR research. "
        f"We analyzed {profiles_df['trait_label'].nunique()} traits with "
        f"sufficient data to assess trends, hype cycles, and era-specific "
        f"dominance.",
        "",
        "## Key Findings",
        "",
        f"### Hype Cycles Detected: {len(hype_traits)}",
        "",
        "The following traits exhibited hype cycle patterns (rapid growth "
        "followed by substantial decline):",
        "",
    ]

    if len(hype_traits) > 0:
        for trait in hype_traits[:10]:
            lines.append(f"- {trait}")
    else:
        lines.append("- None detected with current thresholds")

    lines.extend(
        [
            "",
            "### COVID-19 Era Trends (2020-2021)",
            "",
            "Top traits during the COVID-19 era:",
            "",
        ]
    )

    if len(covid_traits) > 0:
        for trait in covid_traits[:10]:
            lines.append(f"- {trait}")
    else:
        lines.append("- Insufficient data for COVID era")

    lines.extend(
        [
            "",
            "### Recent Trends (2022-2024)",
            "",
            "Emerging traits in recent years:",
            "",
        ]
    )

    if len(recent_traits) > 0:
        for trait in recent_traits[:10]:
            lines.append(f"- {trait}")
    else:
        lines.append("- Insufficient data for recent period")

    lines.extend(
        [
            "",
            "## Era-Dominant Traits",
            "",
            "Top traits per methodological era:",
            "",
        ]
    )

    for era, group in era_df.groupby("era"):
        lines.append(f"### {era.replace('_', ' ').title()}")
        lines.append("")
        top_5 = group.nsmallest(5, "rank_within_era")
        for _, row in top_5.iterrows():
            lines.append(
                f"- {row['trait_label']} "
                f"({row['occurrence_count']} studies, "
                f"{row['percent_of_era_studies']:.1f}% of era)"
            )
        lines.append("")

    lines.extend(
        [
            "## Temporal Stability",
            "",
            "Traits with most stable research focus (low stability score):",
            "",
        ]
    )

    stable_traits = profiles_df.nsmallest(10, "stability_score")
    for _, row in stable_traits.iterrows():
        lines.append(
            f"- {row['trait_label']} "
            f"(stability: {row['stability_score']:.2f}, "
            f"active {row['n_years_active']} years)"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "### Methodological Evolution",
            "",
            "The temporal patterns reveal how MR research priorities have "
            "shifted alongside methodological advances. Early eras show "
            "focus on well-established cardiovascular and metabolic traits, "
            "while recent years demonstrate expansion into behavioral, "
            "psychiatric, and complex disease phenotypes.",
            "",
            "### Hype Cycles",
            "",
            "Detected hype cycles may reflect:",
            "- Initial enthusiasm following novel genetic discoveries",
            "- Subsequent recognition of methodological limitations",
            "- Publication bias favoring positive findings in early studies",
            "- Natural maturation as field moves to other traits",
            "",
            "### Future Directions",
            "",
            "Recent trends suggest continued diversification of trait "
            "selection, with potential growth in:",
            "- Aging-related phenotypes",
            "- Mental health outcomes",
            "- Gene-environment interactions",
            "- Multi-trait and multivariable MR applications",
        ]
    )

    # ---- Write summary ----
    summary_path = output_dir / "fashionable_traits_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated summary: {summary_path}")


def create_metadata_summary(
    top_traits_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    era_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Create JSON metadata summary.

    Args:
        top_traits_df: Top traits by year
        growth_df: Trait growth rates
        era_df: Era-dominant traits
        profiles_df: Trait temporal profiles
        config: Configuration dictionary

    Returns:
        Dictionary with metadata summary
    """
    logger.info("Creating metadata summary...")

    cs5_config = config["case_study_5"]

    # ---- Count hype cycles ----
    n_hype_cycles = growth_df[growth_df["hype_cycle_flag"] == 1][
        "trait_label"
    ].nunique()

    metadata = {
        "analysis": "fashionable_traits",
        "research_question": "RQ4: Fashionable trait trends",
        "model": cs5_config["models_included"][0],
        "summary_statistics": {
            "n_unique_traits": int(profiles_df["trait_label"].nunique()),
            "n_years_analyzed": int(top_traits_df["pub_year"].nunique()),
            "n_eras_analyzed": int(era_df["era"].nunique()),
            "n_hype_cycles_detected": int(n_hype_cycles),
            "top_k_per_year": cs5_config["fashionable_traits"][
                "top_k_per_year"
            ],
        },
        "thresholds": {
            "hype_cycle_growth_threshold": cs5_config["fashionable_traits"][
                "hype_cycle_growth_threshold"
            ],
            "hype_cycle_decline_threshold": cs5_config["fashionable_traits"][
                "hype_cycle_decline_threshold"
            ],
            "min_occurrences_for_trend": cs5_config["fashionable_traits"][
                "min_occurrences_for_trend"
            ],
        },
        "top_traits_overall": profiles_df.nlargest(10, "total_studies")[
            "trait_label"
        ].tolist(),
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
    logger.info("Case Study 5: Fashionable traits analysis (RQ4)")
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

    # ---- Get temporal metadata path ----
    if args.temporal_metadata:
        temporal_path = args.temporal_metadata
    else:
        temporal_path = (
            PROJECT_ROOT / Path(output_config["temporal"]) / "temporal_metadata.csv"
        )

    if not temporal_path.exists():
        logger.error(f"Temporal metadata not found: {temporal_path}")
        logger.error("Please run case_study_5_temporal_preparation.py first")
        sys.exit(1)

    # ---- Create output directories ----
    fashionable_dir = PROJECT_ROOT / Path(output_config["fashionable"])
    figures_dir = PROJECT_ROOT / Path(output_config["figures"])

    if not args.dry_run:
        fashionable_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directories")
    else:
        logger.info("Would create output directories")

    # ---- Load data ----
    model_filter = cs5_config["models_included"][0]
    trait_df = load_trait_data(db_path, model_filter=model_filter)
    temporal_df = load_temporal_metadata(temporal_path)

    # ---- Merge trait and temporal data ----
    logger.info("Merging trait and temporal data...")
    df = trait_df.merge(temporal_df, on=["pmid", "model"], how="left")

    n_missing_temporal = df["pub_year"].isna().sum()
    if n_missing_temporal > 0:
        logger.warning(
            f"{n_missing_temporal} trait occurrences missing temporal metadata"
        )

    # ---- Compute top traits by year ----
    top_k = cs5_config["fashionable_traits"]["top_k_per_year"]
    top_traits_df = compute_top_traits_by_year(df, top_k=top_k)

    # ---- Compute trait growth rates ----
    growth_df = compute_trait_growth_rates(df)

    # ---- Detect hype cycles ----
    growth_threshold = cs5_config["fashionable_traits"][
        "hype_cycle_growth_threshold"
    ]
    decline_threshold = cs5_config["fashionable_traits"][
        "hype_cycle_decline_threshold"
    ]
    growth_df = detect_hype_cycles(
        growth_df, growth_threshold, decline_threshold
    )

    # ---- Compute era-dominant traits ----
    era_df = compute_era_dominant_traits(df, config, top_k=top_k)

    # ---- Compute trait temporal profiles ----
    min_occurrences = cs5_config["fashionable_traits"][
        "min_occurrences_for_trend"
    ]
    profiles_df = compute_trait_temporal_profiles(df, min_occurrences)

    # ---- Generate figures ----
    if not args.dry_run:
        generate_popularity_heatmap(df, figures_dir, top_n=30)
        generate_trajectory_plot(df, figures_dir, config, top_n=20)

        # ---- Generate summary narrative ----
        generate_summary_narrative(
            top_traits_df, growth_df, era_df, profiles_df, fashionable_dir
        )

        # ---- Create metadata summary ----
        metadata = create_metadata_summary(
            top_traits_df, growth_df, era_df, profiles_df, config
        )

        # ---- Write outputs ----
        top_traits_csv = fashionable_dir / "top_traits_by_year.csv"
        top_traits_df.to_csv(top_traits_csv, index=False)
        logger.info(f"Wrote top traits by year: {top_traits_csv}")

        growth_csv = fashionable_dir / "trait_growth_rates.csv"
        growth_df.to_csv(growth_csv, index=False)
        logger.info(f"Wrote growth rates: {growth_csv}")

        era_csv = fashionable_dir / "era_dominant_traits.csv"
        era_df.to_csv(era_csv, index=False)
        logger.info(f"Wrote era-dominant traits: {era_csv}")

        profiles_csv = fashionable_dir / "trait_temporal_profiles.csv"
        profiles_df.to_csv(profiles_csv, index=False)
        logger.info(f"Wrote temporal profiles: {profiles_csv}")

        metadata_json = fashionable_dir / "fashionable_traits_metadata.json"
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Wrote metadata: {metadata_json}")

        logger.info("=" * 60)
        logger.info("Fashionable traits analysis complete!")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("DRY RUN - Summary of what would be written:")
        logger.info(f"  {fashionable_dir / 'top_traits_by_year.csv'}")
        logger.info(f"    {len(top_traits_df)} rows")
        logger.info(f"  {fashionable_dir / 'trait_growth_rates.csv'}")
        logger.info(f"    {len(growth_df)} rows")
        logger.info(f"  {fashionable_dir / 'era_dominant_traits.csv'}")
        logger.info(f"    {len(era_df)} rows")
        logger.info(f"  {fashionable_dir / 'trait_temporal_profiles.csv'}")
        logger.info(f"    {len(profiles_df)} rows")
        logger.info(f"  {fashionable_dir / 'fashionable_traits_summary.md'}")
        logger.info(
            f"  {fashionable_dir / 'fashionable_traits_metadata.json'}"
        )
        logger.info(f"  {figures_dir / 'trait_popularity_heatmap.png/svg'}")
        logger.info(f"  {figures_dir / 'top_traits_trajectories.png/svg'}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
