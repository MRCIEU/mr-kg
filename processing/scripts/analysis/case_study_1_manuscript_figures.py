"""Generate manuscript-ready figures for Case Study 1 using Altair.

This script creates publication-quality figures combining and refining base
plots from the CS1 analysis. All figures are exported as PNG for manuscript
inclusion.

Figures generated:
- Figure 1: Two-column plot showing reproducibility metrics distribution
  - Left panel (top): Overall tier distribution (stacked bar)
  - Left panel (bottom): Tier distribution by outcome category
  - Right panel: Concordance by match quality and outcome category
- Figure 2: Refined study count vs reproducibility histogram with labels

Outputs:
- data/artifacts/manuscript-figures/cs1_fig1_category_reproducibility.png
- data/artifacts/manuscript-figures/cs1_fig2_study_count_reproducibility.png
"""

import argparse
from pathlib import Path
from typing import Dict

import altair as alt
import pandas as pd
import yaml
from loguru import logger
from yiutils.project_utils import find_project_root

from color_schemes import (
    GRUVBOX_DARK,
    get_match_type_colors,
    get_tier_colors,
)

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
MANUSCRIPT_FIGURES_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "manuscript-figures"
)
MANUSCRIPT_TABLES_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "manuscript-tables"
)


# ==== Argument parsing ====


def parse_args():
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
        help="Perform dry run without generating figures",
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
        default=MANUSCRIPT_FIGURES_DIR,
        help=(
            "Override output directory from default "
            f"({MANUSCRIPT_FIGURES_DIR})"
        ),
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


# ==== Figure 1: Distribution of reproducibility metrics ====


def create_figure_1(
    metrics_df: pd.DataFrame, output_dir: Path, dry_run: bool = False
):
    """Create two-column plot of reproducibility metrics distribution.

    Left panel contains subplots A (overall tier distribution) and B (tier
    distribution by category) stacked vertically. Right panel contains
    subplot C (overall concordance distribution) and subplot D
    (category-specific concordance distribution). This layout groups the
    related tier distribution metrics together in the left panel, and
    concordance metrics in the right panel.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        dry_run: If True, show what would be done without executing
    """
    logger.info(
        "Creating Figure 1: Distribution of reproducibility metrics..."
    )

    if dry_run:
        logger.info("DRY RUN - Would generate Figure 1")
        return

    # ---- Load overall tier distribution data ----
    tier_dist_path = MANUSCRIPT_TABLES_DIR / "cs1_tier_distribution.csv"

    if not tier_dist_path.exists():
        logger.error(f"Tier distribution data not found: {tier_dist_path}")
        return

    tier_dist_df = pd.read_csv(tier_dist_path)

    # ---- Remove the "Total" row for visualization ----
    tier_dist_df = tier_dist_df[
        tier_dist_df["Reproducibility Tier"] != "Total"
    ].copy()

    # ---- Create subplot A: Overall tier distribution ----
    tier_order = ["High", "Moderate", "Low", "Discordant"]
    tier_colors = get_tier_colors(use_gruvbox=True)

    # Calculate cumulative percentages for text positioning
    tier_dist_df["tier_index"] = tier_dist_df["Reproducibility Tier"].map(
        {tier: i for i, tier in enumerate(tier_order)}
    )
    tier_dist_df = tier_dist_df.sort_values("tier_index")
    tier_dist_df["cumulative_pct"] = tier_dist_df["Percentage (%)"].cumsum()
    tier_dist_df["label_position"] = (
        tier_dist_df["cumulative_pct"] - tier_dist_df["Percentage (%)"] / 2
    )

    bars_overall = (
        alt.Chart(tier_dist_df)
        .mark_bar(size=50)
        .encode(
            x=alt.X(
                "Percentage (%):Q",
                title="Percentage (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "constant:N",
                title=None,
                axis=None,
            ),
            color=alt.Color(
                "Reproducibility Tier:N",
                scale=alt.Scale(
                    domain=tier_order,
                    range=[tier_colors[t] for t in tier_order],
                ),
                title="Reproducibility Tier",
                legend=None,
            ),
            order=alt.Order("tier_index:Q"),
        )
        .transform_calculate(constant="''")
    )

    # Add text labels showing pair counts
    text_labels_overall = (
        alt.Chart(tier_dist_df)
        .mark_text(
            align="center",
            baseline="middle",
            fontSize=11,
            color="black",
            fontWeight="bold",
        )
        .encode(
            x=alt.X("label_position:Q"),
            y=alt.Y(
                "constant:N",
                title=None,
                axis=None,
            ),
            text=alt.Text("Pairs (n):Q", format="d"),
            order=alt.Order("tier_index:Q"),
        )
        .transform_calculate(constant="''")
    )

    chartA = (bars_overall + text_labels_overall).properties(
        width=400,
        height=80,
        title=alt.TitleParams("A. Overall Tier Distribution", anchor="end"),
    )

    # ---- Prepare data for tier distribution by category subplot ----
    df_clean = metrics_df[
        (metrics_df["outcome_category"] != "uncategorized")
        & (metrics_df["outcome_category"].notna())
    ].copy()

    tier_counts = (
        df_clean.groupby(["outcome_category", "reproducibility_tier"])
        .size()
        .reset_index(name="count")
    )

    tier_order_lower = ["high", "moderate", "low", "discordant"]
    tier_totals = (
        tier_counts.groupby("outcome_category")["count"]
        .sum()
        .reset_index(name="total")
    )
    tier_counts = tier_counts.merge(tier_totals, on="outcome_category")
    tier_counts["percentage"] = (
        100 * tier_counts["count"] / tier_counts["total"]
    )

    # ---- Sort categories by high tier percentage ----
    high_tier_pct = (
        tier_counts[tier_counts["reproducibility_tier"] == "high"]
        .groupby("outcome_category")["percentage"]
        .first()
        .sort_values(ascending=True)
    )
    category_order = high_tier_pct.index.tolist()

    # ---- Create subplot B: Tier distribution by category ----
    bars = (
        alt.Chart(tier_counts)
        .mark_bar()
        .encode(
            x=alt.X(
                "percentage:Q",
                title="Percentage (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "outcome_category:N",
                title="Outcome Category",
                sort=category_order,
            ),
            color=alt.Color(
                "reproducibility_tier:N",
                scale=alt.Scale(
                    domain=tier_order_lower,
                    range=[
                        tier_colors[t.capitalize()] for t in tier_order_lower
                    ],
                ),
                title="Reproducibility Tier",
                legend=alt.Legend(orient="bottom"),
            ),
            order=alt.Order("tier_order:Q"),
        )
        .transform_calculate(
            tier_order=(
                f"indexof({tier_order_lower}, datum.reproducibility_tier)"
            )
        )
    )

    # ---- Add text labels to bars ----
    # Calculate cumulative percentages for proper text positioning
    tier_counts_sorted = tier_counts.sort_values(
        ["outcome_category", "reproducibility_tier"],
        key=lambda x: (
            x
            if x.name != "reproducibility_tier"
            else x.map(lambda v: tier_order_lower.index(v))
        ),
    )
    tier_counts_sorted["cumulative_pct"] = tier_counts_sorted.groupby(
        "outcome_category"
    )["percentage"].cumsum()
    tier_counts_sorted["label_position"] = (
        tier_counts_sorted["cumulative_pct"]
        - tier_counts_sorted["percentage"] / 2
    )

    text_labels = (
        alt.Chart(tier_counts_sorted)
        .mark_text(
            align="center",
            baseline="middle",
            fontSize=9,
            color="black",
        )
        .encode(
            x=alt.X("label_position:Q"),
            y=alt.Y("outcome_category:N", sort=category_order),
            text=alt.Text("count:Q", format="d"),
            order=alt.Order("tier_order:Q"),
        )
        .transform_calculate(
            tier_order=(
                f"indexof({tier_order_lower}, datum.reproducibility_tier)"
            )
        )
    )

    chartB = (bars + text_labels).properties(
        width=400,
        height=300,
        title=alt.TitleParams(
            "B. Tier Distribution by Outcome Category", anchor="end"
        ),
    )

    # ---- Prepare data for concordance subplots ----
    # Extract individual pair concordance values with match type
    df_concordance = metrics_df[
        (metrics_df["outcome_category"] != "uncategorized")
        & (metrics_df["outcome_category"].notna())
    ].copy()

    # Create match_type column based on has_exact_match
    df_concordance["match_type"] = df_concordance["has_exact_match"].apply(
        lambda x: "exact" if x else "fuzzy"
    )

    # Prepare data for subplot C (all categories combined)
    df_all_categories = df_concordance[
        ["match_type", "mean_direction_concordance"]
    ].copy()

    # Prepare data for subplot D (category-specific)
    df_by_category = df_concordance[
        ["outcome_category", "match_type", "mean_direction_concordance"]
    ].copy()

    # ---- Create subplot C: Overall concordance ridgeline plot ----
    match_type_colors = get_match_type_colors(use_gruvbox=True)

    # Use faceted approach for subplot C to match subplot D style
    step_c = 30
    overlap_c = 1.5

    base_chart_c = (
        alt.Chart(df_all_categories)
        .transform_density(
            density="mean_direction_concordance",
            bandwidth=0.15,
            groupby=["match_type"],
            as_=["concordance", "density"],
        )
        .mark_area(
            interpolate="monotone",
            fillOpacity=0.7,
            stroke="lightgray",
            strokeWidth=0.5,
        )
        .encode(
            x=alt.X(
                "concordance:Q",
                title="Direction Concordance",
                scale=alt.Scale(domain=[-1.0, 1.0]),
                axis=alt.Axis(ticks=False, domain=False, grid=False),
            ),
            y=alt.Y(
                "density:Q",
                axis=None,
                scale=alt.Scale(range=[step_c, -step_c * overlap_c]),
            ),
            color=alt.Color(
                "match_type:N",
                scale=alt.Scale(
                    domain=["exact", "fuzzy"],
                    range=[
                        match_type_colors["exact"],
                        match_type_colors["fuzzy"],
                    ],
                ),
                legend=None,
            ),
        )
        .properties(
            width=400,
            height=step_c,
        )
    )

    # chartC = base_chart_c.facet(
    #     row=alt.Row(
    #         "match_type:N",
    #         title=None,
    #         sort=["exact", "fuzzy"],
    #         header=alt.Header(
    #             labelAngle=0,
    #             labelAlign="left",
    #             labelFontSize=11,
    #         ),
    #     )
    # ).properties(
    #     bounds="flush",
    #     title=alt.TitleParams(
    #         "C. Overall Concordance Distribution by Match Type",
    #         anchor="end",
    #     )
    # )
    chartC = base_chart_c.properties(
        bounds="flush",
        title=alt.TitleParams(
            "C. Overall Concordance Distribution by Match Type",
            anchor="end",
        ),
    )

    # ---- Create subplot D: Category-specific ridgeline plots ----
    # We need subplot D to have the same total height as subplot B (300px)
    # With 6 categories, each should get 50px height

    step_d = 42
    overlap_d = 1.5

    base_chart_d = (
        alt.Chart(df_by_category)
        .transform_density(
            density="mean_direction_concordance",
            bandwidth=0.15,
            groupby=["outcome_category", "match_type"],
            as_=["concordance", "density"],
        )
        .mark_area(
            interpolate="monotone",
            fillOpacity=0.7,
            stroke="lightgray",
            strokeWidth=0.5,
        )
        .encode(
            x=alt.X(
                "concordance:Q",
                title="Direction Concordance",
                scale=alt.Scale(domain=[-1.0, 1.0]),
                axis=alt.Axis(ticks=False, domain=False, grid=False),
            ),
            y=alt.Y(
                "density:Q",
                axis=None,
                scale=alt.Scale(range=[step_d, -step_d * overlap_d]),
            ),
            color=alt.Color(
                "match_type:N",
                scale=alt.Scale(
                    domain=["exact", "fuzzy"],
                    range=[
                        match_type_colors["exact"],
                        match_type_colors["fuzzy"],
                    ],
                ),
                legend=alt.Legend(
                    title="Match Type",
                    orient="bottom",
                    direction="horizontal",
                    symbolType="square",
                    symbolSize=150,
                    titleFontSize=11,
                    labelFontSize=10,
                ),
            ),
        )
        .properties(
            width=400,
            height=step_d,
        )
    )

    chartD = base_chart_d.facet(
        row=alt.Row(
            "outcome_category:N",
            title="Outcome Category",
            sort=category_order,
            header=alt.Header(
                labelAngle=0,
                labelAlign="left",
                labelFontSize=11,
            ),
        )
    ).properties(
        bounds="flush",
        title=alt.TitleParams(
            "D. Concordance Distribution by Match Type and Outcome Category",
            anchor="end",
        ),
    )

    # ---- Combine subplots ----
    # Left panel: subplots A and B stacked vertically (tier distributions)
    # Right panel: subplots C and D stacked vertically (concordance distributions)
    # Add spacer above chartC to align it with chartA
    spacer = (
        alt.Chart()
        .mark_text()
        .encode()
        .properties(
            width=400,
            height=1,
        )
    )

    left_panel = alt.vconcat(spacer, chartA, chartB).resolve_scale(color="independent")
    right_panel = alt.vconcat(spacer, chartC, chartD).resolve_scale(
        color="independent"
    )

    combined = (
        alt.hconcat(left_panel, right_panel)
        .resolve_scale(color="independent")
        .properties(
            title=alt.TitleParams(
                "Distribution of reproducibility metrics",
                fontSize=16,
                anchor="middle",
            )
        )
        .configure_facet(spacing=0)
        .configure_view(stroke=None)
    )

    # ---- Save figure ----
    output_file = output_dir / "cs1_fig1_category_reproducibility.png"
    combined.save(str(output_file), ppi=300)
    logger.info(f"Saved figure: {output_file}")


# ==== Figure 2: Study count and reproducibility ====


def create_figure_2(
    metrics_df: pd.DataFrame, output_dir: Path, dry_run: bool = False
):
    """Create refined histogram of study count vs reproducibility.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        dry_run: If True, show what would be done without executing
    """
    logger.info(
        "Creating Figure 2: Study count and reproducibility histogram..."
    )

    if dry_run:
        logger.info("DRY RUN - Would generate Figure 2")
        return

    # ---- Prepare data ----
    df_clean = metrics_df[
        ["study_count_band", "mean_direction_concordance"]
    ].dropna()

    band_order = ["2-3", "4-6", "7-10", "11+"]
    df_clean = df_clean[df_clean["study_count_band"].isin(band_order)]

    # ---- Compute mean values for each band ----
    means = (
        df_clean.groupby("study_count_band")["mean_direction_concordance"]
        .mean()
        .reset_index(name="mean_conc")
    )

    # ---- Add mean_conc and threshold to df_clean for unified data ----
    df_with_lines = df_clean.merge(means, on="study_count_band")
    df_with_lines["threshold"] = 0.7

    # ---- Create histogram ----
    histogram = (
        alt.Chart()
        .mark_bar(color=GRUVBOX_DARK["blue"], opacity=0.7)
        .encode(
            x=alt.X(
                "mean_direction_concordance:Q",
                bin=alt.Bin(maxbins=20),
                title="Direction Concordance",
                scale=alt.Scale(domain=[-1, 1]),
            ),
            y=alt.Y("count():Q", title="Frequency"),
        )
    )

    # ---- Add mean lines (one per facet) ----
    mean_lines = (
        alt.Chart()
        .mark_rule(color=GRUVBOX_DARK["red"], strokeWidth=2)
        .encode(
            x=alt.X("mean_conc:Q"),
        )
        .transform_aggregate(
            mean_conc="mean(mean_conc)", groupby=["study_count_band"]
        )
    )

    # ---- Add text labels to mean lines ----
    mean_labels = (
        alt.Chart()
        .mark_text(
            align="left",
            baseline="top",
            dx=5,
            dy=10,
            fontSize=10,
            color=GRUVBOX_DARK["red"],
            fontWeight="bold",
        )
        .encode(
            x=alt.X("mean_conc:Q"),
            y=alt.value(10),
            text=alt.Text("mean_conc:Q", format=".2f"),
        )
        .transform_aggregate(
            mean_conc="mean(mean_conc)", groupby=["study_count_band"]
        )
    )

    # ---- Add high tier threshold line (0.7) ----
    threshold_lines = (
        alt.Chart()
        .mark_rule(
            color=GRUVBOX_DARK["green"], strokeDash=[5, 5], strokeWidth=1.5
        )
        .encode(x=alt.X("threshold:Q"))
    )

    # ---- Create legend for lines ----
    # Create a dummy dataset for legend
    legend_data = pd.DataFrame(
        [
            {"type": "Mean concordance", "color": GRUVBOX_DARK["red"]},
            {"type": "High tier (0.7)", "color": GRUVBOX_DARK["green"]},
        ]
    )

    legend_chart = (
        alt.Chart(legend_data)
        .mark_point(size=0, opacity=0)
        .encode(
            color=alt.Color(
                "type:N",
                scale=alt.Scale(
                    domain=["Mean concordance", "High tier (0.7)"],
                    range=[GRUVBOX_DARK["red"], GRUVBOX_DARK["green"]],
                ),
                legend=alt.Legend(
                    title="Reference Lines",
                    orient="bottom",
                    direction="horizontal",
                ),
            )
        )
    )

    # ---- Layer and then facet ----
    # Increase width by 1.5x (from default ~200 to 300)
    faceted = (
        alt.layer(
            histogram,
            mean_lines,
            mean_labels,
            threshold_lines,
            data=df_with_lines,
        )
        .properties(width=300, height=200)
        .facet(
            facet=alt.Facet(
                "study_count_band:N",
                sort=band_order,
                title=None,
                header=alt.Header(labelFontSize=12, titleFontSize=14),
            ),
            columns=2,
        )
        .resolve_scale(x="independent", y="independent")
    )

    # ---- Combine with legend ----
    combined = (
        alt.vconcat(
            faceted,
            legend_chart,
        )
        .properties(
            title=alt.TitleParams(
                f"Concordance Distribution by Study Count "
                f"(n={len(df_clean):,} pairs)",
                fontSize=16,
                anchor="middle",
            ),
        )
        .resolve_scale(color="independent")
    )

    # ---- Save figure ----
    output_file = output_dir / "cs1_fig2_study_count_reproducibility.png"
    combined.save(str(output_file), ppi=300)
    logger.info(f"Saved figure: {output_file}")


# ==== Main execution ====


def main():
    """Execute manuscript figure generation for Case Study 1."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Case Study 1: Manuscript figure generation")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    # ---- Determine input CSV path ----
    if args.input_csv:
        input_csv = args.input_csv
    else:
        metrics_dir = PROJECT_ROOT / output_config["metrics"]
        input_csv = metrics_dir / "pair_reproducibility_metrics.csv"

    # ---- Determine output directory ----
    output_dir = args.output_dir

    if args.dry_run:
        logger.info("Dry run - validating configuration and paths")
        logger.info("Input CSV: {}", input_csv)
        logger.info("Output directory: {}", output_dir)

        if not input_csv.exists():
            logger.error("Input CSV not found: {}", input_csv)
            return 1

        logger.info("Dry run complete - configuration validated")
        return 0

    # ---- Validate input CSV exists ----
    if not input_csv.exists():
        logger.error("Input CSV not found: {}", input_csv)
        logger.error(
            "Please run case_study_1_reproducibility_metrics.py first"
        )
        return 1

    # ---- Create output directory ----
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", output_dir)

    # ---- Load metrics data ----
    logger.info("Loading pair metrics from: {}", input_csv)
    metrics_df = pd.read_csv(input_csv)
    logger.info("Loaded {} trait pairs", len(metrics_df))

    # ---- Validate required columns ----
    required_cols = [
        "outcome_category",
        "reproducibility_tier",
        "study_count_band",
        "mean_direction_concordance",
    ]
    missing_cols = [
        col for col in required_cols if col not in metrics_df.columns
    ]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return 1

    # ---- Generate figures ----
    create_figure_1(metrics_df, output_dir, dry_run=args.dry_run)
    create_figure_2(metrics_df, output_dir, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info("Case Study 1 manuscript figures generation complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
