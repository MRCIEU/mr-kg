"""Generate manuscript-ready figures for Case Study 1 using Altair.

This script creates publication-quality figures combining and refining base
plots from the CS1 analysis. All figures are exported as PNG for manuscript
inclusion.

Figures generated:
- Figure 1: Faceted plot combining reproducibility tier distribution and
  outcome category concordance
- Figure 2: Refined study count vs reproducibility histogram with labels

Outputs:
- data/artifacts/manuscript-figures/cs1_fig1_category_reproducibility.png
- data/artifacts/manuscript-figures/cs1_fig2_study_count_reproducibility.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import altair as alt
import pandas as pd
import yaml
from loguru import logger
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
MANUSCRIPT_FIGURES_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "manuscript-figures"
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


# ==== Figure 1: Category reproducibility faceted plot ====


def create_figure_1(
    metrics_df: pd.DataFrame, output_dir: Path, dry_run: bool = False
):
    """Create faceted plot of category tier distribution and concordance.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        dry_run: If True, show what would be done without executing
    """
    logger.info("Creating Figure 1: Category reproducibility faceted plot...")

    if dry_run:
        logger.info("DRY RUN - Would generate Figure 1")
        return

    # ---- Prepare data for tier distribution subplot ----
    df_clean = metrics_df[
        (metrics_df["outcome_category"] != "uncategorized")
        & (metrics_df["outcome_category"].notna())
    ].copy()

    tier_counts = (
        df_clean.groupby(["outcome_category", "reproducibility_tier"])
        .size()
        .reset_index(name="count")
    )

    tier_order = ["high", "moderate", "low", "discordant"]
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

    # ---- Create subplot 1: Tier distribution ----
    tier_colors = {
        "high": "#2ecc71",
        "moderate": "#f39c12",
        "low": "#e74c3c",
        "discordant": "#95a5a6",
    }

    chart1 = (
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
                    domain=tier_order,
                    range=[tier_colors[t] for t in tier_order],
                ),
                title="Reproducibility Tier",
                legend=alt.Legend(orient="bottom"),
            ),
            order=alt.Order("tier_order:Q"),
        )
        .transform_calculate(
            tier_order=(f"indexof({tier_order}, datum.reproducibility_tier)")
        )
        .properties(
            width=400,
            height=300,
            title="Tier Distribution by Outcome Category",
        )
    )

    # ---- Prepare data for concordance subplot ----
    category_match_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "case-study-cs1"
        / "interactions"
        / "category_match_interaction.csv"
    )

    if not category_match_path.exists():
        logger.warning(
            f"Category match interaction data not found: {category_match_path}"
        )
        logger.info("Generating only subplot 1 (tier distribution)")
        output_file = output_dir / "cs1_fig1_category_reproducibility.png"
        chart1.save(str(output_file), ppi=300)
        logger.info(f"Saved figure: {output_file}")
        return

    match_df = pd.read_csv(category_match_path)

    # ---- Create subplot 2: Match type concordance ----
    chart2 = (
        alt.Chart(match_df)
        .mark_bar(size=20)
        .encode(
            x=alt.X(
                "mean_concordance:Q",
                title="Mean Direction Concordance",
                scale=alt.Scale(domain=[-0.1, 1.0]),
            ),
            y=alt.Y(
                "category:N",
                title="Outcome Category",
                sort=category_order,
            ),
            color=alt.Color(
                "match_type:N",
                scale=alt.Scale(
                    domain=["exact", "fuzzy"],
                    range=["#2E7D32", "#F57C00"],
                ),
                title="Match Type",
                legend=alt.Legend(orient="bottom"),
            ),
            yOffset=alt.YOffset("match_type:N"),
        )
        .properties(
            width=400,
            height=300,
            title="Concordance by Match Quality and Outcome Category",
        )
    )

    # ---- Add error bars ----
    error_bars = (
        alt.Chart(match_df)
        .mark_errorbar()
        .encode(
            x=alt.X("ci_lower:Q"),
            x2=alt.X2("ci_upper:Q"),
            y=alt.Y("category:N", sort=category_order),
            yOffset=alt.YOffset("match_type:N"),
        )
    )

    chart2_with_errors = chart2 + error_bars

    # ---- Combine subplots ----
    combined = alt.hconcat(chart1, chart2_with_errors).properties(
        title=alt.TitleParams(
            f"Reproducibility and Outcome Category (n={len(df_clean):,} "
            f"pairs)",
            fontSize=16,
            anchor="middle",
        )
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

    # ---- Create histogram ----
    histogram = (
        alt.Chart(df_clean)
        .mark_bar(color="steelblue", opacity=0.7)
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

    # ---- Add mean lines ----
    mean_lines = (
        alt.Chart(means)
        .mark_rule(color="red", strokeWidth=2)
        .encode(
            x=alt.X("mean_conc:Q", scale=alt.Scale(domain=[-1, 1])),
        )
    )

    # ---- Add high tier threshold line (0.7) ----
    threshold_df = pd.DataFrame([{"threshold": 0.7}])
    threshold_lines = (
        alt.Chart(threshold_df)
        .mark_rule(color="green", strokeDash=[5, 5], strokeWidth=1.5)
        .encode(
            x=alt.X("threshold:Q", scale=alt.Scale(domain=[-1, 1])),
        )
    )

    # ---- Layer charts and then facet ----
    combined = (
        alt.layer(histogram, mean_lines, threshold_lines, data=df_clean)
        .facet(
            facet=alt.Facet(
                "study_count_band:N",
                sort=band_order,
                title=None,
                header=alt.Header(labelFontSize=12, titleFontSize=14),
            ),
            columns=2,
        )
        .properties(
            title=alt.TitleParams(
                f"Concordance Distribution by Study Count "
                f"(n={len(df_clean):,} pairs)",
                fontSize=16,
                anchor="middle",
            ),
        )
        .resolve_scale(x="independent", y="independent")
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
