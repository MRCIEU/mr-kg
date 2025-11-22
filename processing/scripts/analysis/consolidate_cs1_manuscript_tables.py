#!/usr/bin/env python3

"""
Consolidate Case Study 1 reproducibility analysis into manuscript tables.

Produces LaTeX-formatted tables for:
1. Overall reproducibility tier distribution
2. Concordance statistics by match type and outcome category
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "case-study-cs1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts" / "manuscript-tables"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- --input-dir ----
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing CS1 analysis outputs (default: {DEFAULT_INPUT_DIR})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for manuscript tables (default: {DEFAULT_OUTPUT_DIR})",
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    return parser.parse_args()


def load_tier_distribution(input_dir: Path) -> pd.DataFrame:
    """Load overall tier distribution."""
    tier_path = input_dir / "metrics" / "tier_distribution.csv"
    return pd.read_csv(tier_path)


def load_pair_metrics(input_dir: Path) -> pd.DataFrame:
    """Load individual pair metrics for concordance analysis."""
    metrics_path = input_dir / "metrics" / "pair_reproducibility_metrics.csv"
    return pd.read_csv(metrics_path)


def create_tier_distribution_table(tier_dist: pd.DataFrame) -> pd.DataFrame:
    """Create tier distribution table with percentages."""
    total = tier_dist["count"].sum()

    table = tier_dist.copy()
    table["percentage"] = (table["count"] / total * 100).round(1)

    table["tier"] = table["tier"].str.title()

    table.columns = [
        "Reproducibility Tier",
        "Pairs (n)",
        "Percentage (%)",
    ]

    tier_order = ["High", "Moderate", "Low", "Discordant"]
    table["Reproducibility Tier"] = pd.Categorical(
        table["Reproducibility Tier"], categories=tier_order, ordered=True
    )
    table = table.sort_values("Reproducibility Tier")

    total_row = pd.DataFrame(
        {
            "Reproducibility Tier": ["Total"],
            "Pairs (n)": [total],
            "Percentage (%)": [100.0],
        }
    )
    table = pd.concat([table, total_row], ignore_index=True)

    return table


def create_match_type_by_category_table(
    pair_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Create concordance statistics by match type and outcome category.

    This table shows summary statistics for the concordance distributions
    displayed in subplots C (overall) and D (by category) of the manuscript
    figures. Overall statistics are concatenated at the top.
    """
    # Filter to valid data
    df_concordance = pair_metrics[
        (pair_metrics["outcome_category"] != "uncategorized")
        & (pair_metrics["outcome_category"].notna())
    ].copy()

    # Create match_type column
    df_concordance["match_type"] = df_concordance["has_exact_match"].apply(
        lambda x: "exact" if x else "fuzzy"
    )

    # Compute overall statistics (for subplot C)
    overall_stats = (
        df_concordance.groupby("match_type")["mean_direction_concordance"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    overall_stats.insert(0, "outcome_category", "All")

    # Compute category-specific statistics (for subplot D)
    category_stats = (
        df_concordance.groupby(["outcome_category", "match_type"])[
            "mean_direction_concordance"
        ]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    # Concatenate overall at the top
    combined_stats = pd.concat(
        [overall_stats, category_stats], ignore_index=True
    )

    # Rename and format match type labels first
    combined_stats["match_type"] = (
        combined_stats["match_type"].str.replace("_", " ").str.title()
    )

    # Rename columns
    combined_stats.columns = [
        "Outcome Category",
        "Match Type",
        "N",
        "Mean",
        "Median",
        "SD",
    ]

    # Format numeric columns
    combined_stats["Mean"] = combined_stats["Mean"].round(3)
    combined_stats["Median"] = combined_stats["Median"].round(3)
    combined_stats["SD"] = combined_stats["SD"].round(3)

    # Sort: All first, then by category name, then by match type
    combined_stats["sort_category"] = combined_stats["Outcome Category"].apply(
        lambda x: "0_All" if x == "All" else f"1_{x}"
    )
    combined_stats = combined_stats.sort_values(
        ["sort_category", "Match Type"], ascending=[True, False]
    )
    combined_stats = combined_stats.drop(columns=["sort_category"])

    return combined_stats

    return combined_stats


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format DataFrame as LaTeX table."""
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (len(df.columns) - 1),
        caption=caption,
        label=label,
    )
    return latex


def main():
    """Main function."""
    args = parse_args()

    if args.dry_run:
        print("[DRY RUN] Would process:")
        print(f"  Input dir: {args.input_dir}")
        print(f"  Output dir: {args.output_dir}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    tier_dist = load_tier_distribution(args.input_dir)
    pair_metrics = load_pair_metrics(args.input_dir)

    print("Creating tier distribution table...")
    tier_table = create_tier_distribution_table(tier_dist)

    print("Creating match type by category table...")
    match_category_table = create_match_type_by_category_table(pair_metrics)

    print("Saving CSV tables...")
    tier_table.to_csv(
        args.output_dir / "cs1_tier_distribution.csv", index=False
    )
    match_category_table.to_csv(
        args.output_dir / "cs1_match_type_by_category.csv", index=False
    )

    print("Generating LaTeX tables...")
    tier_latex = format_latex_table(
        tier_table,
        caption=(
            "Distribution of reproducibility tiers across multi-study MR pairs."
        ),
        label="tab:cs1-tier-distribution",
    )

    match_category_latex = format_latex_table(
        match_category_table,
        caption=(
            "Concordance distribution statistics by match type overall (All) and "
            "by outcome category."
        ),
        label="tab:cs1-match-type-category",
    )

    with open(args.output_dir / "cs1_tier_distribution.tex", "w") as f:
        f.write(tier_latex)

    with open(args.output_dir / "cs1_match_type_by_category.tex", "w") as f:
        f.write(match_category_latex)

    print("\nOutput files:")
    print(f"  {args.output_dir / 'cs1_tier_distribution.csv'}")
    print(f"  {args.output_dir / 'cs1_tier_distribution.tex'}")
    print(f"  {args.output_dir / 'cs1_match_type_by_category.csv'}")
    print(f"  {args.output_dir / 'cs1_match_type_by_category.tex'}")

    metadata = {
        "script": Path(__file__).name,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "tables_generated": [
            "cs1_tier_distribution",
            "cs1_match_type_by_category",
        ],
    }

    with open(args.output_dir / "cs1_tables_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
