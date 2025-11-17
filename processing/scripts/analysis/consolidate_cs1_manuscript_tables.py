#!/usr/bin/env python3

"""
Consolidate Case Study 1 reproducibility analysis into manuscript tables.

Produces LaTeX-formatted tables for:
1. Overall reproducibility tier distribution
2. Stratification by match type quality
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


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
        default=Path("../data/processed/case-study-cs1"),
        help="Input directory containing CS1 analysis outputs",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/processed/manuscript-tables"),
        help="Output directory for manuscript tables",
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


def load_match_type_stratification(input_dir: Path) -> pd.DataFrame:
    """Load stratification by match type."""
    match_path = input_dir / "metrics" / "stratified_by_match_type.csv"
    return pd.read_csv(match_path)


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


def create_match_type_table(match_type: pd.DataFrame) -> pd.DataFrame:
    """Create match type stratification table."""
    table = match_type.copy()

    table["match_type"] = table["match_type"].str.replace("_", " ").str.title()

    result = table[
        [
            "match_type",
            "n_pairs",
            "n_high",
            "pct_high",
            "n_discordant",
            "pct_discordant",
        ]
    ].copy()

    result.columns = [
        "Match Type",
        "Total Pairs",
        "High (n)",
        "High (%)",
        "Discordant (n)",
        "Discordant (%)",
    ]

    result["High (%)"] = result["High (%)"].round(1)
    result["Discordant (%)"] = result["Discordant (%)"].round(1)

    match_order = ["Exact", "Fuzzy", "Efo"]
    result["Match Type"] = pd.Categorical(
        result["Match Type"], categories=match_order, ordered=True
    )
    result = result.sort_values("Match Type")

    return result


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
    match_type = load_match_type_stratification(args.input_dir)

    print("Creating tier distribution table...")
    tier_table = create_tier_distribution_table(tier_dist)

    print("Creating match type table...")
    match_table = create_match_type_table(match_type)

    print("Saving CSV tables...")
    tier_table.to_csv(
        args.output_dir / "cs1_tier_distribution.csv", index=False
    )
    match_table.to_csv(
        args.output_dir / "cs1_match_type_stratification.csv", index=False
    )

    print("Generating LaTeX tables...")
    tier_latex = format_latex_table(
        tier_table,
        caption=(
            "Distribution of reproducibility tiers across multi-study MR pairs."
        ),
        label="tab:cs1-tier-distribution",
    )

    match_latex = format_latex_table(
        match_table,
        caption=(
            "Reproducibility by trait matching quality: exact ontology matches "
            "show higher concordance than fuzzy matches."
        ),
        label="tab:cs1-match-type",
    )

    with open(args.output_dir / "cs1_tier_distribution.tex", "w") as f:
        f.write(tier_latex)

    with open(args.output_dir / "cs1_match_type_stratification.tex", "w") as f:
        f.write(match_latex)

    print("\nOutput files:")
    print(f"  {args.output_dir / 'cs1_tier_distribution.csv'}")
    print(f"  {args.output_dir / 'cs1_tier_distribution.tex'}")
    print(f"  {args.output_dir / 'cs1_match_type_stratification.csv'}")
    print(f"  {args.output_dir / 'cs1_match_type_stratification.tex'}")

    metadata = {
        "script": Path(__file__).name,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "tables_generated": [
            "cs1_tier_distribution",
            "cs1_match_type_stratification",
        ],
    }

    with open(args.output_dir / "cs1_tables_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
