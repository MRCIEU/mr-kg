#!/usr/bin/env python3

"""
Consolidate Case Study 5 temporal trends analysis into manuscript tables.

Produces LaTeX-formatted tables for:
1. Era-level summary statistics
2. STROBE-MR impact on reporting completeness
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
        default=Path("../data/processed/case-study-cs5"),
        help="Input directory containing CS5 analysis outputs",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/artifacts/manuscript-tables"),
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


def load_era_statistics(input_dir: Path) -> pd.DataFrame:
    """Load era-level statistics from temporal metadata."""
    era_stats_path = input_dir / "temporal" / "era_statistics.csv"
    return pd.read_csv(era_stats_path)


def load_diversity_by_era(input_dir: Path) -> pd.DataFrame:
    """Load trait diversity metrics by era."""
    diversity_path = input_dir / "diversity" / "trait_counts_by_era.csv"
    return pd.read_csv(diversity_path)


def load_strobe_impact(input_dir: Path) -> pd.DataFrame:
    """Load STROBE-MR impact on reporting completeness."""
    strobe_path = input_dir / "completeness" / "strobe_impact_on_reporting.csv"
    return pd.read_csv(strobe_path)


def load_completeness_metadata(input_dir: Path) -> dict:
    """Load completeness metadata containing study counts."""
    metadata_path = input_dir / "completeness" / "completeness_metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)


def create_era_summary_table(
    era_stats: pd.DataFrame, diversity: pd.DataFrame
) -> pd.DataFrame:
    """Create consolidated era summary table."""
    merged = era_stats.merge(
        diversity[["era", "mean_trait_count", "median_trait_count"]],
        on="era",
        how="left",
    )

    summary = merged[
        [
            "era",
            "n_studies",
            "era_start",
            "era_end",
            "mean_trait_count",
            "median_trait_count",
        ]
    ].copy()

    summary.columns = [
        "Era",
        "Studies (n)",
        "Year Start",
        "Year End",
        "Mean Traits/Study",
        "Median Traits/Study",
    ]

    summary["Era"] = summary["Era"].str.replace("_", " ").str.title()

    summary["Mean Traits/Study"] = summary["Mean Traits/Study"].round(2)
    summary["Median Traits/Study"] = summary["Median Traits/Study"].round(1)
    summary["Year Start"] = summary["Year Start"].fillna(0).astype(int)
    summary["Year End"] = summary["Year End"].fillna(0).astype(int)

    summary = summary[summary["Era"] != "Unknown"]

    return summary


def create_strobe_impact_table(strobe_impact: pd.DataFrame) -> pd.DataFrame:
    """Create STROBE-MR impact table for key fields."""
    key_fields = [
        "confidence_interval",
        "p_value",
        "direction",
        "effect_size_or",
        "effect_size_beta",
    ]

    filtered = strobe_impact[strobe_impact["field"].isin(key_fields)].copy()

    table = filtered[
        [
            "field",
            "pre_pct",
            "post_pct",
            "change_pct",
            "chi2_statistic",
            "p_value",
        ]
    ].copy()

    table.columns = [
        "Field",
        "Pre-STROBE (%)",
        "Post-STROBE (%)",
        "Change (pp)",
        "Chi-square",
        "P-value",
    ]

    table["Pre-STROBE (%)"] = table["Pre-STROBE (%)"].round(1)
    table["Post-STROBE (%)"] = table["Post-STROBE (%)"].round(1)
    table["Change (pp)"] = table["Change (pp)"].round(1)

    table["Field"] = (
        table["Field"]
        .str.replace("effect_size_", "")
        .str.replace("_", " ")
        .str.title()
    )

    field_order = [
        "Confidence Interval",
        "P Value",
        "Direction",
        "Or",
        "Beta",
    ]
    table["Field"] = pd.Categorical(
        table["Field"], categories=field_order, ordered=True
    )
    table = table.sort_values("Field")

    table["P-value"] = table["P-value"].apply(lambda x: f"{x:.2e}")

    return table


def create_strobe_study_counts_table(metadata: dict) -> pd.DataFrame:
    """Create supplementary table with study counts by period.

    Args:
        metadata: Completeness metadata containing study count information

    Returns:
        DataFrame with period, year range, study counts, and percentages
    """
    strobe_analysis = metadata["strobe_impact_analysis"]

    pre_studies = strobe_analysis["pre_strobe"]["n_studies"]
    post_studies = strobe_analysis["post_strobe"]["n_studies"]
    total_studies = pre_studies + post_studies

    data = {
        "Period": ["Pre-STROBE", "Post-STROBE"],
        "Year Range": ["2003-2020", "2021-2025"],
        "n_studies": [pre_studies, post_studies],
        "Percent of Total": [
            (pre_studies / total_studies) * 100,
            (post_studies / total_studies) * 100,
        ],
    }

    table = pd.DataFrame(data)
    table["Percent of Total"] = table["Percent of Total"].round(1)

    return table


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
    era_stats = load_era_statistics(args.input_dir)
    diversity = load_diversity_by_era(args.input_dir)
    strobe_impact = load_strobe_impact(args.input_dir)
    completeness_metadata = load_completeness_metadata(args.input_dir)

    print("Creating era summary table...")
    era_table = create_era_summary_table(era_stats, diversity)

    print("Creating STROBE impact table...")
    strobe_table = create_strobe_impact_table(strobe_impact)

    print("Creating STROBE study counts supplementary table...")
    study_counts_table = create_strobe_study_counts_table(
        completeness_metadata
    )

    print("Saving CSV tables...")
    era_table.to_csv(args.output_dir / "cs5_era_summary.csv", index=False)
    strobe_table.to_csv(args.output_dir / "cs5_strobe_impact.csv", index=False)
    study_counts_table.to_csv(
        args.output_dir / "cs5_strobe_study_counts.csv", index=False
    )

    print("Generating LaTeX tables...")
    era_latex = format_latex_table(
        era_table,
        caption=(
            "Temporal evolution of MR research across methodological eras."
        ),
        label="tab:cs5-era-summary",
    )

    strobe_latex = format_latex_table(
        strobe_table,
        caption=(
            "Impact of STROBE-MR guidelines (2021) on reporting completeness."
        ),
        label="tab:cs5-strobe-impact",
    )

    study_counts_latex = format_latex_table(
        study_counts_table,
        caption=("Study counts by STROBE-MR period (2021 publication)."),
        label="tab:cs5-strobe-study-counts",
    )

    with open(args.output_dir / "cs5_era_summary.tex", "w") as f:
        f.write(era_latex)

    with open(args.output_dir / "cs5_strobe_impact.tex", "w") as f:
        f.write(strobe_latex)

    with open(args.output_dir / "cs5_strobe_study_counts.tex", "w") as f:
        f.write(study_counts_latex)

    print("\nOutput files:")
    print(f"  {args.output_dir / 'cs5_era_summary.csv'}")
    print(f"  {args.output_dir / 'cs5_era_summary.tex'}")
    print(f"  {args.output_dir / 'cs5_strobe_impact.csv'}")
    print(f"  {args.output_dir / 'cs5_strobe_impact.tex'}")
    print(f"  {args.output_dir / 'cs5_strobe_study_counts.csv'}")
    print(f"  {args.output_dir / 'cs5_strobe_study_counts.tex'}")

    metadata = {
        "script": Path(__file__).name,
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "tables_generated": [
            "cs5_era_summary",
            "cs5_strobe_impact",
            "cs5_strobe_study_counts",
        ],
    }

    with open(args.output_dir / "cs5_tables_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
