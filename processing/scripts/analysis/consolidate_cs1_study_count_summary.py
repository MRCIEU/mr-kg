"""Generate summary statistics table for study count bands (Case Study 1).

This script produces a summary statistics table for direction concordance
stratified by study count band, corresponding to the figure
cs1_fig2_study_count_reproducibility.png.

Input:
    - data/processed/case-study-cs1/metrics/
        pair_reproducibility_metrics.csv

Output:
    - data/artifacts/manuscript-tables/
        cs1_study_count_band_summary.csv
        cs1_study_count_band_summary.tex
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
DEFAULT_OUTPUT_DIR = DATA_DIR / "artifacts" / "manuscript-tables"


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

    # ---- --output-dir ----
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files",
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


def compute_band_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each study count band.

    Args:
        df: DataFrame with study_count_band and mean_direction_concordance

    Returns:
        DataFrame with per-band statistics
    """
    logger.info("Computing summary statistics by study count band...")

    band_order = ["2-3", "4-6", "7-10", "11+"]
    df_clean = df[["study_count_band", "mean_direction_concordance"]].dropna()
    df_clean = df_clean[df_clean["study_count_band"].isin(band_order)]

    # ---- Compute statistics ----
    stats = (
        df_clean.groupby("study_count_band")["mean_direction_concordance"]
        .agg(
            [
                ("n", "count"),
                ("mean", "mean"),
                ("std", "std"),
                ("median", "median"),
                ("q25", lambda x: x.quantile(0.25)),
                ("q75", lambda x: x.quantile(0.75)),
                ("min", "min"),
                ("max", "max"),
            ]
        )
        .reset_index()
    )

    # ---- Reorder by band ----
    band_to_order = {b: i for i, b in enumerate(band_order)}
    stats["band_order"] = stats["study_count_band"].map(band_to_order)
    stats = stats.sort_values("band_order").drop(columns="band_order")

    # ---- Compute standard error ----
    stats["se"] = stats["std"] / np.sqrt(stats["n"])

    # ---- Reorder columns ----
    stats = stats[
        [
            "study_count_band",
            "n",
            "mean",
            "std",
            "se",
            "median",
            "q25",
            "q75",
            "min",
            "max",
        ]
    ]

    logger.info(f"Computed statistics for {len(stats)} bands")
    logger.info(f"Total observations: {stats['n'].sum()}")

    res = stats
    return res


def format_csv(stats_df: pd.DataFrame) -> str:
    """Format statistics as CSV.

    Args:
        stats_df: DataFrame with summary statistics

    Returns:
        CSV string
    """
    # ---- Round values for CSV ----
    output_df = stats_df.copy()
    output_df["mean"] = output_df["mean"].round(3)
    output_df["std"] = output_df["std"].round(3)
    output_df["se"] = output_df["se"].round(3)
    output_df["median"] = output_df["median"].round(3)
    output_df["q25"] = output_df["q25"].round(3)
    output_df["q75"] = output_df["q75"].round(3)
    output_df["min"] = output_df["min"].round(3)
    output_df["max"] = output_df["max"].round(3)

    res = output_df.to_csv(index=False)
    return res


def format_latex(stats_df: pd.DataFrame) -> str:
    """Format statistics as LaTeX table body.

    Args:
        stats_df: DataFrame with summary statistics

    Returns:
        LaTeX table string
    """
    lines = [
        "% Case Study 1: Summary statistics of direction concordance "
        "by study count band",
        "% Generated by consolidate_cs1_study_count_summary.py",
        "% Corresponds to Figure cs1_fig2_study_count_reproducibility.png",
        r"\begin{tabular}{lrrrrrrr}",
        r"  \toprule",
        r"  Study count & n & Mean & SD & SE & Median & IQR & Range \\",
        r"  \midrule",
    ]

    for _, row in stats_df.iterrows():
        band = row["study_count_band"].replace("-", "--")
        n = f"{int(row['n']):,}"
        mean = f"{row['mean']:.2f}"
        std = f"{row['std']:.2f}"
        se = f"{row['se']:.2f}"
        median = f"{row['median']:.2f}"

        # ---- Format IQR ----
        q25 = f"{row['q25']:.2f}"
        q75 = f"{row['q75']:.2f}"
        iqr = f"[{q25}, {q75}]"

        # ---- Format range with proper negative signs ----
        min_val = row["min"]
        max_val = row["max"]
        min_str = f"$-${abs(min_val):.2f}" if min_val < 0 else f"{min_val:.2f}"
        max_str = f"$-${abs(max_val):.2f}" if max_val < 0 else f"{max_val:.2f}"
        range_str = f"[{min_str}, {max_str}]"

        line = (
            f"  {band} & {n} & {mean} & {std} & {se} & "
            f"{median} & {iqr} & {range_str} \\\\"
        )
        lines.append(line)

    lines.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )

    res = "\n".join(lines)
    return res


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # ---- Setup logging ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=" * 60)
    logger.info("GENERATE STUDY COUNT BAND SUMMARY TABLE")
    logger.info("=" * 60)

    logger.info(f"Loading configuration from: {args.config.resolve()}")
    config = load_config(args.config)

    # ---- Resolve paths ----
    metrics_dir = PROJECT_ROOT / config["output"]["case_study_1"]["metrics"]
    input_path = metrics_dir / "pair_reproducibility_metrics.csv"

    if args.dry_run:
        logger.info("[DRY RUN] Would generate study count band summary table")
        logger.info(f"[DRY RUN] Input: {input_path}")
        logger.info(f"[DRY RUN] Output: {args.output_dir}/")
        return

    # ---- Create output directory ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir.resolve()}")

    # ---- Load input data ----
    logger.info(f"Loading metrics from: {input_path.resolve()}")
    metrics_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(metrics_df)} pair metrics")

    # ---- Compute statistics ----
    stats_df = compute_band_statistics(metrics_df)

    # ---- Display summary ----
    logger.info("\nSummary statistics by study count band:")
    for _, row in stats_df.iterrows():
        logger.info(
            f"  {row['study_count_band']:5s}: "
            f"n={int(row['n']):5d}, "
            f"mean={row['mean']:.3f}, "
            f"median={row['median']:.3f}"
        )

    # ---- Save CSV ----
    csv_path = args.output_dir / "cs1_study_count_band_summary.csv"
    csv_content = format_csv(stats_df)
    with open(csv_path, "w") as f:
        f.write(csv_content)
    logger.info(f"Saved CSV: {csv_path.resolve()}")

    # ---- Save LaTeX ----
    tex_path = args.output_dir / "cs1_study_count_band_summary.tex"
    tex_content = format_latex(stats_df)
    with open(tex_path, "w") as f:
        f.write(tex_content)
    logger.info(f"Saved LaTeX: {tex_path.resolve()}")

    logger.info("=" * 60)
    logger.info("Study count band summary table generation complete!")


if __name__ == "__main__":
    main()
