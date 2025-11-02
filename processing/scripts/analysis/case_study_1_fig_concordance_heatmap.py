"""Generate concordance variance heatmap for Case Study 1.

This script creates a heatmap visualization showing concordance variance
across study count bands and reproducibility tiers. This highlights
high-variance clusters that may require manual audit.

Outputs:
- concordance_variance_heatmap.png: Publication-quality figure
- concordance_variance_heatmap.svg: Vector format for editing
- variance_pivot.csv: Pivot table of variance values
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
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


# ==== Data preparation ====


def create_variance_pivot(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create pivot table of concordance variance.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics

    Returns:
        Pivot table with study_count_band as rows, tier as columns
    """
    logger.info("Creating variance pivot table...")

    pivot_data = (
        metrics_df.groupby(["study_count_band", "reproducibility_tier"])[
            "concordance_variance"
        ]
        .mean()
        .reset_index()
    )

    pivot_table = pivot_data.pivot(
        index="study_count_band",
        columns="reproducibility_tier",
        values="concordance_variance",
    )

    tier_order = ["high", "moderate", "low", "discordant"]
    available_tiers = [t for t in tier_order if t in pivot_table.columns]
    pivot_table = pivot_table[available_tiers]

    band_order = ["2-3", "4-6", "7-10", "11+"]
    available_bands = [b for b in band_order if b in pivot_table.index]
    pivot_table = pivot_table.loc[available_bands]

    res = pivot_table
    return res


# ==== Plotting functions ====


def plot_variance_heatmap(
    pivot_table: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create heatmap of concordance variance.

    Args:
        pivot_table: Pivot table with variance values
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating concordance variance heatmap...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["single"]),
        dpi=fig_config["dpi"],
    )

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Concordance Variance"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Reproducibility Tier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Study Count Band", fontsize=12, fontweight="bold")
    ax.set_title(
        "Concordance Variance by Study Count and Reproducibility Tier",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticklabels(
        [label.get_text().capitalize() for label in ax.get_xticklabels()],
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(
        [label.get_text() for label in ax.get_yticklabels()],
        rotation=0,
    )

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"concordance_variance_heatmap.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute concordance variance heatmap generation."""
    args = make_args()

    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    if args.input_csv:
        input_csv = args.input_csv
    else:
        metrics_dir = PROJECT_ROOT / output_config["metrics"]
        input_csv = metrics_dir / "pair_reproducibility_metrics.csv"

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = PROJECT_ROOT / output_config["figures"]

    if args.dry_run:
        logger.info("Dry run - validating configuration and paths")
        logger.info("Input CSV: {}", input_csv)
        logger.info("Output directory: {}", output_dir)

        if not input_csv.exists():
            logger.error("Input CSV not found: {}", input_csv)
            return 1

        logger.info("Dry run complete - configuration validated")
        return 0

    if not input_csv.exists():
        logger.error("Input CSV not found: {}", input_csv)
        logger.error(
            "Please run case_study_1_reproducibility_metrics.py first"
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", output_dir)

    logger.info("Loading pair reproducibility metrics from: {}", input_csv)
    metrics_df = pd.read_csv(input_csv)
    logger.info("Loaded {} trait pairs", len(metrics_df))

    pivot_table = create_variance_pivot(metrics_df)
    logger.info(
        "Created pivot table with shape: {}",
        pivot_table.shape,
    )

    plot_variance_heatmap(pivot_table, output_dir, config)

    variance_file = output_dir / "variance_pivot.csv"
    pivot_table.to_csv(variance_file)
    logger.info("Saved variance pivot table: {}", variance_file)

    metadata = {
        "script": "case_study_1_fig_concordance_heatmap.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_pairs": len(metrics_df),
        "pivot_shape": list(pivot_table.shape),
        "study_count_bands": pivot_table.index.tolist(),
        "reproducibility_tiers": pivot_table.columns.tolist(),
        "variance_summary": {
            "min": float(pivot_table.min().min()),
            "max": float(pivot_table.max().max()),
            "mean": float(pivot_table.mean().mean()),
        },
    }

    metadata_file = output_dir / "variance_heatmap_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Concordance variance heatmap generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
