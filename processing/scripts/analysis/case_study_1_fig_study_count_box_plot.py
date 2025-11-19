"""Generate study count box plot for Case Study 1.

This script creates a box plot showing concordance distributions across
study count bands, annotated with sample sizes and high tier percentages,
and color-coded by mean reproducibility tier.

Outputs:
- study_count_box_plot.png: Publication-quality figure
- study_count_box_plot.svg: Vector format for editing
- study_count_box_plot_metadata.json: Visualization metadata
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# ==== Plotting functions ====


def plot_study_count_box_plot(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create box plot of study count bands vs concordance.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating study count box plot...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["double"]),
        dpi=fig_config["dpi"],
    )

    df_clean = metrics_df[
        [
            "study_count_band",
            "mean_direction_concordance",
            "reproducibility_tier",
        ]
    ].dropna()

    band_order = ["2-3", "4-6", "7-10", "11+"]
    df_clean = df_clean[df_clean["study_count_band"].isin(band_order)]

    box_data = [
        df_clean[df_clean["study_count_band"] == band][
            "mean_direction_concordance"
        ].values
        for band in band_order
    ]

    colors = []
    for band in band_order:
        band_data = df_clean[df_clean["study_count_band"] == band]
        mean_conc = band_data["mean_direction_concordance"].mean()

        if mean_conc >= 0.7:
            colors.append("lightgreen")
        elif mean_conc >= 0.3:
            colors.append("lightyellow")
        elif mean_conc >= 0.0:
            colors.append("lightcoral")
        else:
            colors.append("lightgray")

    bp = ax.boxplot(
        box_data,
        positions=range(len(band_order)),
        widths=0.6,
        patch_artist=True,
        notch=False,
        showmeans=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=8,
        ),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.5)
        whisker.set_linestyle("--")

    for cap in bp["caps"]:
        cap.set_linewidth(1.5)

    for median in bp["medians"]:
        median.set_color("darkblue")
        median.set_linewidth(2)

    for i, band in enumerate(band_order):
        band_data = df_clean[df_clean["study_count_band"] == band]
        n = len(band_data)
        pct_high = (
            100 * (band_data["reproducibility_tier"] == "high").sum() / n
        )

        ax.text(
            i,
            -0.85,
            f"n={n}\n{pct_high:.0f}% high",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8,
            ),
        )

    ax.axhline(
        y=0.7,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="High tier threshold",
    )

    ax.axhline(
        y=0.0,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label="Discordant threshold",
    )

    ax.set_xticks(range(len(band_order)))
    ax.set_xticklabels(band_order)
    ax.set_xlabel("Study Count Band", fontsize=12)
    ax.set_ylabel("Mean Direction Concordance", fontsize=12)
    ax.set_title(
        f"Concordance by Study Count Band (n={len(df_clean):,} pairs)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_ylim(-1.05, 1.05)

    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"study_count_box_plot.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute study count box plot figure generation."""
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

    logger.info("Loading pair metrics from: {}", input_csv)
    metrics_df = pd.read_csv(input_csv)
    logger.info("Loaded {} trait pairs", len(metrics_df))

    plot_study_count_box_plot(metrics_df, output_dir, config)

    df_clean = metrics_df[
        [
            "study_count_band",
            "mean_direction_concordance",
            "reproducibility_tier",
        ]
    ].dropna()

    band_order = ["2-3", "4-6", "7-10", "11+"]
    band_stats = {}

    for band in band_order:
        band_data = df_clean[df_clean["study_count_band"] == band]
        band_stats[band] = {
            "n": len(band_data),
            "mean": float(band_data["mean_direction_concordance"].mean()),
            "median": float(band_data["mean_direction_concordance"].median()),
            "q25": float(
                band_data["mean_direction_concordance"].quantile(0.25)
            ),
            "q75": float(
                band_data["mean_direction_concordance"].quantile(0.75)
            ),
            "pct_high": float(
                100
                * (band_data["reproducibility_tier"] == "high").sum()
                / len(band_data)
            ),
        }

    metadata = {
        "script": "case_study_1_fig_study_count_box_plot.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_pairs": len(df_clean),
        "band_statistics": band_stats,
    }

    metadata_file = output_dir / "study_count_box_plot_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Study count box plot figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
