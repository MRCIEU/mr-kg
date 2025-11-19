"""Generate stacked histograms by study count band for Case Study 1.

This script creates a 2x2 grid of histograms showing the distribution
of direction concordance within each study count band, with mean and
threshold lines for reference.

Outputs:
- study_count_stacked_histograms.png: Publication-quality figure
- study_count_stacked_histograms.svg: Vector format for editing
- study_count_stacked_histograms_metadata.json: Visualization metadata
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
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


def plot_study_count_stacked_histograms(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create 2x2 grid of histograms by study count band.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating study count stacked histograms...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=tuple(fig_config["figsize"]["large"]),
        dpi=fig_config["dpi"],
    )

    df_clean = metrics_df[
        ["study_count_band", "mean_direction_concordance"]
    ].dropna()

    band_order = ["2-3", "4-6", "7-10", "11+"]
    band_titles = [
        "Small Sample (2-3 studies)",
        "Medium Sample (4-6 studies)",
        "Large Sample (7-10 studies)",
        "Very Large Sample (11+ studies)",
    ]

    bins = 20
    hist_range = (-1, 1)

    for idx, (band, title) in enumerate(zip(band_order, band_titles)):
        ax = axes[idx // 2, idx % 2]

        band_data = df_clean[df_clean["study_count_band"] == band][
            "mean_direction_concordance"
        ]

        if len(band_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(title, fontsize=11, fontweight="bold")
            continue

        ax.hist(
            band_data,
            bins=bins,
            range=hist_range,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.8,
        )

        band_mean = band_data.mean()
        ax.axvline(
            band_mean,
            color="red",
            linestyle="-",
            linewidth=2.5,
            label=f"Mean = {band_mean:.2f}",
        )

        ax.axvline(
            0.7,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="High (0.7)",
        )

        ax.axvline(
            0.0,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.5,
            label="Discordant (0.0)",
        )

        ax.set_xlabel("Direction Concordance", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(
            f"{title}\n(n={len(band_data):,})",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlim(-1.05, 1.05)

        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")

    fig.suptitle(
        f"Concordance Distributions by Study Count Band (n={len(df_clean):,} total)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"study_count_stacked_histograms.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute study count stacked histograms generation."""
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

    plot_study_count_stacked_histograms(metrics_df, output_dir, config)

    df_clean = metrics_df[
        ["study_count_band", "mean_direction_concordance"]
    ].dropna()

    band_order = ["2-3", "4-6", "7-10", "11+"]
    band_stats = {}

    for band in band_order:
        band_data = df_clean[df_clean["study_count_band"] == band][
            "mean_direction_concordance"
        ]
        band_stats[band] = {
            "n": len(band_data),
            "mean": float(band_data.mean()),
            "median": float(band_data.median()),
            "std": float(band_data.std()),
            "min": float(band_data.min()),
            "max": float(band_data.max()),
        }

    metadata = {
        "script": "case_study_1_fig_study_count_stacked_histograms.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_pairs": len(df_clean),
        "histogram_bins": 20,
        "histogram_range": [-1, 1],
        "band_statistics": band_stats,
    }

    metadata_file = output_dir / "study_count_stacked_histograms_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Study count stacked histograms generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
