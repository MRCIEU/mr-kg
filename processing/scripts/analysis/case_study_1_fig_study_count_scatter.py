"""Generate study count vs concordance scatter plot for Case Study 1.

This script creates a scatter plot with LOWESS curve showing the
relationship between study count and direction concordance, highlighting
the negative association and tail behavior beyond 10 studies.

Outputs:
- study_count_scatter.png: Publication-quality figure
- study_count_scatter.svg: Vector format for editing
- study_count_scatter_summary.csv: Summary statistics by study count band
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
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
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


def plot_study_count_scatter(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create scatter plot of study count vs concordance with LOWESS.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating study count scatter plot...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["double"]),
        dpi=fig_config["dpi"],
    )

    df_clean = metrics_df[
        ["study_count", "mean_direction_concordance"]
    ].dropna()

    jitter_x = np.random.normal(0, 0.2, len(df_clean))
    x_jittered = df_clean["study_count"] + jitter_x

    scatter = ax.scatter(
        x_jittered,
        df_clean["mean_direction_concordance"],
        alpha=0.3,
        s=20,
        c=df_clean["study_count"],
        cmap="viridis",
        edgecolors="none",
    )

    lowess_result = lowess(
        df_clean["mean_direction_concordance"],
        df_clean["study_count"],
        frac=0.2,
    )

    ax.plot(
        lowess_result[:, 0],
        lowess_result[:, 1],
        color="red",
        linewidth=2.5,
        label="LOWESS curve",
        zorder=10,
    )

    ax.axhline(
        y=0.0,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Discordant threshold",
    )

    ax.axvline(
        x=10,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="10 studies threshold",
    )

    rho, pval = spearmanr(
        df_clean["study_count"],
        df_clean["mean_direction_concordance"],
    )

    ax.set_xlabel("Number of Studies", fontsize=12)
    ax.set_ylabel("Mean Direction Concordance", fontsize=12)
    ax.set_title(
        f"Study Count vs Reproducibility (n={len(df_clean):,} pairs)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.text(
        0.98,
        0.98,
        f"Spearman rho = {rho:.3f}\np < 0.001"
        if pval < 0.001
        else (f"Spearman rho = {rho:.3f}\np = {pval:.3f}"),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax, label="Study Count")
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"study_count_scatter.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


def compute_summary_stats(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics by study count band.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics

    Returns:
        DataFrame with summary statistics
    """
    logger.info("Computing summary statistics...")

    res = (
        metrics_df.groupby("study_count_band")
        .agg(
            n_pairs=("study_count", "count"),
            mean_study_count=("study_count", "mean"),
            mean_concordance=("mean_direction_concordance", "mean"),
            median_concordance=("mean_direction_concordance", "median"),
            std_concordance=("mean_direction_concordance", "std"),
            min_concordance=("mean_direction_concordance", "min"),
            max_concordance=("mean_direction_concordance", "max"),
        )
        .reset_index()
    )

    return res


# ==== Main execution ====


def main():
    """Execute study count scatter figure generation."""
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

    plot_study_count_scatter(metrics_df, output_dir, config)

    summary_df = compute_summary_stats(metrics_df)
    summary_file = output_dir / "study_count_scatter_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info("Saved summary data: {}", summary_file)

    df_clean = metrics_df[
        ["study_count", "mean_direction_concordance"]
    ].dropna()
    rho, pval = spearmanr(
        df_clean["study_count"],
        df_clean["mean_direction_concordance"],
    )

    metadata = {
        "script": "case_study_1_fig_study_count_scatter.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_pairs": len(df_clean),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
    }

    metadata_file = output_dir / "study_count_scatter_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Study count scatter figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
