"""Generate match type stacked bar chart for Case Study 1.

This script creates stacked bar charts showing tier distribution
by match type (exact, fuzzy, EFO) to illustrate the impact of
matching quality on reproducibility.

Outputs:
- match_type_stacked.png: Publication-quality figure
- match_type_stacked.svg: Vector format for editing
- match_type_stacked_summary.csv: Aggregated match type statistics
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


def plot_match_type_stacked(
    match_type_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create stacked bar chart by match type.

    Args:
        match_type_df: DataFrame with match type stratification
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating match type stacked bar chart...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["single"]),
        dpi=fig_config["dpi"],
    )

    tier_cols = ["pct_high", "pct_moderate", "pct_low", "pct_discordant"]
    tier_labels = ["High", "Moderate", "Low", "Discordant"]
    tier_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]

    match_types = match_type_df["match_type"].values
    x_pos = range(len(match_types))

    bottom = [0] * len(match_types)

    for tier_col, tier_label, color in zip(
        tier_cols,
        tier_labels,
        tier_colors,
    ):
        values = match_type_df[tier_col].values

        bars = ax.bar(
            x_pos,
            values,
            bottom=bottom,
            label=tier_label,
            color=color,
            edgecolor="white",
            linewidth=1.5,
        )

        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 5:
                height = bar.get_height()
                y_pos = bar.get_y() + height / 2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white" if tier_label != "Discordant" else "black",
                )

        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [mt.capitalize() for mt in match_types],
        fontsize=11,
    )
    ax.set_ylabel("Percentage of Pairs (%)", fontsize=12)
    ax.set_title(
        "Reproducibility by Match Type",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        title="Tier",
    )

    for i, (match_type, n_pairs) in enumerate(
        zip(match_type_df["match_type"], match_type_df["n_pairs"])
    ):
        ax.text(
            i,
            -8,
            f"n={n_pairs:,}",
            ha="center",
            va="top",
            fontsize=9,
            color="gray",
        )

    ax.set_ylim(0, 110)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"match_type_stacked.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute match type stacked bar figure generation."""
    args = make_args()

    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    if args.input_csv:
        input_csv = args.input_csv
    else:
        metrics_dir = PROJECT_ROOT / output_config["metrics"]
        input_csv = metrics_dir / "stratified_by_match_type.csv"

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

    logger.info("Loading match type data from: {}", input_csv)
    match_type_df = pd.read_csv(input_csv)
    logger.info("Loaded {} match types", len(match_type_df))

    plot_match_type_stacked(match_type_df, output_dir, config)

    summary_file = output_dir / "match_type_stacked_summary.csv"
    match_type_df.to_csv(summary_file, index=False)
    logger.info("Saved summary data: {}", summary_file)

    metadata = {
        "script": "case_study_1_fig_match_type_stacked.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_match_types": len(match_type_df),
    }

    metadata_file = output_dir / "match_type_stacked_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Match type stacked bar figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
