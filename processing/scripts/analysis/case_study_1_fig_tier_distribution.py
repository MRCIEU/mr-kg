"""Generate tier distribution bar chart for Case Study 1.

This script creates a 100% stacked bar chart showing the proportion
of trait pairs across reproducibility tiers (high, moderate, low,
discordant) with annotated counts.

Outputs:
- tier_distribution_bar.png: Publication-quality figure
- tier_distribution_bar.svg: Vector format for editing
- tier_distribution_summary.csv: Aggregated tier statistics
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


def plot_tier_distribution(
    tier_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create 100% stacked bar chart of tier distribution.

    Args:
        tier_df: DataFrame with tier, count, and percentage columns
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating tier distribution bar chart...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["single"]),
        dpi=fig_config["dpi"],
    )

    tier_colors = {
        "high": "#2ecc71",
        "moderate": "#f39c12",
        "low": "#e74c3c",
        "discordant": "#95a5a6",
    }

    total_pairs = tier_df["count"].sum()

    tier_df = tier_df.set_index("tier")
    tier_order = ["high", "moderate", "low", "discordant"]
    tier_df = tier_df.reindex(tier_order)

    colors = [tier_colors[tier] for tier in tier_order]

    ax.barh(
        [0],
        tier_df["percentage"],
        left=tier_df["percentage"].shift(fill_value=0).cumsum(),
        color=colors,
        edgecolor="white",
        linewidth=2,
    )

    cumsum = 0
    for idx, (tier, row) in enumerate(tier_df.iterrows()):
        pct = row["percentage"]
        count = row["count"]
        center = cumsum + pct / 2
        cumsum += pct

        label = f"{tier.capitalize()}\n{count:,} ({pct:.1f}%)"

        ax.text(
            center,
            0,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white" if tier != "discordant" else "black",
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Percentage of Trait Pairs (%)", fontsize=12)
    ax.set_yticks([])
    ax.set_title(
        f"Reproducibility Tier Distribution (n={total_pairs:,} pairs)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"tier_distribution_bar.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute tier distribution figure generation."""
    args = make_args()

    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    if args.input_csv:
        input_csv = args.input_csv
    else:
        metrics_dir = PROJECT_ROOT / output_config["metrics"]
        input_csv = metrics_dir / "tier_distribution.csv"

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

    logger.info("Loading tier distribution data from: {}", input_csv)
    tier_df = pd.read_csv(input_csv)
    logger.info("Loaded {} tier categories", len(tier_df))

    plot_tier_distribution(tier_df, output_dir, config)

    summary_file = output_dir / "tier_distribution_summary.csv"
    tier_df.to_csv(summary_file, index=False)
    logger.info("Saved summary data: {}", summary_file)

    metadata = {
        "script": "case_study_1_fig_tier_distribution.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "total_pairs": int(tier_df["count"].sum()),
        "tier_counts": tier_df.set_index("tier")["count"].to_dict(),
    }

    metadata_file = output_dir / "tier_distribution_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Tier distribution figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
