"""Generate temporal era comparison figure for Case Study 1.

This script creates a grouped bar chart comparing reproducibility tier
percentages between early and recent temporal eras, illustrating
temporal shifts and sample imbalance.

Outputs:
- temporal_era_bars.png: Publication-quality figure
- temporal_era_bars.svg: Vector format for editing
- temporal_era_summary.csv: Aggregated era statistics
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


def plot_temporal_era_comparison(
    era_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create grouped bar chart comparing tier percentages by era.

    Args:
        era_df: DataFrame with temporal era stratification data
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating temporal era comparison chart...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["double"]),
        dpi=fig_config["dpi"],
    )

    era_df_clean = era_df[era_df["temporal_era"] != "other"].copy()

    if len(era_df_clean) == 0:
        logger.warning("No valid era data after filtering 'other'")
        return

    tier_cols = ["pct_high", "pct_moderate", "pct_low", "pct_discordant"]
    tier_labels = ["High", "Moderate", "Low", "Discordant"]
    tier_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]

    x = np.arange(len(tier_labels))
    width = 0.35

    eras = era_df_clean["temporal_era"].unique()

    if "early" in eras and "recent" in eras:
        early_data = era_df_clean[
            era_df_clean["temporal_era"] == "early"
        ].iloc[0]
        recent_data = era_df_clean[
            era_df_clean["temporal_era"] == "recent"
        ].iloc[0]

        early_pcts = [early_data[col] for col in tier_cols]
        recent_pcts = [recent_data[col] for col in tier_cols]
        early_n = early_data["n_pairs"]
        recent_n = recent_data["n_pairs"]

        bars1 = ax.bar(
            x - width / 2,
            early_pcts,
            width,
            label=f"Early (n={early_n:,})",
            color=tier_colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        bars2 = ax.bar(
            x + width / 2,
            recent_pcts,
            width,
            label=f"Recent (n={recent_n:,})",
            color=tier_colors,
            alpha=0.95,
            edgecolor="black",
            linewidth=0.5,
        )

        for bars, pcts in [(bars1, early_pcts), (bars2, recent_pcts)]:
            for bar, pct in zip(bars, pcts):
                height = bar.get_height()
                if height > 5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height / 2,
                        f"{pct:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white" if pct > 20 else "black",
                    )

        ax.set_xlabel("Reproducibility Tier", fontsize=12)
        ax.set_ylabel("Percentage of Pairs (%)", fontsize=12)
        ax.set_title(
            "Reproducibility Tier Distribution by Temporal Era",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels)
        ax.legend(loc="upper right", frameon=True, fancybox=True)
        ax.set_ylim(0, max(max(early_pcts), max(recent_pcts)) * 1.15)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()

        for fmt in fig_config["format"]:
            output_file = output_dir / f"temporal_era_bars.{fmt}"
            plt.savefig(
                output_file,
                dpi=fig_config["dpi"],
                bbox_inches="tight",
            )
            logger.info(f"Saved figure: {output_file}")

        plt.close()
    else:
        logger.warning("Could not find both early and recent eras in data")


# ==== Main execution ====


def main():
    """Execute temporal era comparison figure generation."""
    args = make_args()

    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    if args.input_csv:
        input_csv = args.input_csv
    else:
        metrics_dir = PROJECT_ROOT / output_config["metrics"]
        input_csv = metrics_dir / "stratified_by_temporal_era.csv"

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

    logger.info("Loading temporal era data from: {}", input_csv)
    era_df = pd.read_csv(input_csv)
    logger.info("Loaded {} era groups", len(era_df))

    plot_temporal_era_comparison(era_df, output_dir, config)

    summary_file = output_dir / "temporal_era_summary.csv"
    era_df.to_csv(summary_file, index=False)
    logger.info("Saved summary data: {}", summary_file)

    metadata = {
        "script": "case_study_1_fig_temporal_era.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "era_groups": len(era_df),
        "temporal_eras": era_df["temporal_era"].tolist(),
    }

    metadata_file = output_dir / "temporal_era_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Temporal era figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
