"""Generate tier distribution by outcome category for Case Study 1.

This script creates a stacked bar chart showing the distribution of
reproducibility tiers across outcome categories, highlighting
category-specific reproducibility patterns.

Outputs:
- category_tier_distribution.png: Publication-quality figure
- category_tier_distribution.svg: Vector format for editing
- category_tier_distribution_metadata.json: Visualization metadata
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


def plot_category_tier_distribution(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create stacked bar chart of tier percentages by category.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating category tier distribution figure...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["double"]),
        dpi=fig_config["dpi"],
    )

    df_clean = metrics_df[
        (metrics_df["outcome_category"] != "uncategorized")
        & (metrics_df["outcome_category"].notna())
    ].copy()

    tier_counts = (
        df_clean.groupby(["outcome_category", "reproducibility_tier"])
        .size()
        .unstack(fill_value=0)
    )

    tier_order = ["high", "moderate", "low", "discordant"]
    tier_counts = tier_counts.reindex(columns=tier_order, fill_value=0)

    tier_pcts = tier_counts.div(tier_counts.sum(axis=1), axis=0) * 100

    category_order = tier_pcts.sort_values(
        "high", ascending=True
    ).index.tolist()

    tier_pcts = tier_pcts.loc[category_order]
    tier_counts_ordered = tier_counts.loc[category_order]

    colors = {
        "high": "#2ecc71",
        "moderate": "#f39c12",
        "low": "#e74c3c",
        "discordant": "#95a5a6",
    }

    tier_pcts.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[colors[tier] for tier in tier_order],
        edgecolor="black",
        linewidth=0.8,
    )

    for i, category in enumerate(category_order):
        n_total = tier_counts_ordered.loc[category].sum()
        ax.text(
            105,
            i,
            f"n={n_total}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("Outcome Category", fontsize=12)
    ax.set_title(
        f"Reproducibility Tier Distribution by Outcome Category "
        f"(n={len(df_clean):,} pairs)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlim(0, 115)

    ax.legend(
        title="Reproducibility Tier",
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="x")

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"category_tier_distribution.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute category tier distribution figure generation."""
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

    if "outcome_category" not in metrics_df.columns:
        logger.error(
            "outcome_category column not found - "
            "please run case_study_1_reproducibility_metrics.py with "
            "category_analysis enabled"
        )
        return 1

    plot_category_tier_distribution(metrics_df, output_dir, config)

    df_clean = metrics_df[
        (metrics_df["outcome_category"] != "uncategorized")
        & (metrics_df["outcome_category"].notna())
    ]

    category_stats = (
        df_clean.groupby("outcome_category")
        .agg(
            n=("outcome_category", "count"),
            pct_high=(
                "reproducibility_tier",
                lambda x: 100 * (x == "high").sum() / len(x),
            ),
            mean_concordance=("mean_direction_concordance", "mean"),
        )
        .to_dict("index")
    )

    metadata = {
        "script": "case_study_1_fig_category_tier_distribution.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_pairs": len(df_clean),
        "n_categories": len(category_stats),
        "category_statistics": {
            k: {
                "n": int(v["n"]),
                "pct_high": float(v["pct_high"]),
                "mean_concordance": float(v["mean_concordance"]),
            }
            for k, v in category_stats.items()
        },
    }

    metadata_file = output_dir / "category_tier_distribution_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Category tier distribution figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
