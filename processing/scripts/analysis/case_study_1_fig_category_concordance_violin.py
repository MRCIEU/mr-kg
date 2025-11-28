"""Generate concordance distribution by outcome category for Case Study 1.

This script creates a violin plot showing the distribution of direction
concordance across outcome categories, with overlaid mean markers and
sample sizes.

Outputs:
- category_concordance_violin.png: Publication-quality figure
- category_concordance_violin.svg: Vector format for editing
- category_concordance_violin_metadata.json: Visualization metadata
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


def plot_category_concordance_violin(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
):
    """Create violin plot of concordance distributions by category.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating category concordance violin plot...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["double"]),
        dpi=fig_config["dpi"],
    )

    df_clean = metrics_df[
        (metrics_df["outcome_category"] != "uncategorized")
        & (metrics_df["outcome_category"].notna())
        & (metrics_df["mean_direction_concordance"].notna())
    ].copy()

    category_order = (
        df_clean.groupby("outcome_category")["mean_direction_concordance"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    violin_parts = ax.violinplot(
        [
            df_clean[df_clean["outcome_category"] == cat][
                "mean_direction_concordance"
            ].values
            for cat in category_order
        ],
        positions=range(len(category_order)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        vert=False,
    )

    for pc in violin_parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
        pc.set_linewidth(1.5)

    for i, cat in enumerate(category_order):
        cat_data = df_clean[df_clean["outcome_category"] == cat][
            "mean_direction_concordance"
        ]

        cat_mean = cat_data.mean()
        ax.scatter(
            cat_mean,
            i,
            color="red",
            marker="D",
            s=100,
            zorder=10,
            edgecolors="darkred",
            linewidths=1.5,
            label="Mean" if i == 0 else None,
        )

        n_total = len(cat_data)
        pct_high = (
            100
            * (
                df_clean[df_clean["outcome_category"] == cat][
                    "reproducibility_tier"
                ]
                == "high"
            ).sum()
            / n_total
        )

        ax.text(
            -1.15,
            i,
            f"n={n_total}\n{pct_high:.0f}% high",
            ha="right",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8,
            ),
        )

    ax.axvline(
        0.7,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="High tier (0.7)",
    )

    ax.axvline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label="Discordant (0.0)",
    )

    ax.set_yticks(range(len(category_order)))
    ax.set_yticklabels(category_order)
    ax.set_ylabel("Outcome Category", fontsize=12)
    ax.set_xlabel("Mean Direction Concordance", fontsize=12)
    ax.set_title(
        f"Concordance Distribution by Outcome Category "
        f"(n={len(df_clean):,} pairs)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlim(-1.3, 1.05)

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="x")

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"category_concordance_violin.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute category concordance violin figure generation."""
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

    plot_category_concordance_violin(metrics_df, output_dir, config)

    df_clean = metrics_df[
        (metrics_df["outcome_category"] != "uncategorized")
        & (metrics_df["outcome_category"].notna())
        & (metrics_df["mean_direction_concordance"].notna())
    ]

    category_stats = {}
    for cat in df_clean["outcome_category"].unique():
        cat_data = df_clean[df_clean["outcome_category"] == cat]
        category_stats[cat] = {
            "n": len(cat_data),
            "mean": float(cat_data["mean_direction_concordance"].mean()),
            "median": float(cat_data["mean_direction_concordance"].median()),
            "std": float(cat_data["mean_direction_concordance"].std()),
        }

    metadata = {
        "script": "case_study_1_fig_category_concordance_violin.py",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_pairs": len(df_clean),
        "n_categories": len(category_stats),
        "category_statistics": category_stats,
    }

    metadata_file = output_dir / "category_concordance_violin_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata: {}", metadata_file)

    logger.info("Category concordance violin figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
