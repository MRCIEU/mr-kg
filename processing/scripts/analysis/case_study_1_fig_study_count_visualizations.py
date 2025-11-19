"""Generate study count visualizations for Case Study 1.

This script creates three types of plots showing the relationship between
study count and reproducibility (direction concordance):
- Violin plot: Shows distribution with strip overlay
- Box plot: Shows quartiles with color-coded tiers
- Stacked histograms: Shows 2x2 grid of histograms by study count band

Outputs:
- study_count_violin.png/svg: Violin plot figure
- study_count_box_plot.png/svg: Box plot figure
- study_count_stacked_histograms.png/svg: Stacked histograms figure
- *_metadata.json: Metadata for each generated figure
"""

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


def plot_violin(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
) -> Dict:
    """Create violin plot of study count bands vs concordance.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary

    Returns:
        Metadata dictionary for the generated plot
    """
    logger.info("Creating study count violin plot...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, ax = plt.subplots(
        figsize=tuple(fig_config["figsize"]["double"]),
        dpi=fig_config["dpi"],
    )

    df_clean = metrics_df[
        ["study_count_band", "mean_direction_concordance"]
    ].dropna()

    band_order = ["2-3", "4-6", "7-10", "11+"]
    df_clean = df_clean[df_clean["study_count_band"].isin(band_order)]

    violin_parts = ax.violinplot(
        [
            df_clean[df_clean["study_count_band"] == band][
                "mean_direction_concordance"
            ].values
            for band in band_order
        ],
        positions=range(len(band_order)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for pc in violin_parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
        pc.set_linewidth(1.5)

    for i, band in enumerate(band_order):
        band_data = df_clean[df_clean["study_count_band"] == band][
            "mean_direction_concordance"
        ]

        jitter_x = np.random.normal(i, 0.04, len(band_data))
        ax.scatter(
            jitter_x,
            band_data,
            alpha=0.2,
            s=15,
            color="gray",
            edgecolors="none",
            zorder=1,
        )

        band_mean = band_data.mean()
        ax.scatter(
            i,
            band_mean,
            color="red",
            marker="D",
            s=100,
            zorder=10,
            edgecolors="darkred",
            linewidths=1.5,
            label="Mean" if i == 0 else None,
        )

    ax.axhline(
        y=0.7,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="High tier threshold (0.7)",
    )

    ax.axhline(
        y=0.0,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label="Discordant threshold (0.0)",
    )

    ax.set_xticks(range(len(band_order)))
    ax.set_xticklabels(band_order)
    ax.set_xlabel("Study Count Band", fontsize=12)
    ax.set_ylabel("Mean Direction Concordance", fontsize=12)
    ax.set_title(
        f"Concordance Distribution by Study Count (n={len(df_clean):,} pairs)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_ylim(-1.05, 1.05)

    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, axis="y")

    plt.tight_layout()

    for fmt in fig_config["format"]:
        output_file = output_dir / f"study_count_violin.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()

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
        }

    res = {
        "plot_type": "violin",
        "n_pairs": len(df_clean),
        "band_statistics": band_stats,
    }

    return res


def plot_box(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
) -> Dict:
    """Create box plot of study count bands vs concordance.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary

    Returns:
        Metadata dictionary for the generated plot
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

    res = {
        "plot_type": "box",
        "n_pairs": len(df_clean),
        "band_statistics": band_stats,
    }

    return res


def plot_histograms(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    config: Dict,
) -> Dict:
    """Create 2x2 grid of histograms by study count band.

    Args:
        metrics_df: DataFrame with pair reproducibility metrics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary

    Returns:
        Metadata dictionary for the generated plot
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

    res = {
        "plot_type": "stacked_histograms",
        "n_pairs": len(df_clean),
        "histogram_bins": bins,
        "histogram_range": list(hist_range),
        "band_statistics": band_stats,
    }

    return res


# ==== Main execution ====


def main() -> int:
    """Main entry point for study count visualizations generation.

    Generates all three plot types (violin, box, histograms) every time.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info("Loading configuration from: {}", DEFAULT_CONFIG)
    config = load_config(DEFAULT_CONFIG)

    output_config = config["output"]["case_study_1"]

    metrics_dir = PROJECT_ROOT / output_config["metrics"]
    input_csv = metrics_dir / "pair_reproducibility_metrics.csv"

    output_dir = PROJECT_ROOT / output_config["figures"]

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

    logger.info("Generating all three plot types...")

    # ---- Generate violin plot ----
    metadata_violin = plot_violin(metrics_df, output_dir, config)
    metadata_file = output_dir / "study_count_violin_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(
            {
                "script": "case_study_1_fig_study_count_visualizations.py",
                "input_csv": str(input_csv),
                "output_dir": str(output_dir),
                **metadata_violin,
            },
            f,
            indent=2,
        )
    logger.info("Generated violin plot")

    # ---- Generate box plot ----
    metadata_box = plot_box(metrics_df, output_dir, config)
    metadata_file = output_dir / "study_count_box_plot_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(
            {
                "script": "case_study_1_fig_study_count_visualizations.py",
                "input_csv": str(input_csv),
                "output_dir": str(output_dir),
                **metadata_box,
            },
            f,
            indent=2,
        )
    logger.info("Generated box plot")

    # ---- Generate stacked histograms ----
    metadata_histograms = plot_histograms(metrics_df, output_dir, config)
    metadata_file = output_dir / "study_count_stacked_histograms_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(
            {
                "script": "case_study_1_fig_study_count_visualizations.py",
                "input_csv": str(input_csv),
                "output_dir": str(output_dir),
                **metadata_histograms,
            },
            f,
            indent=2,
        )
    logger.info("Generated stacked histograms")

    logger.info("Study count visualizations generation complete!")
    logger.info("Generated 3 plots: violin, box, histograms")

    return 0


if __name__ == "__main__":
    exit(main())
