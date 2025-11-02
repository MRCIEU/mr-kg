"""Generate regression diagnostics figure for Case Study 1.

This script creates a multi-panel diagnostic plot with three subplots:
1. Residuals vs fitted values
2. QQ plot for normality assessment
3. Autocorrelation function (ACF) for lag 1-20

These diagnostics evaluate the temporal trend model and identify
potential violations of linear regression assumptions.

Outputs:
- temporal_model_diagnostics.png: Publication-quality figure
- temporal_model_diagnostics.svg: Vector format for editing
- diagnostics_summary.json: Key diagnostic statistics
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
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

    # ---- --predictions-csv ----
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        help="Override predictions CSV path from config",
    )

    # ---- --diagnostics-json ----
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        help="Override diagnostics JSON path from config",
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


# ==== Diagnostic utility functions ====


def compute_acf(residuals: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute autocorrelation function for residuals.

    Args:
        residuals: Array of residual values
        max_lag: Maximum lag to compute (default: 20)

    Returns:
        Array of autocorrelation values for lags 1 to max_lag
    """
    n = len(residuals)
    mean = np.mean(residuals)
    c0 = np.sum((residuals - mean) ** 2) / n

    acf_values = []
    for lag in range(1, max_lag + 1):
        c_lag = (
            np.sum((residuals[:-lag] - mean) * (residuals[lag:] - mean)) / n
        )
        acf_values.append(c_lag / c0)

    res = np.array(acf_values)
    return res


# ==== Plotting functions ====


def plot_regression_diagnostics(
    predictions_df: pd.DataFrame,
    diagnostics_dict: Dict,
    output_dir: Path,
    config: Dict,
):
    """Create three-panel regression diagnostic figure.

    Args:
        predictions_df: DataFrame with observed, predicted, residual
        diagnostics_dict: Dictionary with model diagnostics
        output_dir: Directory to save figure outputs
        config: Configuration dictionary
    """
    logger.info("Creating regression diagnostics figure...")

    fig_config = config["figures"]
    plt.style.use(fig_config["style"])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=tuple(fig_config["figsize"]["large"]),
        dpi=fig_config["dpi"],
    )

    residuals = predictions_df["residual"].values
    predicted = predictions_df["predicted"].values
    n_obs = len(residuals)

    ax1, ax2, ax3 = axes

    ax1.scatter(
        predicted,
        residuals,
        alpha=0.5,
        s=20,
        edgecolor="black",
        linewidth=0.3,
    )
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=2)

    lowess_result = np.polyfit(predicted, residuals, 2)
    x_smooth = np.linspace(predicted.min(), predicted.max(), 100)
    y_smooth = np.polyval(lowess_result, x_smooth)
    ax1.plot(x_smooth, y_smooth, color="blue", linewidth=2, alpha=0.7)

    ax1.set_xlabel("Fitted Values", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("(a) Residuals vs Fitted", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")

    standardized_residuals = (residuals - residuals.mean()) / residuals.std()
    theoretical_quantiles = stats.norm.ppf(
        (np.arange(1, n_obs + 1) - 0.5) / n_obs
    )
    sorted_residuals = np.sort(standardized_residuals)

    ax2.scatter(
        theoretical_quantiles,
        sorted_residuals,
        alpha=0.6,
        s=20,
        edgecolor="black",
        linewidth=0.3,
    )

    ax2.plot(
        theoretical_quantiles,
        theoretical_quantiles,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Normal distribution",
    )

    ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax2.set_ylabel("Sample Quantiles", fontsize=11)
    ax2.set_title("(b) Normal Q-Q Plot", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3, linestyle="--")

    acf_values = compute_acf(residuals, max_lag=20)
    lags = np.arange(1, 21)

    confidence_bound = 1.96 / np.sqrt(n_obs)

    ax3.bar(
        lags,
        acf_values,
        width=0.6,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )

    ax3.axhline(
        y=confidence_bound,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"95% CI: +/-{confidence_bound:.3f}",
    )
    ax3.axhline(
        y=-confidence_bound, color="red", linestyle="--", linewidth=1.5
    )
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1)

    ax3.set_xlabel("Lag", fontsize=11)
    ax3.set_ylabel("Autocorrelation", fontsize=11)
    ax3.set_title(
        "(c) Autocorrelation Function",
        fontsize=12,
        fontweight="bold",
    )
    ax3.legend(loc="upper right", fontsize=9)
    ax3.set_xticks(lags[::2])
    ax3.set_ylim(-0.5, 1.0)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    durbin_watson = diagnostics_dict.get("durbin_watson", "N/A")
    if isinstance(durbin_watson, float):
        fig.text(
            0.5,
            0.02,
            f"Durbin-Watson: {durbin_watson:.3f} | "
            f"R-squared: {diagnostics_dict['r_squared']:.3f} | "
            f"n={n_obs:,}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    for fmt in fig_config["format"]:
        output_file = output_dir / f"temporal_model_diagnostics.{fmt}"
        plt.savefig(output_file, dpi=fig_config["dpi"], bbox_inches="tight")
        logger.info(f"Saved figure: {output_file}")

    plt.close()


# ==== Main execution ====


def main():
    """Execute regression diagnostics figure generation."""
    args = make_args()

    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    if args.predictions_csv:
        predictions_csv = args.predictions_csv
    else:
        models_dir = PROJECT_ROOT / output_config["models"]
        predictions_csv = models_dir / "temporal_predictions.csv"

    if args.diagnostics_json:
        diagnostics_json = args.diagnostics_json
    else:
        models_dir = PROJECT_ROOT / output_config["models"]
        diagnostics_json = models_dir / "temporal_model_diagnostics.json"

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = PROJECT_ROOT / output_config["figures"]

    if args.dry_run:
        logger.info("Dry run - validating configuration and paths")
        logger.info("Predictions CSV: {}", predictions_csv)
        logger.info("Diagnostics JSON: {}", diagnostics_json)
        logger.info("Output directory: {}", output_dir)

        if not predictions_csv.exists():
            logger.error("Predictions CSV not found: {}", predictions_csv)
            return 1

        if not diagnostics_json.exists():
            logger.error("Diagnostics JSON not found: {}", diagnostics_json)
            return 1

        logger.info("Dry run complete - configuration validated")
        return 0

    if not predictions_csv.exists():
        logger.error("Predictions CSV not found: {}", predictions_csv)
        logger.error("Please run case_study_1_temporal_model.py first")
        return 1

    if not diagnostics_json.exists():
        logger.error("Diagnostics JSON not found: {}", diagnostics_json)
        logger.error("Please run case_study_1_temporal_model.py first")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", output_dir)

    logger.info("Loading predictions from: {}", predictions_csv)
    predictions_df = pd.read_csv(predictions_csv)
    logger.info("Loaded {} observations", len(predictions_df))

    logger.info("Loading diagnostics from: {}", diagnostics_json)
    with diagnostics_json.open("r") as f:
        diagnostics_dict = json.load(f)

    plot_regression_diagnostics(
        predictions_df,
        diagnostics_dict,
        output_dir,
        config,
    )

    summary_stats = {
        "script": "case_study_1_fig_regression_diagnostics.py",
        "predictions_csv": str(predictions_csv),
        "diagnostics_json": str(diagnostics_json),
        "output_dir": str(output_dir),
        "n_observations": len(predictions_df),
        "residual_mean": float(predictions_df["residual"].mean()),
        "residual_std": float(predictions_df["residual"].std()),
        "r_squared": diagnostics_dict["r_squared"],
        "durbin_watson": diagnostics_dict["durbin_watson"],
    }

    summary_file = output_dir / "diagnostics_summary.json"
    with summary_file.open("w") as f:
        json.dump(summary_stats, f, indent=2)
    logger.info("Saved summary statistics: {}", summary_file)

    logger.info("Regression diagnostics figure generation complete!")

    return 0


if __name__ == "__main__":
    exit(main())
