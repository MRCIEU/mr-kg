"""Ordinal trend test for study count effect on concordance (Case Study 1).

This script implements a formal test for linear trend across ordered study
count categories, as requested by manuscript reviewers. It tests whether
concordance shows a monotonic trend as study count increases.

The approach encodes study count bands as ordinal integers (1, 2, 3, 4) and
tests the linear trend coefficient in an OLS regression model.

Input:
    - data/processed/case-study-cs1/metrics/
        pair_reproducibility_metrics.csv
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs1/models/
        ordinal_trend_test_results.csv
        ordinal_trend_test_diagnostics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from loguru import logger
import statsmodels.api as sm
from scipy import stats

from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # ---- --config ----
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to configuration file",
    )

    res = parser.parse_args()
    return res


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        res = yaml.safe_load(f)
    return res


def create_ordinal_encoding(
    df: pd.DataFrame, band_definitions: List[List[int]]
) -> pd.DataFrame:
    """Create ordinal encoding for study count bands.

    Args:
        df: DataFrame with study_count_band column
        band_definitions: List of [min, max] for each band from config

    Returns:
        DataFrame with study_count_ordinal column added
    """
    logger.info("Creating ordinal encoding for study count bands...")

    # ---- Create band label to ordinal mapping ----
    band_to_ordinal = {}
    for i, (min_val, max_val) in enumerate(band_definitions, start=1):
        if max_val >= 999:
            label = f"{min_val}+"
        else:
            label = f"{min_val}-{max_val}"
        band_to_ordinal[label] = i
        logger.info(f"  Band '{label}' -> ordinal {i}")

    # ---- Apply mapping ----
    df = df.copy()
    df["study_count_ordinal"] = df["study_count_band"].map(band_to_ordinal)

    # ---- Check for unmapped values ----
    unmapped = df["study_count_ordinal"].isna().sum()
    if unmapped > 0:
        unmapped_bands = df[df["study_count_ordinal"].isna()][
            "study_count_band"
        ].unique()
        logger.warning(
            f"Found {unmapped} observations with unmapped bands: {unmapped_bands}"
        )
        df = df.dropna(subset=["study_count_ordinal"])

    logger.info(f"Created ordinal encoding for {len(df)} observations")
    res = df
    return res


def compute_band_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each study count band.

    Args:
        df: DataFrame with study_count_band and mean_direction_concordance

    Returns:
        DataFrame with per-band statistics
    """
    logger.info("Computing per-band statistics...")

    stats_df = (
        df.groupby(["study_count_band", "study_count_ordinal"])
        .agg(
            n=("mean_direction_concordance", "count"),
            mean_concordance=("mean_direction_concordance", "mean"),
            std_concordance=("mean_direction_concordance", "std"),
            median_concordance=("mean_direction_concordance", "median"),
            min_concordance=("mean_direction_concordance", "min"),
            max_concordance=("mean_direction_concordance", "max"),
        )
        .reset_index()
        .sort_values("study_count_ordinal")
    )

    # ---- Compute standard error ----
    stats_df["se_concordance"] = stats_df["std_concordance"] / np.sqrt(
        stats_df["n"]
    )

    logger.info(f"Computed statistics for {len(stats_df)} bands")
    res = stats_df
    return res


def fit_ordinal_trend_model(
    df: pd.DataFrame, include_covariates: bool = True
) -> Dict[str, Any]:
    """Fit OLS model with ordinal study count predictor.

    Args:
        df: DataFrame with study_count_ordinal and covariates
        include_covariates: Whether to include match_type_exact as covariate

    Returns:
        Dictionary with model results
    """
    logger.info("Fitting ordinal trend model...")

    # ---- Prepare predictors ----
    if include_covariates:
        predictor_cols = ["study_count_ordinal", "match_type_exact"]
        model_name = "ordinal_trend_with_covariates"
    else:
        predictor_cols = ["study_count_ordinal"]
        model_name = "ordinal_trend_unadjusted"

    X = df[predictor_cols].copy()

    # ---- Ensure numeric types ----
    X["study_count_ordinal"] = X["study_count_ordinal"].astype(float)
    if "match_type_exact" in X.columns:
        X["match_type_exact"] = X["match_type_exact"].astype(int)

    # ---- Add constant ----
    X = sm.add_constant(X)

    # ---- Prepare outcome ----
    y = df["mean_direction_concordance"]

    # ---- Fit model ----
    model = sm.OLS(y, X)
    results = model.fit()

    logger.info(f"Model: {model_name}")
    logger.info(f"R-squared: {results.rsquared:.4f}")

    # ---- Extract trend coefficient ----
    trend_coef = results.params["study_count_ordinal"]
    trend_se = results.bse["study_count_ordinal"]
    trend_t = results.tvalues["study_count_ordinal"]
    trend_p = results.pvalues["study_count_ordinal"]
    conf_int = results.conf_int(alpha=0.05)
    trend_ci_lower = conf_int.loc["study_count_ordinal", 0]
    trend_ci_upper = conf_int.loc["study_count_ordinal", 1]

    logger.info(f"Trend coefficient: {trend_coef:.4f}")
    logger.info(f"Trend p-value: {trend_p:.4e}")
    logger.info(f"95% CI: [{trend_ci_lower:.4f}, {trend_ci_upper:.4f}]")

    # ---- Interpretation ----
    if trend_p < 0.001:
        significance = "highly significant (p < 0.001)"
    elif trend_p < 0.01:
        significance = "significant (p < 0.01)"
    elif trend_p < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"

    if trend_coef < 0:
        direction = "negative (declining concordance with more studies)"
    else:
        direction = "positive (increasing concordance with more studies)"

    res = {
        "model_name": model_name,
        "trend_coefficient": float(trend_coef),
        "trend_std_error": float(trend_se),
        "trend_t_statistic": float(trend_t),
        "trend_p_value": float(trend_p),
        "trend_ci_lower": float(trend_ci_lower),
        "trend_ci_upper": float(trend_ci_upper),
        "r_squared": float(results.rsquared),
        "adj_r_squared": float(results.rsquared_adj),
        "n_observations": int(len(df)),
        "f_statistic": float(results.fvalue),
        "f_pvalue": float(results.f_pvalue),
        "significance": significance,
        "direction": direction,
        "full_summary": str(results.summary()),
        "all_coefficients": {
            var: {
                "coefficient": float(results.params[var]),
                "std_error": float(results.bse[var]),
                "t_statistic": float(results.tvalues[var]),
                "p_value": float(results.pvalues[var]),
                "ci_lower": float(conf_int.loc[var, 0]),
                "ci_upper": float(conf_int.loc[var, 1]),
            }
            for var in results.params.index
        },
    }
    return res


def perform_spearman_correlation(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute Spearman correlation as non-parametric trend test.

    Args:
        df: DataFrame with study_count_ordinal and mean_direction_concordance

    Returns:
        Dictionary with correlation results
    """
    logger.info(
        "Computing Spearman correlation (non-parametric trend test)..."
    )

    rho, p_value = stats.spearmanr(
        df["study_count_ordinal"], df["mean_direction_concordance"]
    )

    logger.info(f"Spearman rho: {rho:.4f}")
    logger.info(f"Spearman p-value: {p_value:.4e}")

    res = {
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "interpretation": (
            "negative monotonic trend"
            if rho < 0
            else "positive monotonic trend"
        ),
    }
    return res


def perform_kruskal_wallis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform Kruskal-Wallis H-test for differences across bands.

    Args:
        df: DataFrame with study_count_band and mean_direction_concordance

    Returns:
        Dictionary with test results
    """
    logger.info("Performing Kruskal-Wallis H-test...")

    groups = [
        group["mean_direction_concordance"].values
        for _, group in df.groupby("study_count_ordinal")
    ]

    h_stat, p_value = stats.kruskal(*groups)

    logger.info(f"Kruskal-Wallis H: {h_stat:.4f}")
    logger.info(f"Kruskal-Wallis p-value: {p_value:.4e}")

    res = {
        "kruskal_wallis_h": float(h_stat),
        "kruskal_wallis_p_value": float(p_value),
        "n_groups": len(groups),
    }
    return res


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # ---- Setup logging ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info("=" * 60)
    logger.info("ORDINAL TREND TEST FOR STUDY COUNT EFFECT ON CONCORDANCE")
    logger.info("=" * 60)

    logger.info(f"Loading configuration from: {args.config.resolve()}")
    config = load_config(args.config)

    # ---- Resolve paths ----
    output_base = PROJECT_ROOT / config["output"]["case_study_1"]["models"]
    metrics_dir = PROJECT_ROOT / config["output"]["case_study_1"]["metrics"]

    if args.dry_run:
        logger.info("[DRY RUN] Would perform ordinal trend test analysis")
        logger.info(
            f"[DRY RUN] Input: {metrics_dir / 'pair_reproducibility_metrics.csv'}"
        )
        logger.info(f"[DRY RUN] Output: {output_base}/")
        return

    # ---- Create output directory ----
    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_base.resolve()}")

    # ---- Load input data ----
    input_path = metrics_dir / "pair_reproducibility_metrics.csv"
    logger.info(f"Loading metrics from: {input_path.resolve()}")
    metrics_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(metrics_df)} pair metrics")

    # ---- Check required columns ----
    required_cols = [
        "study_count_band",
        "mean_direction_concordance",
        "has_exact_match",
    ]
    missing_cols = [c for c in required_cols if c not in metrics_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    # ---- Prepare match_type_exact ----
    if "match_type_exact" not in metrics_df.columns:
        metrics_df["match_type_exact"] = metrics_df["has_exact_match"]

    # ---- Create ordinal encoding ----
    band_definitions = config["case_study_1"]["study_count_bands"]
    df = create_ordinal_encoding(metrics_df, band_definitions)

    # ---- Compute band statistics ----
    band_stats = compute_band_statistics(df)
    logger.info("\nPer-band statistics:")
    for _, row in band_stats.iterrows():
        logger.info(
            f"  {row['study_count_band']:6s}: "
            f"n={row['n']:4d}, "
            f"mean={row['mean_concordance']:.3f}, "
            f"std={row['std_concordance']:.3f}"
        )

    # ---- Fit ordinal trend models ----
    logger.info("\n" + "-" * 60)
    logger.info("MODEL 1: Ordinal trend (unadjusted)")
    logger.info("-" * 60)
    unadjusted_results = fit_ordinal_trend_model(df, include_covariates=False)

    logger.info("\n" + "-" * 60)
    logger.info("MODEL 2: Ordinal trend (adjusted for match type)")
    logger.info("-" * 60)
    adjusted_results = fit_ordinal_trend_model(df, include_covariates=True)

    # ---- Non-parametric tests ----
    logger.info("\n" + "-" * 60)
    logger.info("NON-PARAMETRIC TESTS")
    logger.info("-" * 60)
    spearman_results = perform_spearman_correlation(df)
    kruskal_results = perform_kruskal_wallis(df)

    # ---- Compile all results ----
    results_summary = {
        "unadjusted_model": {
            "trend_coefficient": unadjusted_results["trend_coefficient"],
            "trend_std_error": unadjusted_results["trend_std_error"],
            "trend_p_value": unadjusted_results["trend_p_value"],
            "trend_ci_lower": unadjusted_results["trend_ci_lower"],
            "trend_ci_upper": unadjusted_results["trend_ci_upper"],
            "r_squared": unadjusted_results["r_squared"],
            "significance": unadjusted_results["significance"],
            "direction": unadjusted_results["direction"],
        },
        "adjusted_model": {
            "trend_coefficient": adjusted_results["trend_coefficient"],
            "trend_std_error": adjusted_results["trend_std_error"],
            "trend_p_value": adjusted_results["trend_p_value"],
            "trend_ci_lower": adjusted_results["trend_ci_lower"],
            "trend_ci_upper": adjusted_results["trend_ci_upper"],
            "r_squared": adjusted_results["r_squared"],
            "significance": adjusted_results["significance"],
            "direction": adjusted_results["direction"],
            "all_coefficients": adjusted_results["all_coefficients"],
        },
        "spearman_correlation": spearman_results,
        "kruskal_wallis": kruskal_results,
        "band_statistics": band_stats.to_dict(orient="records"),
        "n_observations": int(len(df)),
        "n_bands": int(len(band_stats)),
    }

    # ---- Save results ----
    # ---- Save diagnostics JSON ----
    diag_path = output_base / "ordinal_trend_test_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"\nSaved diagnostics: {diag_path.resolve()}")

    # ---- Save results CSV ----
    results_df = pd.DataFrame(
        [
            {
                "test": "ordinal_trend_unadjusted",
                "coefficient": unadjusted_results["trend_coefficient"],
                "std_error": unadjusted_results["trend_std_error"],
                "t_statistic": unadjusted_results["trend_t_statistic"],
                "p_value": unadjusted_results["trend_p_value"],
                "ci_lower": unadjusted_results["trend_ci_lower"],
                "ci_upper": unadjusted_results["trend_ci_upper"],
                "r_squared": unadjusted_results["r_squared"],
            },
            {
                "test": "ordinal_trend_adjusted",
                "coefficient": adjusted_results["trend_coefficient"],
                "std_error": adjusted_results["trend_std_error"],
                "t_statistic": adjusted_results["trend_t_statistic"],
                "p_value": adjusted_results["trend_p_value"],
                "ci_lower": adjusted_results["trend_ci_lower"],
                "ci_upper": adjusted_results["trend_ci_upper"],
                "r_squared": adjusted_results["r_squared"],
            },
            {
                "test": "spearman_correlation",
                "coefficient": spearman_results["spearman_rho"],
                "std_error": np.nan,
                "t_statistic": np.nan,
                "p_value": spearman_results["spearman_p_value"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "r_squared": np.nan,
            },
            {
                "test": "kruskal_wallis",
                "coefficient": kruskal_results["kruskal_wallis_h"],
                "std_error": np.nan,
                "t_statistic": np.nan,
                "p_value": kruskal_results["kruskal_wallis_p_value"],
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "r_squared": np.nan,
            },
        ]
    )
    results_path = output_base / "ordinal_trend_test_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results CSV: {results_path.resolve()}")

    # ---- Save band statistics ----
    band_stats_path = output_base / "ordinal_trend_band_statistics.csv"
    band_stats.to_csv(band_stats_path, index=False)
    logger.info(f"Saved band statistics: {band_stats_path.resolve()}")

    # ---- Save model summaries ----
    summary_path = output_base / "ordinal_trend_model_summaries.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ORDINAL TREND TEST: UNADJUSTED MODEL\n")
        f.write("=" * 70 + "\n\n")
        f.write(unadjusted_results["full_summary"])
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("ORDINAL TREND TEST: ADJUSTED MODEL (with match type)\n")
        f.write("=" * 70 + "\n\n")
        f.write(adjusted_results["full_summary"])
    logger.info(f"Saved model summaries: {summary_path.resolve()}")

    # ---- Final summary ----
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY OF ORDINAL TREND TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"N observations: {len(df)}")
    logger.info(f"N bands: {len(band_stats)}")
    logger.info("")
    logger.info("Unadjusted ordinal trend:")
    logger.info(
        f"  beta = {unadjusted_results['trend_coefficient']:.4f} "
        f"(SE = {unadjusted_results['trend_std_error']:.4f})"
    )
    logger.info(
        f"  95% CI: [{unadjusted_results['trend_ci_lower']:.4f}, "
        f"{unadjusted_results['trend_ci_upper']:.4f}]"
    )
    logger.info(f"  p = {unadjusted_results['trend_p_value']:.4e}")
    logger.info(f"  {unadjusted_results['significance']}")
    logger.info("")
    logger.info("Adjusted ordinal trend (controlling for match type):")
    logger.info(
        f"  beta = {adjusted_results['trend_coefficient']:.4f} "
        f"(SE = {adjusted_results['trend_std_error']:.4f})"
    )
    logger.info(
        f"  95% CI: [{adjusted_results['trend_ci_lower']:.4f}, "
        f"{adjusted_results['trend_ci_upper']:.4f}]"
    )
    logger.info(f"  p = {adjusted_results['trend_p_value']:.4e}")
    logger.info(f"  {adjusted_results['significance']}")
    logger.info("")
    logger.info("Non-parametric tests:")
    logger.info(
        f"  Spearman rho = {spearman_results['spearman_rho']:.4f}, "
        f"p = {spearman_results['spearman_p_value']:.4e}"
    )
    logger.info(
        f"  Kruskal-Wallis H = {kruskal_results['kruskal_wallis_h']:.4f}, "
        f"p = {kruskal_results['kruskal_wallis_p_value']:.4e}"
    )
    logger.info("=" * 60)
    logger.info("\nOrdinal trend test analysis complete!")


if __name__ == "__main__":
    main()
