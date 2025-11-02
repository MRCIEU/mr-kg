"""Temporal trend analysis for Case Study 1 reproducibility metrics.

This script analyzes how reproducibility changes over time by fitting
linear regression models to direction concordance metrics as a function
of publication year and other predictors. It produces model coefficients,
confidence intervals, and diagnostic statistics.

Input:
    - data/processed/case-study-cs1/metrics/
        pair_reproducibility_metrics.csv
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs1/models/
        temporal_model_coefficients.csv
        temporal_model_diagnostics.json
        temporal_predictions.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import yaml
from loguru import logger
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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


def prepare_temporal_data(
    metrics_df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Prepare data for temporal regression modeling.

    Args:
        metrics_df: DataFrame with pair metrics (already includes pub year)
        config: Configuration dictionary

    Returns:
        DataFrame ready for modeling with all required columns
    """
    logger.info("Preparing data for temporal modeling...")

    # ---- Check for required columns ----
    required_cols = ["publication_year", "mean_direction_concordance"]
    missing_cols = [c for c in required_cols if c not in metrics_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    # ---- Filter out missing years ----
    missing_years = metrics_df["publication_year"].isna().sum()
    if missing_years > 0:
        logger.warning(
            f"Dropping {missing_years} pairs with missing publication years"
        )
        metrics_df = metrics_df.dropna(subset=["publication_year"])

    # ---- Prepare predictor columns ----

    # ---- Add match_type_exact indicator ----
    if "match_type_exact" not in metrics_df.columns:
        metrics_df["match_type_exact"] = metrics_df["has_exact_match"]

    logger.info(f"Prepared {len(metrics_df)} pairs for temporal modeling")
    res = metrics_df
    return res


def fit_temporal_model(
    data_df: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Fit linear regression model for temporal trends.

    Args:
        data_df: DataFrame with prepared temporal data
        config: Configuration dictionary with modeling parameters

    Returns:
        Dictionary containing model results and diagnostics
    """
    logger.info("Fitting temporal linear regression model...")

    modeling_config = config["modeling"]["temporal"]
    predictor_names = modeling_config["predictors"]

    # ---- Check minimum observations ----
    min_obs = modeling_config["min_observations"]
    if len(data_df) < min_obs:
        logger.error(
            f"Insufficient data: {len(data_df)} < {min_obs} observations"
        )
        sys.exit(1)

    # ---- Prepare predictors ----
    X = data_df[predictor_names].copy()

    # ---- Handle categorical variables ----
    X["match_type_exact"] = X["match_type_exact"].astype(int)

    # ---- Add constant term ----
    X = sm.add_constant(X)

    # ---- Prepare outcome ----
    y = data_df["mean_direction_concordance"]

    # ---- Fit model ----
    model = sm.OLS(y, X)
    results = model.fit()

    logger.info("Model fitted successfully")
    logger.info(f"R-squared: {results.rsquared:.4f}")
    logger.info(f"Adjusted R-squared: {results.rsquared_adj:.4f}")

    # ---- Extract coefficients with confidence intervals ----
    conf_level = modeling_config["confidence_level"]
    conf_int = results.conf_int(alpha=1 - conf_level)

    coefficients_data = []
    for var in results.params.index:
        coefficients_data.append(
            {
                "variable": var,
                "coefficient": results.params[var],
                "std_error": results.bse[var],
                "t_statistic": results.tvalues[var],
                "p_value": results.pvalues[var],
                "ci_lower": conf_int.loc[var, 0],
                "ci_upper": conf_int.loc[var, 1],
            }
        )

    coefficients_df = pd.DataFrame(coefficients_data)

    # ---- Compute diagnostics ----
    logger.info("Computing model diagnostics...")

    # ---- VIF for multicollinearity ----
    vif_data = []
    X_for_vif = X.drop(columns=["const"])
    for i, col in enumerate(X_for_vif.columns):
        vif_value = variance_inflation_factor(X_for_vif.values, i)
        vif_data.append({"variable": col, "vif": vif_value})
    vif_df = pd.DataFrame(vif_data)

    diagnostics = {
        "n_observations": int(len(data_df)),
        "n_predictors": int(len(predictor_names)),
        "r_squared": float(results.rsquared),
        "adj_r_squared": float(results.rsquared_adj),
        "f_statistic": float(results.fvalue),
        "f_pvalue": float(results.f_pvalue),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "residual_std_error": float(np.sqrt(results.mse_resid)),
        "durbin_watson": float(
            sm.stats.stattools.durbin_watson(results.resid)
        ),
        "vif_max": float(vif_df["vif"].max()),
        "vif_details": vif_df.to_dict(orient="records"),
    }

    logger.info(f"Max VIF: {diagnostics['vif_max']:.2f}")
    logger.info(f"Durbin-Watson: {diagnostics['durbin_watson']:.2f}")

    # ---- Generate predictions ----
    logger.info("Generating model predictions...")
    predictions = results.predict(X)
    residuals = results.resid

    predictions_df = data_df[["study1_pmid", "study1_model", "title"]].copy()
    predictions_df["observed"] = y.values
    predictions_df["predicted"] = predictions
    predictions_df["residual"] = residuals

    res = {
        "coefficients": coefficients_df,
        "diagnostics": diagnostics,
        "predictions": predictions_df,
        "model_summary": str(results.summary()),
    }
    return res


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # ---- Setup logging ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info(f"Loading configuration from: {args.config.resolve()}")
    config = load_config(args.config)

    modeling_config = config["modeling"]["temporal"]

    # ---- Resolve paths ----

    output_base = PROJECT_ROOT / config["output"]["case_study_1"]["models"]
    metrics_dir = PROJECT_ROOT / config["output"]["case_study_1"]["metrics"]

    if args.dry_run:
        logger.info("[DRY RUN] Would perform temporal model analysis")
        logger.info(
            f"[DRY RUN] Input: {metrics_dir / 'pair_reproducibility_metrics.csv'}"
        )
        logger.info(f"[DRY RUN] Output: {output_base}/")
        logger.info(f"[DRY RUN] Predictors: {modeling_config['predictors']}")
        return

    # ---- Create output directory ----
    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_base.resolve()}")

    # ---- Load input data ----
    input_path = metrics_dir / "pair_reproducibility_metrics.csv"
    logger.info(f"Loading metrics from: {input_path.resolve()}")
    metrics_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(metrics_df)} pair metrics")

    # ---- Prepare temporal data ----
    temporal_df = prepare_temporal_data(metrics_df, config)

    # ---- Fit model ----
    model_results = fit_temporal_model(temporal_df, config)

    # ---- Save coefficients ----
    coef_path = output_base / "temporal_model_coefficients.csv"
    model_results["coefficients"].to_csv(coef_path, index=False)
    logger.info(f"Saved model coefficients: {coef_path.resolve()}")

    # ---- Save diagnostics ----
    diag_path = output_base / "temporal_model_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(model_results["diagnostics"], f, indent=2)
    logger.info(f"Saved model diagnostics: {diag_path.resolve()}")

    # ---- Save predictions ----
    pred_path = output_base / "temporal_predictions.csv"
    model_results["predictions"].to_csv(pred_path, index=False)
    logger.info(f"Saved model predictions: {pred_path.resolve()}")

    # ---- Save model summary ----
    summary_path = output_base / "temporal_model_summary.txt"
    with open(summary_path, "w") as f:
        f.write(model_results["model_summary"])
    logger.info(f"Saved model summary: {summary_path.resolve()}")

    # ---- Summary output ----
    logger.info("\n" + "=" * 60)
    logger.info("TEMPORAL MODEL SUMMARY")
    logger.info("=" * 60)

    diag = model_results["diagnostics"]
    logger.info(f"N observations: {diag['n_observations']}")
    logger.info(f"R-squared: {diag['r_squared']:.4f}")
    logger.info(f"Adjusted R-squared: {diag['adj_r_squared']:.4f}")
    logger.info(
        f"F-statistic: {diag['f_statistic']:.2f} (p={diag['f_pvalue']:.4e})"
    )
    logger.info(f"\nMax VIF: {diag['vif_max']:.2f}")

    logger.info("\nSignificant predictors (p < 0.05):")
    coef_df = model_results["coefficients"]
    sig_coefs = coef_df[coef_df["p_value"] < 0.05]
    for _, row in sig_coefs.iterrows():
        logger.info(
            f"  {row['variable']:25s}: "
            f"beta = {row['coefficient']:7.4f}, "
            f"p = {row['p_value']:.4e}"
        )

    logger.info("=" * 60)
    logger.info("\nTemporal model analysis complete!")


if __name__ == "__main__":
    main()
