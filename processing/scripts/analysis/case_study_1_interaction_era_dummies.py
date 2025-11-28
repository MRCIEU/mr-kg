"""Temporal regression with era dummies for Case Study 1.

This script compares a categorical era dummy model to the original continuous
year model to test whether categorical eras capture non-linear temporal patterns
better than continuous year.

Research Question:
    Does modeling time categorically (eras) fit better than continuous year?

Statistical Approach:
    - Original model: concordance ~ year + study_count + match_type_exact
    - Era dummy model: concordance ~ era_dummies + study_count + match_type_exact
    - Reference category: early_mr (omitted)
    - Model comparison: R², AIC, BIC, likelihood ratio test

Outputs:
    - temporal_model_era_dummies.csv: Coefficient estimates for era model
    - temporal_model_comparison.csv: Model fit comparison metrics
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from loguru import logger
from scipy import stats
from statsmodels.formula.api import ols
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
    with config_path.open("r") as f:
        res = yaml.safe_load(f)
    return res


# ==== Data loading ====


def load_data(config: Dict) -> pd.DataFrame:
    """Load pair reproducibility metrics with temporal variables.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with year, era dummies, study_count, match_type, concordance
    """
    output_config = config["output"]["case_study_1"]
    metrics_dir = PROJECT_ROOT / output_config["metrics"]
    input_csv = metrics_dir / "pair_reproducibility_metrics.csv"

    logger.info("Loading pair metrics from: {}", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Loaded {} trait pairs", len(df))

    required_cols = [
        "publication_year",
        "temporal_era",
        "study_count",
        "has_exact_match",
        "mean_direction_concordance",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.rename(
        columns={
            "publication_year": "year",
            "has_exact_match": "match_type_exact",
            "mean_direction_concordance": "direction_concordance",
        }
    )

    df_clean = df.dropna(
        subset=["year", "temporal_era", "match_type_exact"]
    ).copy()
    n_excluded = len(df) - len(df_clean)
    logger.info("Excluded {} pairs with missing data", n_excluded)

    # ---- Filter out 'other' and 'unknown' eras ----
    df_clean = df_clean[
        ~df_clean["temporal_era"].isin(["other", "unknown"])
    ].copy()
    logger.info(
        "Filtered to {} pairs after excluding other/unknown eras",
        len(df_clean),
    )

    eras = config["case_study_1"]["temporal_eras"].keys()
    for era in eras:
        df_clean[f"era_{era}"] = (df_clean["temporal_era"] == era).astype(int)

    logger.info(
        "Created era dummy variables: {}",
        ", ".join([f"era_{e}" for e in eras]),
    )

    res = df_clean
    return res


# ==== Model fitting ====


def fit_continuous_model(df: pd.DataFrame) -> Dict:
    """Fit original continuous year model.

    Args:
        df: DataFrame with year, study_count, match_type_exact, concordance

    Returns:
        Dictionary with model results
    """
    logger.info("Fitting continuous year model...")

    model = ols(
        "direction_concordance ~ year + study_count + match_type_exact",
        data=df,
    ).fit()

    results = {
        "model_type": "continuous_year",
        "formula": "concordance ~ year + study_count + match_type_exact",
        "n_observations": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "df_model": int(model.df_model),
        "df_residual": int(model.df_resid),
    }

    logger.info("Continuous model: R² = {:.4f}", results["r_squared"])

    res = (model, results)
    return res


def fit_era_dummy_model(df: pd.DataFrame, config: Dict) -> Dict:
    """Fit era dummy model with early_mr as reference.

    Args:
        df: DataFrame with era dummies, study_count, match_type_exact
        config: Configuration dictionary

    Returns:
        Dictionary with model results and coefficient table
    """
    logger.info("Fitting era dummy model...")

    eras = list(config["case_study_1"]["temporal_eras"].keys())

    era_predictors = [f"era_{era}" for era in eras if era != "early_mr"]

    formula = (
        "direction_concordance ~ "
        + " + ".join(era_predictors)
        + " + study_count + match_type_exact"
    )

    model = ols(formula, data=df).fit()

    coef_df = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "t_statistic": model.tvalues.values,
            "p_value": model.pvalues.values,
        }
    )

    coef_df["ci_lower"] = coef_df["coefficient"] - 1.96 * coef_df["std_error"]
    coef_df["ci_upper"] = coef_df["coefficient"] + 1.96 * coef_df["std_error"]

    coef_df["significant"] = coef_df["p_value"] < 0.05

    results = {
        "model_type": "era_dummy",
        "formula": formula,
        "reference_era": "early_mr",
        "n_observations": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "log_likelihood": float(model.llf),
        "df_model": int(model.df_model),
        "df_residual": int(model.df_resid),
        "coefficients": coef_df,
    }

    logger.info("Era dummy model: R² = {:.4f}", results["r_squared"])

    res = (model, results)
    return res


# ==== Model comparison ====


def compare_models(continuous_results: Dict, era_results: Dict) -> Dict:
    """Compare continuous and era dummy models.

    Args:
        continuous_results: Results from continuous model
        era_results: Results from era dummy model

    Returns:
        Dictionary with comparison metrics
    """
    logger.info("Comparing continuous and era dummy models...")

    r2_diff = era_results["r_squared"] - continuous_results["r_squared"]
    aic_diff = era_results["aic"] - continuous_results["aic"]
    bic_diff = era_results["bic"] - continuous_results["bic"]

    df_diff = era_results["df_model"] - continuous_results["df_model"]
    ll_diff = (
        era_results["log_likelihood"] - continuous_results["log_likelihood"]
    )
    lr_stat = 2 * ll_diff
    lr_p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

    if aic_diff < 0:
        aic_preferred = "era_dummy"
    elif aic_diff > 0:
        aic_preferred = "continuous"
    else:
        aic_preferred = "tie"

    if bic_diff < 0:
        bic_preferred = "era_dummy"
    elif bic_diff > 0:
        bic_preferred = "continuous"
    else:
        bic_preferred = "tie"

    if lr_p_value < 0.05:
        interpretation = (
            "Era dummy model significantly improves fit "
            f"(LR test: χ²({df_diff}) = {lr_stat:.2f}, p < 0.05)"
        )
    else:
        interpretation = (
            "No significant improvement with era dummies "
            f"(LR test: χ²({df_diff}) = {lr_stat:.2f}, p = {lr_p_value:.3f})"
        )

    comparison = {
        "r_squared_difference": r2_diff,
        "aic_difference": aic_diff,
        "bic_difference": bic_diff,
        "aic_preferred_model": aic_preferred,
        "bic_preferred_model": bic_preferred,
        "likelihood_ratio_statistic": lr_stat,
        "likelihood_ratio_df": df_diff,
        "likelihood_ratio_p_value": lr_p_value,
        "interpretation": interpretation,
    }

    logger.info("Model comparison: {}", interpretation)

    res = comparison
    return res


# ==== Main execution ====


def main() -> int:
    """Main entry point for era dummy temporal regression.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info("Starting era dummy temporal regression analysis...")

    config = load_config(DEFAULT_CONFIG)

    output_config = config["output"]["case_study_1"]
    models_dir = PROJECT_ROOT / output_config["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(config)

    continuous_model, continuous_results = fit_continuous_model(df)

    era_model, era_results = fit_era_dummy_model(df, config)

    comparison = compare_models(continuous_results, era_results)

    coef_csv = models_dir / "temporal_model_era_dummies.csv"
    era_results["coefficients"].to_csv(coef_csv, index=False)
    logger.info("Saved era dummy coefficients: {}", coef_csv)

    comparison_data = [
        {
            "model": "continuous_year",
            "r_squared": continuous_results["r_squared"],
            "adj_r_squared": continuous_results["adj_r_squared"],
            "aic": continuous_results["aic"],
            "bic": continuous_results["bic"],
            "df_model": continuous_results["df_model"],
            "df_residual": continuous_results["df_residual"],
        },
        {
            "model": "era_dummy",
            "r_squared": era_results["r_squared"],
            "adj_r_squared": era_results["adj_r_squared"],
            "aic": era_results["aic"],
            "bic": era_results["bic"],
            "df_model": era_results["df_model"],
            "df_residual": era_results["df_residual"],
        },
    ]

    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv = models_dir / "temporal_model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info("Saved model comparison: {}", comparison_csv)

    metadata_json = models_dir / "temporal_model_era_dummies_metadata.json"
    with metadata_json.open("w") as f:
        json.dump(
            {
                "script": "case_study_1_interaction_era_dummies.py",
                "continuous_model": {
                    k: v
                    for k, v in continuous_results.items()
                    if k != "coefficients"
                },
                "era_model": {
                    k: v for k, v in era_results.items() if k != "coefficients"
                },
                "comparison": comparison,
            },
            f,
            indent=2,
        )
    logger.info("Saved analysis metadata: {}", metadata_json)

    logger.info("Era dummy temporal regression analysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
