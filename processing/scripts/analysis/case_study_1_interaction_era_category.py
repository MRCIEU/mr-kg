"""Analyze era × category interaction for Case Study 1.

This script tests whether temporal improvement in reproducibility varies
across disease categories using two-way ANOVA and stratified trend analysis.

Research Question:
    Does the magnitude of reproducibility improvement over time differ
    across disease categories?

Statistical Approach:
    - Two-way ANOVA: concordance ~ era + category + era:category
    - Stratified analysis: Compute trends for each category separately
    - Effect sizes: Eta-squared for each term
    - Post-hoc: Category-specific temporal improvements

Outputs:
    - era_category_interaction.csv: Mean concordance by (category, era)
    - era_category_anova.json: ANOVA results and interpretation
    - era_category_interaction.png/svg: Line plot visualization
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
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
    """Load pair reproducibility metrics with era and category information.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with columns: pair_id, temporal_era, category,
            direction_concordance, study_count
    """
    output_config = config["output"]["case_study_1"]
    metrics_dir = PROJECT_ROOT / output_config["metrics"]
    input_csv = metrics_dir / "pair_reproducibility_metrics.csv"

    logger.info("Loading pair metrics from: {}", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Loaded {} trait pairs", len(df))

    required_cols = [
        "temporal_era",
        "outcome_category",
        "mean_direction_concordance",
        "study_count",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.rename(
        columns={
            "outcome_category": "category",
            "mean_direction_concordance": "direction_concordance",
        }
    )

    df_clean = df.dropna(subset=["temporal_era", "category"]).copy()
    n_excluded = len(df) - len(df_clean)
    logger.info("Excluded {} pairs with missing era or category", n_excluded)

    # ---- Filter out 'other' and 'unknown' eras ----
    df_clean = df_clean[
        ~df_clean["temporal_era"].isin(["other", "unknown"])
    ].copy()
    logger.info(
        "Filtered to {} pairs after excluding other/unknown eras",
        len(df_clean),
    )

    # ---- Filter out 'uncategorized' category ----
    df_clean = df_clean[df_clean["category"] != "uncategorized"].copy()
    logger.info(
        "Filtered to {} pairs after excluding uncategorized", len(df_clean)
    )

    res = df_clean
    return res


# ==== ANOVA analysis ====


def perform_anova(df: pd.DataFrame) -> Dict:
    """Perform two-way ANOVA with interaction term.

    Args:
        df: DataFrame with temporal_era, category, direction_concordance

    Returns:
        Dictionary with ANOVA results, F-statistics, p-values, effect sizes
    """
    logger.info("Performing two-way ANOVA with interaction...")

    model = ols(
        "direction_concordance ~ C(temporal_era) + C(category) + "
        "C(temporal_era):C(category)",
        data=df,
    ).fit()

    anova_table = anova_lm(model, typ=2)

    total_ss = anova_table["sum_sq"].sum()

    results = {
        "model": ("concordance ~ era + category + era:category"),
        "n_observations": len(df),
        "main_effects": {
            "era": {
                "F": float(anova_table.loc["C(temporal_era)", "F"]),
                "p": float(anova_table.loc["C(temporal_era)", "PR(>F)"]),
                "eta_sq": float(
                    anova_table.loc["C(temporal_era)", "sum_sq"] / total_ss
                ),
                "df": int(anova_table.loc["C(temporal_era)", "df"]),
            },
            "category": {
                "F": float(anova_table.loc["C(category)", "F"]),
                "p": float(anova_table.loc["C(category)", "PR(>F)"]),
                "eta_sq": float(
                    anova_table.loc["C(category)", "sum_sq"] / total_ss
                ),
                "df": int(anova_table.loc["C(category)", "df"]),
            },
        },
        "interaction": {
            "era_category": {
                "F": float(
                    anova_table.loc["C(temporal_era):C(category)", "F"]
                ),
                "p": float(
                    anova_table.loc["C(temporal_era):C(category)", "PR(>F)"]
                ),
                "eta_sq": float(
                    anova_table.loc["C(temporal_era):C(category)", "sum_sq"]
                    / total_ss
                ),
                "df": int(
                    anova_table.loc["C(temporal_era):C(category)", "df"]
                ),
            }
        },
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
    }

    interaction_p = results["interaction"]["era_category"]["p"]
    if interaction_p < 0.001:
        sig_str = "p < 0.001"
    elif interaction_p < 0.05:
        sig_str = f"p = {interaction_p:.3f}"
    else:
        sig_str = f"p = {interaction_p:.3f} (not significant)"

    if interaction_p < 0.05:
        results["interpretation"] = (
            f"Significant interaction ({sig_str}): "
            "temporal improvement varies by category"
        )
    else:
        results["interpretation"] = (
            f"No significant interaction ({sig_str}): "
            "temporal improvement is uniform across categories"
        )

    logger.info("ANOVA complete: {}", results["interpretation"])

    res = results
    return res


# ==== Stratified analysis ====


def stratified_analysis(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute mean concordance and improvement for each (category, era).

    Args:
        df: DataFrame with temporal_era, category, direction_concordance
        config: Configuration dictionary

    Returns:
        DataFrame with columns: category, era, n_pairs, mean_concordance,
            se_concordance, ci_lower, ci_upper, improvement
    """
    logger.info("Performing stratified analysis by (category, era)...")

    era_order = list(config["case_study_1"]["temporal_eras"].keys())

    grouped = (
        df.groupby(["category", "temporal_era"])["direction_concordance"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "category",
        "temporal_era",
        "n_pairs",
        "mean_concordance",
        "std_concordance",
    ]

    grouped["se_concordance"] = grouped["std_concordance"] / np.sqrt(
        grouped["n_pairs"]
    )

    grouped["ci_lower"] = (
        grouped["mean_concordance"] - 1.96 * grouped["se_concordance"]
    )
    grouped["ci_upper"] = (
        grouped["mean_concordance"] + 1.96 * grouped["se_concordance"]
    )

    grouped["ci_lower"] = grouped["ci_lower"].clip(lower=-1.0)
    grouped["ci_upper"] = grouped["ci_upper"].clip(upper=1.0)

    categories = grouped["category"].unique()
    improvements = []

    for category in categories:
        cat_data = grouped[grouped["category"] == category]

        early_data = cat_data[cat_data["temporal_era"] == "early_mr"]
        strobe_data = cat_data[cat_data["temporal_era"] == "strobe_mr"]

        if len(early_data) > 0 and len(strobe_data) > 0:
            improvement = (
                strobe_data["mean_concordance"].values[0]
                - early_data["mean_concordance"].values[0]
            )
        else:
            improvement = np.nan

        for _, row in cat_data.iterrows():
            improvements.append(improvement)

    grouped["improvement_early_to_strobe"] = improvements

    grouped = grouped.sort_values(
        ["category", "temporal_era"],
        key=lambda x: (
            x.map(lambda y: era_order.index(y))
            if x.name == "temporal_era"
            else x
        ),
    )

    logger.info(
        "Stratified analysis complete: {} (category, era) cells",
        len(grouped),
    )

    small_cells = grouped[grouped["n_pairs"] < 5]
    if len(small_cells) > 0:
        logger.warning(
            "{} cells have n < 5, results may be underpowered",
            len(small_cells),
        )
        for _, row in small_cells.iterrows():
            logger.warning(
                "  {} × {} (n={})",
                row["category"],
                row["temporal_era"],
                row["n_pairs"],
            )

    res = grouped
    return res


def identify_extreme_categories(
    stratified_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Identify categories with strongest and weakest improvement.

    Args:
        stratified_df: DataFrame from stratified_analysis

    Returns:
        Tuple of (strongest_improvement, weakest_improvement) category lists
    """
    improvement_by_cat = (
        stratified_df.groupby("category")["improvement_early_to_strobe"]
        .first()
        .dropna()
        .sort_values(ascending=False)
    )

    n_top = min(3, len(improvement_by_cat))

    strongest = improvement_by_cat.head(n_top).index.tolist()
    weakest = improvement_by_cat.tail(n_top).index.tolist()

    logger.info(
        "Strongest improvement categories: {}",
        ", ".join(strongest),
    )
    logger.info(
        "Weakest improvement categories: {}",
        ", ".join(weakest),
    )

    res = (strongest, weakest)
    return res


# ==== Visualization ====


def create_visualization(
    stratified_df: pd.DataFrame,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """Create line plot showing concordance over eras for each category.

    Args:
        stratified_df: DataFrame from stratified_analysis
        config: Configuration dictionary
        output_dir: Output directory for figures

    Returns:
        Metadata dictionary with plot details
    """
    logger.info("Creating era × category interaction visualization...")

    fig_config = config["figures"]
    era_order = list(config["case_study_1"]["temporal_eras"].keys())

    categories = sorted(stratified_df["category"].unique())
    n_categories = len(categories)

    colors = plt.cm.tab10(np.linspace(0, 1, n_categories))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=fig_config["dpi"])

    for i, category in enumerate(categories):
        cat_data = stratified_df[stratified_df["category"] == category].copy()

        cat_data["era_order"] = cat_data["temporal_era"].map(
            lambda x: era_order.index(x)
        )
        cat_data = cat_data.sort_values("era_order")

        ax.errorbar(
            cat_data["era_order"],
            cat_data["mean_concordance"],
            yerr=[
                cat_data["mean_concordance"] - cat_data["ci_lower"],
                cat_data["ci_upper"] - cat_data["mean_concordance"],
            ],
            marker="o",
            markersize=6,
            capsize=4,
            capthick=1.5,
            linewidth=2,
            label=category.title(),
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xticks(range(len(era_order)))
    ax.set_xticklabels(
        [era.replace("_", " ").title() for era in era_order],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Temporal Era", fontsize=12, fontweight="bold")
    ax.set_ylabel("Direction Concordance", fontsize=12, fontweight="bold")
    ax.set_title(
        "Temporal Improvement in Reproducibility by Disease Category",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylim(-0.1, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=10,
    )

    plt.tight_layout()

    png_file = output_dir / "era_category_interaction.png"
    svg_file = output_dir / "era_category_interaction.svg"

    fig.savefig(png_file, dpi=fig_config["dpi"], bbox_inches="tight")
    fig.savefig(svg_file, format="svg", bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved interaction plot: {}", png_file)
    logger.info("Saved interaction plot: {}", svg_file)

    res = {
        "n_categories": n_categories,
        "n_eras": len(era_order),
        "figure_files": [str(png_file), str(svg_file)],
    }
    return res


# ==== Main execution ====


def main() -> int:
    """Main entry point for era × category interaction analysis.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info("Starting era × category interaction analysis...")

    config = load_config(DEFAULT_CONFIG)

    output_config = config["output"]["case_study_1"]
    interactions_dir = PROJECT_ROOT / output_config["base"] / "interactions"
    interactions_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = PROJECT_ROOT / output_config["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(config)

    anova_results = perform_anova(df)

    stratified_df = stratified_analysis(df, config)

    strongest, weakest = identify_extreme_categories(stratified_df)

    anova_results["strongest_improvement"] = strongest
    anova_results["weakest_improvement"] = weakest

    output_csv = interactions_dir / "era_category_interaction.csv"
    stratified_df.to_csv(output_csv, index=False)
    logger.info("Saved stratified results: {}", output_csv)

    output_json = interactions_dir / "era_category_anova.json"
    with output_json.open("w") as f:
        json.dump(
            {
                "script": "case_study_1_interaction_era_category.py",
                "test": "two_way_anova",
                **anova_results,
            },
            f,
            indent=2,
        )
    logger.info("Saved ANOVA results: {}", output_json)

    plot_metadata = create_visualization(stratified_df, config, figures_dir)

    logger.info("Era × category interaction analysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
