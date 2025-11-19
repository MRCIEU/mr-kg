"""Analyze category × match type interaction for Case Study 1.

This script tests whether the effect of match quality (exact vs fuzzy) on
reproducibility varies across disease categories using two-way ANOVA and
stratified analysis.

Research Question:
    Does the effect of match quality differ across disease categories?

Statistical Approach:
    - Two-way ANOVA: concordance ~ category + match_type + category:match_type
    - Stratified analysis: Compute match quality penalty for each category
    - Effect sizes: Eta-squared for each term
    - Post-hoc: Category-specific sensitivity to match quality

Outputs:
    - category_match_interaction.csv: Mean concordance by (category, match_type)
    - category_match_anova.json: ANOVA results and interpretation
    - category_match_interaction.png/svg: Grouped bar chart visualization
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
    """Load pair reproducibility metrics with category and match type.

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame with columns: pair_id, category, match_type,
            direction_concordance, study_count
    """
    output_config = config["output"]["case_study_1"]
    metrics_dir = PROJECT_ROOT / output_config["metrics"]
    input_csv = metrics_dir / "pair_reproducibility_metrics.csv"

    logger.info("Loading pair metrics from: {}", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Loaded {} trait pairs", len(df))

    required_cols = [
        "outcome_category",
        "has_exact_match",
        "mean_direction_concordance",
        "study_count",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["match_type"] = df["has_exact_match"].apply(
        lambda x: "exact" if x else "fuzzy"
    )

    df = df.rename(
        columns={
            "outcome_category": "category",
            "mean_direction_concordance": "direction_concordance",
        }
    )

    df_clean = df.dropna(subset=["category", "match_type"]).copy()
    n_excluded = len(df) - len(df_clean)
    logger.info(
        "Excluded {} pairs with missing category or match_type",
        n_excluded,
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
        df: DataFrame with category, match_type, direction_concordance

    Returns:
        Dictionary with ANOVA results, F-statistics, p-values, effect sizes
    """
    logger.info("Performing two-way ANOVA with interaction...")

    model = ols(
        "direction_concordance ~ C(category) + C(match_type) + "
        "C(category):C(match_type)",
        data=df,
    ).fit()

    anova_table = anova_lm(model, typ=2)

    total_ss = anova_table["sum_sq"].sum()

    results = {
        "model": ("concordance ~ category + match_type + category:match_type"),
        "n_observations": len(df),
        "main_effects": {
            "category": {
                "F": float(anova_table.loc["C(category)", "F"]),
                "p": float(anova_table.loc["C(category)", "PR(>F)"]),
                "eta_sq": float(
                    anova_table.loc["C(category)", "sum_sq"] / total_ss
                ),
                "df": int(anova_table.loc["C(category)", "df"]),
            },
            "match_type": {
                "F": float(anova_table.loc["C(match_type)", "F"]),
                "p": float(anova_table.loc["C(match_type)", "PR(>F)"]),
                "eta_sq": float(
                    anova_table.loc["C(match_type)", "sum_sq"] / total_ss
                ),
                "df": int(anova_table.loc["C(match_type)", "df"]),
            },
        },
        "interaction": {
            "category_match_type": {
                "F": float(anova_table.loc["C(category):C(match_type)", "F"]),
                "p": float(
                    anova_table.loc["C(category):C(match_type)", "PR(>F)"]
                ),
                "eta_sq": float(
                    anova_table.loc["C(category):C(match_type)", "sum_sq"]
                    / total_ss
                ),
                "df": int(anova_table.loc["C(category):C(match_type)", "df"]),
            }
        },
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
    }

    interaction_p = results["interaction"]["category_match_type"]["p"]
    if interaction_p < 0.001:
        sig_str = "p < 0.001"
    elif interaction_p < 0.05:
        sig_str = f"p = {interaction_p:.3f}"
    else:
        sig_str = f"p = {interaction_p:.3f} (not significant)"

    if interaction_p < 0.05:
        results["interpretation"] = (
            f"Significant interaction ({sig_str}): "
            "match quality effect varies by category"
        )
    else:
        results["interpretation"] = (
            f"No significant interaction ({sig_str}): "
            "match quality effect is uniform across categories"
        )

    logger.info("ANOVA complete: {}", results["interpretation"])

    res = results
    return res


# ==== Stratified analysis ====


def stratified_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean concordance and penalty for each (category, match_type).

    Args:
        df: DataFrame with category, match_type, direction_concordance

    Returns:
        DataFrame with columns: category, match_type, n_pairs,
            mean_concordance, se_concordance, ci_lower, ci_upper,
            match_quality_penalty
    """
    logger.info("Performing stratified analysis by (category, match_type)...")

    grouped = (
        df.groupby(["category", "match_type"])["direction_concordance"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "category",
        "match_type",
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
    penalties = []

    for category in categories:
        cat_data = grouped[grouped["category"] == category]

        exact_data = cat_data[cat_data["match_type"] == "exact"]
        fuzzy_data = cat_data[cat_data["match_type"] == "fuzzy"]

        if len(exact_data) > 0 and len(fuzzy_data) > 0:
            penalty = (
                exact_data["mean_concordance"].iloc[0]
                - fuzzy_data["mean_concordance"].iloc[0]
            )
        else:
            penalty = np.nan

        for _, row in cat_data.iterrows():
            penalties.append(penalty)

    grouped["match_quality_penalty"] = penalties

    grouped = grouped.sort_values(["category", "match_type"])

    logger.info(
        "Stratified analysis complete: {} (category, match_type) cells",
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
                row["match_type"],
                row["n_pairs"],
            )

    res = grouped
    return res


def identify_extreme_categories(
    stratified_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Identify categories most and least sensitive to match quality.

    Args:
        stratified_df: DataFrame from stratified_analysis

    Returns:
        Tuple of (most_sensitive, least_sensitive) category lists
    """
    penalty_by_cat = (
        stratified_df.groupby("category")["match_quality_penalty"]
        .first()
        .dropna()
        .sort_values(ascending=False)
    )

    n_top = min(3, len(penalty_by_cat))

    most_sensitive = penalty_by_cat.head(n_top).index.tolist()
    least_sensitive = penalty_by_cat.tail(n_top).index.tolist()

    logger.info(
        "Most sensitive categories: {}",
        ", ".join(most_sensitive),
    )
    logger.info(
        "Least sensitive categories: {}",
        ", ".join(least_sensitive),
    )

    res = (most_sensitive, least_sensitive)
    return res


# ==== Visualization ====


def create_visualization(
    stratified_df: pd.DataFrame,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """Create grouped bar chart showing concordance by category and match type.

    Args:
        stratified_df: DataFrame from stratified_analysis
        config: Configuration dictionary
        output_dir: Output directory for figures

    Returns:
        Metadata dictionary with plot details
    """
    logger.info("Creating category × match type interaction visualization...")

    fig_config = config["figures"]

    categories = sorted(stratified_df["category"].unique())
    match_types = sorted(stratified_df["match_type"].unique())

    n_categories = len(categories)
    bar_width = 0.35
    x = np.arange(n_categories)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=fig_config["dpi"])

    colors = {"exact": "#2E7D32", "fuzzy": "#F57C00"}

    for i, match_type in enumerate(match_types):
        match_data = stratified_df[stratified_df["match_type"] == match_type]
        match_data = match_data.set_index("category").reindex(categories)

        means = match_data["mean_concordance"].values
        ci_lower = match_data["ci_lower"].values
        ci_upper = match_data["ci_upper"].values
        yerr = [means - ci_lower, ci_upper - means]

        ax.bar(
            x + i * bar_width,
            means,
            bar_width,
            yerr=yerr,
            capsize=4,
            label=match_type.title(),
            color=colors.get(match_type, "gray"),
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

    ax.set_xlabel("Disease Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Direction Concordance", fontsize=12, fontweight="bold")
    ax.set_title(
        "Match Quality Effect on Reproducibility by Disease Category",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(
        [cat.title() for cat in categories],
        rotation=45,
        ha="right",
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylim(-0.1, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    ax.legend(
        title="Match Type",
        loc="best",
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=10,
    )

    plt.tight_layout()

    png_file = output_dir / "category_match_interaction.png"
    svg_file = output_dir / "category_match_interaction.svg"

    fig.savefig(png_file, dpi=fig_config["dpi"], bbox_inches="tight")
    fig.savefig(svg_file, format="svg", bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved interaction plot: {}", png_file)
    logger.info("Saved interaction plot: {}", svg_file)

    res = {
        "n_categories": n_categories,
        "n_match_types": len(match_types),
        "figure_files": [str(png_file), str(svg_file)],
    }
    return res


# ==== Main execution ====


def main() -> int:
    """Main entry point for category × match type interaction analysis.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger.info("Starting category × match type interaction analysis...")

    config = load_config(DEFAULT_CONFIG)

    output_config = config["output"]["case_study_1"]
    interactions_dir = PROJECT_ROOT / output_config["base"] / "interactions"
    interactions_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = PROJECT_ROOT / output_config["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(config)

    anova_results = perform_anova(df)

    stratified_df = stratified_analysis(df)

    most_sensitive, least_sensitive = identify_extreme_categories(
        stratified_df
    )

    anova_results["most_sensitive"] = most_sensitive
    anova_results["least_sensitive"] = least_sensitive

    output_csv = interactions_dir / "category_match_interaction.csv"
    stratified_df.to_csv(output_csv, index=False)
    logger.info("Saved stratified results: {}", output_csv)

    output_json = interactions_dir / "category_match_anova.json"
    with output_json.open("w") as f:
        json.dump(
            {
                "script": "case_study_1_interaction_category_match.py",
                "test": "two_way_anova",
                **anova_results,
            },
            f,
            indent=2,
        )
    logger.info("Saved ANOVA results: {}", output_json)

    plot_metadata = create_visualization(stratified_df, config, figures_dir)

    logger.info("Category × match type interaction analysis complete!")

    return 0


if __name__ == "__main__":
    exit(main())
