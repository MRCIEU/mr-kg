"""Analyze pleiotropy awareness trends over time for Case Study 5 (RQ5).

This script analyzes how awareness and investigation of pleiotropic
relationships has evolved over time, particularly testing for the expected
decline in pleiotropy investigation after MR-PRESSO publication (2018).
All analyses filter to gpt-5 model only.

Research Question 5:
Has pleiotropy awareness increased over time?

Input:
    - data/db/vector_store.db (model_result_traits, model_results)
    - data/processed/case-study-cs5/temporal/temporal_metadata.csv
    - config/case_studies.yml

Output:
    - data/processed/case-study-cs5/pleiotropy/
        pleiotropy_rates_by_year.csv
        pleiotropy_rates_by_era.csv
        temporal_trend_tests.csv
        mr_presso_impact_analysis.csv
        pleiotropy_awareness_metadata.json
        pleiotropy_awareness_summary.md
    - data/processed/case-study-cs5/figures/
        pleiotropy_awareness_over_time.png/svg
        pleiotropy_by_era.png/svg
        mr_presso_impact.png/svg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats
from sklearn.linear_model import LogisticRegression

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

    # ---- --db ----
    parser.add_argument(
        "-d",
        "--db",
        type=Path,
        default=None,
        help="Path to vector_store database (overrides config)",
    )

    # ---- --temporal-metadata ----
    parser.add_argument(
        "-t",
        "--temporal-metadata",
        type=Path,
        default=None,
        help="Path to temporal_metadata.csv (overrides default)",
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


def load_trait_data(
    db_path: Path, model_filter: str = "gpt-5"
) -> pd.DataFrame:
    """Load trait data from vector_store database.

    Extracts exposure-outcome trait pairs by parsing the metadata JSON
    from model_results and joining with model_result_traits.

    Args:
        db_path: Path to vector_store.db
        model_filter: Model to filter to (default: gpt-5)

    Returns:
        DataFrame with columns: pmid, model, exposure_trait, outcome_trait
    """
    logger.info(f"Loading trait data from {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)

    # ---- Load model results with metadata ----
    query = """
    SELECT
        mr.id as model_result_id,
        mr.pmid,
        mr.model,
        mr.metadata
    FROM model_results mr
    WHERE mr.model = ?
    """

    df_results = con.execute(query, [model_filter]).df()
    logger.info(f"Loaded {len(df_results):,} model results")

    # ---- Parse metadata to extract exposure-outcome pairs ----
    trait_pairs = []

    for _, row in df_results.iterrows():
        pmid = row["pmid"]
        model = row["model"]
        metadata = json.loads(row["metadata"])

        exposures = metadata.get("exposures", [])
        outcomes = metadata.get("outcomes", [])

        # ---- Create all exposure-outcome combinations ----
        for exp in exposures:
            exp_trait = exp.get("trait", "")
            for out in outcomes:
                out_trait = out.get("trait", "")
                if exp_trait and out_trait:
                    trait_pairs.append(
                        {
                            "pmid": pmid,
                            "model": model,
                            "exposure_trait": exp_trait,
                            "outcome_trait": out_trait,
                        }
                    )

    con.close()

    df = pd.DataFrame(trait_pairs)

    if len(df) > 0:
        logger.info(
            f"Loaded {len(df):,} trait pairs from "
            f"{df['pmid'].nunique():,} studies"
        )
    else:
        logger.warning("No trait pairs found")

    res = df
    return res


def load_temporal_metadata(temporal_path: Path) -> pd.DataFrame:
    """Load temporal metadata from Phase 1 output.

    Args:
        temporal_path: Path to temporal_metadata.csv

    Returns:
        DataFrame with temporal annotations
    """
    logger.info(f"Loading temporal metadata from {temporal_path}")

    if not temporal_path.exists():
        logger.error(f"Temporal metadata not found: {temporal_path}")
        logger.error("Run case_study_5_temporal_preparation.py first")
        sys.exit(1)

    df = pd.read_csv(temporal_path)

    # ---- Ensure pmid is integer type for merging ----
    df["pmid"] = df["pmid"].astype(int)

    logger.info(f"Loaded metadata for {len(df):,} studies")

    res = df
    return res


def normalize_trait(trait: str) -> str:
    """Normalize trait string for matching.

    Args:
        trait: Raw trait string

    Returns:
        Normalized trait string (lowercase, stripped)
    """
    if pd.isna(trait):
        return ""
    res = str(trait).lower().strip()
    return res


def is_pleiotropic_pair(
    exposure: str, outcome: str, canonical_pairs: List[List[str]]
) -> bool:
    """Check if trait pair matches any canonical pleiotropic relationship.

    Args:
        exposure: Exposure trait
        outcome: Outcome trait
        canonical_pairs: List of canonical pleiotropic pairs

    Returns:
        True if pair matches any canonical relationship (bidirectional)
    """
    exp_norm = normalize_trait(exposure)
    out_norm = normalize_trait(outcome)

    for pair in canonical_pairs:
        trait1_norm = normalize_trait(pair[0])
        trait2_norm = normalize_trait(pair[1])

        # ---- Check bidirectional match ----
        if (exp_norm == trait1_norm and out_norm == trait2_norm) or (
            exp_norm == trait2_norm and out_norm == trait1_norm
        ):
            return True

    return False


def identify_pleiotropic_studies(
    trait_df: pd.DataFrame, canonical_pairs: List[List[str]]
) -> pd.DataFrame:
    """Identify studies investigating pleiotropic relationships.

    Args:
        trait_df: DataFrame with trait pairs
        canonical_pairs: List of canonical pleiotropic pairs

    Returns:
        DataFrame with added is_pleiotropic column
    """
    logger.info("Identifying pleiotropic studies")

    trait_df["is_pleiotropic"] = trait_df.apply(
        lambda row: is_pleiotropic_pair(
            row["exposure_trait"], row["outcome_trait"], canonical_pairs
        ),
        axis=1,
    )

    n_pleiotropic = trait_df["is_pleiotropic"].sum()
    n_studies_pleio = trait_df[trait_df["is_pleiotropic"]]["pmid"].nunique()

    logger.info(
        f"Found {n_pleiotropic:,} pleiotropic pairs in "
        f"{n_studies_pleio:,} studies"
    )

    res = trait_df
    return res


def compute_pleiotropy_rates_by_year(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pleiotropy investigation rates by year.

    Args:
        df: DataFrame with is_pleiotropic and pub_year columns

    Returns:
        DataFrame with yearly rates
    """
    logger.info("Computing pleiotropy rates by year")

    # ---- Group by year ----
    yearly = (
        df.groupby("pub_year")
        .agg(
            total_studies=("pmid", "nunique"),
            pleiotropic_studies=("is_pleiotropic", lambda x: (x).sum()),
            total_pairs=("is_pleiotropic", "count"),
            pleiotropic_pairs=("is_pleiotropic", "sum"),
        )
        .reset_index()
    )

    # ---- Compute rates ----
    yearly["pleiotropy_rate_studies"] = (
        yearly["pleiotropic_studies"] / yearly["total_studies"]
    )
    yearly["pleiotropy_rate_pairs"] = (
        yearly["pleiotropic_pairs"] / yearly["total_pairs"]
    )

    res = yearly.sort_values("pub_year")
    return res


def compute_pleiotropy_rates_by_era(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pleiotropy investigation rates by era.

    Args:
        df: DataFrame with is_pleiotropic and era columns

    Returns:
        DataFrame with era rates
    """
    logger.info("Computing pleiotropy rates by era")

    # ---- Group by era ----
    era_stats = (
        df.groupby("era")
        .agg(
            total_studies=("pmid", "nunique"),
            pleiotropic_studies=("is_pleiotropic", lambda x: (x).sum()),
            total_pairs=("is_pleiotropic", "count"),
            pleiotropic_pairs=("is_pleiotropic", "sum"),
        )
        .reset_index()
    )

    # ---- Compute rates ----
    era_stats["pleiotropy_rate_studies"] = (
        era_stats["pleiotropic_studies"] / era_stats["total_studies"]
    )
    era_stats["pleiotropy_rate_pairs"] = (
        era_stats["pleiotropic_pairs"] / era_stats["total_pairs"]
    )

    res = era_stats
    return res


def test_temporal_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """Test for temporal trend in pleiotropy investigation using logistic
    regression.

    Args:
        df: DataFrame with pub_year and is_pleiotropic columns

    Returns:
        Dictionary with test results
    """
    logger.info("Testing temporal trend with logistic regression")

    # ---- Prepare data ----
    X = df[["pub_year"]].values
    y = df["is_pleiotropic"].values.astype(int)

    # ---- Fit model ----
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    # ---- Get coefficient and p-value using likelihood ratio test ----
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]

    # ---- Predict probabilities for interpretation ----
    years = np.array([df["pub_year"].min(), df["pub_year"].max()]).reshape(
        -1, 1
    )
    probs = model.predict_proba(years)[:, 1]

    res = {
        "test_type": "logistic_regression",
        "coefficient": float(coef),
        "intercept": float(intercept),
        "n_observations": len(df),
        "prob_at_min_year": float(probs[0]),
        "prob_at_max_year": float(probs[1]),
        "change": float(probs[1] - probs[0]),
        "interpretation": "positive" if coef > 0 else "negative",
    }

    return res


def test_mr_presso_impact(
    df: pd.DataFrame, breakpoint: int = 2018
) -> Dict[str, Any]:
    """Test for change in pleiotropy investigation before/after MR-PRESSO.

    Args:
        df: DataFrame with pub_year and is_pleiotropic columns
        breakpoint: Year of MR-PRESSO publication

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing MR-PRESSO impact (breakpoint: {breakpoint})")

    # ---- Split data ----
    pre = df[df["pub_year"] < breakpoint]
    post = df[df["pub_year"] >= breakpoint]

    # ---- Compute rates ----
    pre_rate = pre["is_pleiotropic"].mean()
    post_rate = post["is_pleiotropic"].mean()

    # ---- Chi-square test ----
    contingency = pd.crosstab(
        df["pub_year"] >= breakpoint, df["is_pleiotropic"]
    )
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # ---- Effect size (Cramer's V) ----
    n = len(df)
    cramers_v = np.sqrt(chi2 / n)

    res = {
        "test_type": "chi_square",
        "breakpoint": breakpoint,
        "pre_n": len(pre),
        "post_n": len(post),
        "pre_rate": float(pre_rate),
        "post_rate": float(post_rate),
        "rate_change": float(post_rate - pre_rate),
        "percentage_point_change": float((post_rate - pre_rate) * 100),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "significant": p_value < 0.05,
    }

    return res


def test_era_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Test for differences in pleiotropy rates across eras.

    Args:
        df: DataFrame with era and is_pleiotropic columns

    Returns:
        DataFrame with pairwise era comparisons
    """
    logger.info("Testing era differences with pairwise chi-square tests")

    eras = sorted(df["era"].unique())
    results = []

    for i, era1 in enumerate(eras):
        for era2 in eras[i + 1 :]:
            # ---- Filter to two eras ----
            subset = df[df["era"].isin([era1, era2])]

            # ---- Chi-square test ----
            contingency = pd.crosstab(subset["era"], subset["is_pleiotropic"])

            if contingency.shape == (2, 2):
                chi2, p_value, dof, expected = stats.chi2_contingency(
                    contingency
                )

                # ---- Rates ----
                rate1 = df[df["era"] == era1]["is_pleiotropic"].mean()
                rate2 = df[df["era"] == era2]["is_pleiotropic"].mean()

                results.append(
                    {
                        "era1": era1,
                        "era2": era2,
                        "era1_rate": rate1,
                        "era2_rate": rate2,
                        "rate_difference": rate2 - rate1,
                        "chi2": chi2,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                )

    res = pd.DataFrame(results)
    return res


def plot_pleiotropy_over_time(
    yearly_df: pd.DataFrame,
    output_dir: Path,
    formats: List[str] = ["png", "svg"],
) -> None:
    """Plot pleiotropy investigation rate over time.

    Args:
        yearly_df: DataFrame with yearly pleiotropy rates
        output_dir: Output directory for figures
        formats: Image formats to save
    """
    logger.info("Plotting pleiotropy awareness over time")

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- Plot rate with confidence intervals ----
    ax.plot(
        yearly_df["pub_year"],
        yearly_df["pleiotropy_rate_pairs"],
        marker="o",
        linewidth=2,
        label="Pleiotropy investigation rate",
    )

    # ---- Add MR-PRESSO breakpoint ----
    ax.axvline(
        x=2018,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="MR-PRESSO publication (2018)",
    )

    ax.set_xlabel("Publication Year", fontsize=12)
    ax.set_ylabel("Proportion of Pleiotropic Pairs", fontsize=12)
    ax.set_title(
        "Pleiotropy Awareness Over Time", fontsize=14, fontweight="bold"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ---- Save in multiple formats ----
    for fmt in formats:
        output_path = output_dir / f"pleiotropy_awareness_over_time.{fmt}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close()


def plot_pleiotropy_by_era(
    era_df: pd.DataFrame, output_dir: Path, formats: List[str] = ["png", "svg"]
) -> None:
    """Plot pleiotropy investigation rate by era.

    Args:
        era_df: DataFrame with era pleiotropy rates
        output_dir: Output directory for figures
        formats: Image formats to save
    """
    logger.info("Plotting pleiotropy awareness by era")

    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- Bar plot ----
    bars = ax.bar(
        era_df["era"],
        era_df["pleiotropy_rate_pairs"],
        color="steelblue",
        alpha=0.7,
    )

    # ---- Add value labels ----
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Era", fontsize=12)
    ax.set_ylabel("Proportion of Pleiotropic Pairs", fontsize=12)
    ax.set_title("Pleiotropy Awareness by Era", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # ---- Save in multiple formats ----
    for fmt in formats:
        output_path = output_dir / f"pleiotropy_by_era.{fmt}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close()


def plot_mr_presso_impact(
    df: pd.DataFrame,
    breakpoint: int,
    output_dir: Path,
    formats: List[str] = ["png", "svg"],
) -> None:
    """Plot MR-PRESSO impact on pleiotropy investigation.

    Args:
        df: DataFrame with yearly rates
        breakpoint: MR-PRESSO publication year
        output_dir: Output directory for figures
        formats: Image formats to save
    """
    logger.info("Plotting MR-PRESSO impact")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: Time series with breakpoint ----
    pre = df[df["pub_year"] < breakpoint]
    post = df[df["pub_year"] >= breakpoint]

    ax1.plot(
        pre["pub_year"],
        pre["pleiotropy_rate_pairs"],
        marker="o",
        linewidth=2,
        color="steelblue",
        label="Pre-MR-PRESSO",
    )
    ax1.plot(
        post["pub_year"],
        post["pleiotropy_rate_pairs"],
        marker="o",
        linewidth=2,
        color="coral",
        label="Post-MR-PRESSO",
    )
    ax1.axvline(x=breakpoint, color="red", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Publication Year", fontsize=12)
    ax1.set_ylabel("Proportion of Pleiotropic Pairs", fontsize=12)
    ax1.set_title("Temporal Trend with MR-PRESSO Breakpoint", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Right: Before/After comparison ----
    pre_rate = pre["pleiotropy_rate_pairs"].mean()
    post_rate = post["pleiotropy_rate_pairs"].mean()

    bars = ax2.bar(
        ["Pre-MR-PRESSO", "Post-MR-PRESSO"],
        [pre_rate, post_rate],
        color=["steelblue", "coral"],
        alpha=0.7,
    )

    # ---- Add value labels ----
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax2.set_ylabel("Mean Proportion of Pleiotropic Pairs", fontsize=12)
    ax2.set_title("Before vs After MR-PRESSO", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # ---- Save in multiple formats ----
    for fmt in formats:
        output_path = output_dir / f"mr_presso_impact.{fmt}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved figure: {output_path}")

    plt.close()


def generate_summary(
    yearly_df: pd.DataFrame,
    era_df: pd.DataFrame,
    trend_test: Dict[str, Any],
    presso_test: Dict[str, Any],
    era_tests: pd.DataFrame,
    config: Dict[str, Any],
) -> str:
    """Generate markdown summary of pleiotropy awareness analysis.

    Args:
        yearly_df: Yearly rates DataFrame
        era_df: Era rates DataFrame
        trend_test: Temporal trend test results
        presso_test: MR-PRESSO impact test results
        era_tests: Era comparison test results
        config: Configuration dictionary

    Returns:
        Markdown summary string
    """
    logger.info("Generating summary markdown")

    summary = f"""# Pleiotropy Awareness Analysis Summary

## Research Question 5
Has pleiotropy awareness increased over time?

## Key Findings

### Overall Temporal Trend
- **Trend direction**: {trend_test["interpretation"]}
- **Logistic regression coefficient**: {trend_test["coefficient"]:.4f}
- **Change over time**: {trend_test["change"]:.3f}
- **Observations**: {trend_test["n_observations"]:,}

### MR-PRESSO Impact (Breakpoint: {presso_test["breakpoint"]})
- **Pre-MR-PRESSO rate**: {presso_test["pre_rate"]:.3f}
- **Post-MR-PRESSO rate**: {presso_test["post_rate"]:.3f}
- **Rate change**: {presso_test["rate_change"]:.3f} ({presso_test["percentage_point_change"]:.1f} pp)
- **Statistical significance**: {"Yes" if presso_test["significant"] else "No"} (p = {presso_test["p_value"]:.4f})
- **Effect size (Cramer's V)**: {presso_test["cramers_v"]:.3f}

### Era Comparison
| Era | Pleiotropy Rate | Total Studies | Pleiotropic Studies |
|-----|----------------|---------------|---------------------|
"""

    for _, row in era_df.iterrows():
        summary += f"| {row['era']} | {row['pleiotropy_rate_pairs']:.3f} | {row['total_studies']:,} | {row['pleiotropic_studies']:,} |\n"

    summary += """
### Pairwise Era Comparisons
| Era 1 | Era 2 | Rate Difference | p-value | Significant |
|-------|-------|----------------|---------|-------------|
"""

    for _, row in era_tests.iterrows():
        summary += f"| {row['era1']} | {row['era2']} | {row['rate_difference']:.3f} | {row['p_value']:.4f} | {'Yes' if row['significant'] else 'No'} |\n"

    summary += """
## Canonical Pleiotropic Pairs
"""

    for pair in config["case_study_5"]["pleiotropy"]["canonical_pairs"]:
        summary += f"- {pair[0]} <-> {pair[1]}\n"

    summary += f"""
## Data Quality
- **Total years analyzed**: {len(yearly_df)}
- **Year range**: {yearly_df["pub_year"].min()} - {yearly_df["pub_year"].max()}
- **Total trait pairs**: {yearly_df["total_pairs"].sum():,}
- **Total pleiotropic pairs**: {yearly_df["pleiotropic_pairs"].sum():,}
- **Overall pleiotropy rate**: {yearly_df["pleiotropic_pairs"].sum() / yearly_df["total_pairs"].sum():.3f}

## Interpretation

The analysis reveals that pleiotropy investigation has {trend_test["interpretation"]} over time.
The MR-PRESSO publication in {presso_test["breakpoint"]} {"had a significant impact" if presso_test["significant"] else "did not have a significant impact"} on pleiotropy investigation rates,
with a {abs(presso_test["percentage_point_change"]):.1f} percentage point {"decrease" if presso_test["rate_change"] < 0 else "increase"} in the post-MR-PRESSO era.

## Configuration
- **Model filter**: gpt-5
- **MR-PRESSO breakpoint**: {config["case_study_5"]["pleiotropy"]["mr_presso_breakpoint"]}
- **Canonical pairs**: {len(config["case_study_5"]["pleiotropy"]["canonical_pairs"])}
- **CS2 hotspots**: {len(config["case_study_5"]["pleiotropy"]["cs2_hotspots"])}
"""

    res = summary
    return res


def save_metadata(
    config: Dict[str, Any],
    n_studies: int,
    n_pairs: int,
    n_pleiotropic: int,
    output_dir: Path,
) -> None:
    """Save analysis metadata to JSON.

    Args:
        config: Configuration dictionary
        n_studies: Total number of studies
        n_pairs: Total number of trait pairs
        n_pleiotropic: Total number of pleiotropic pairs
        output_dir: Output directory
    """
    metadata = {
        "analysis": "pleiotropy_awareness",
        "case_study": "CS5",
        "research_question": "RQ5",
        "model_filter": "gpt-5",
        "n_studies": int(n_studies),
        "n_trait_pairs": int(n_pairs),
        "n_pleiotropic_pairs": int(n_pleiotropic),
        "canonical_pairs": config["case_study_5"]["pleiotropy"][
            "canonical_pairs"
        ],
        "cs2_hotspots": config["case_study_5"]["pleiotropy"]["cs2_hotspots"],
        "mr_presso_breakpoint": config["case_study_5"]["pleiotropy"][
            "mr_presso_breakpoint"
        ],
    }

    output_path = output_dir / "pleiotropy_awareness_metadata.json"
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata: {output_path}")


def main() -> None:
    """Main analysis pipeline."""
    args = parse_args()

    # ---- Configure logging ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # ---- Load configuration ----
    config = load_config(args.config)

    # ---- Resolve paths ----
    db_path = (
        args.db
        if args.db
        else PROJECT_ROOT / Path(config["databases"]["vector_store"])
    )
    temporal_path = (
        args.temporal_metadata
        if args.temporal_metadata
        else PROJECT_ROOT
        / Path(config["output"]["case_study_5"]["temporal"])
        / "temporal_metadata.csv"
    )
    output_dir = PROJECT_ROOT / Path(
        config["output"]["case_study_5"]["pleiotropy"]
    )
    figures_dir = PROJECT_ROOT / Path(
        config["output"]["case_study_5"]["figures"]
    )

    # ---- Create output directories ----
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Case Study 5: Pleiotropy Awareness Analysis (Phase 6)")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    # ==== Load data ====
    # ---- Get model filter from config ----
    models_included = config["case_study_5"]["models_included"]
    if not models_included:
        logger.error(
            "No models specified in config['case_study_5']['models_included']"
        )
        sys.exit(1)
    model_filter = models_included[0]  # Use first model
    logger.info(f"Filtering to model: {model_filter}")

    trait_df = load_trait_data(db_path, model_filter=model_filter)
    temporal_df = load_temporal_metadata(temporal_path)

    # ==== Merge datasets ====
    logger.info("Merging trait and temporal data")

    # ---- Ensure consistent types for merge keys ----
    trait_df["pmid"] = trait_df["pmid"].astype(int)
    temporal_df["pmid"] = temporal_df["pmid"].astype(int)

    merged_df = trait_df.merge(
        temporal_df[["pmid", "model", "pub_year", "era"]],
        on=["pmid", "model"],
        how="inner",
    )
    logger.info(f"Merged dataset: {len(merged_df):,} rows")

    # ==== Identify pleiotropic studies ====
    canonical_pairs = config["case_study_5"]["pleiotropy"]["canonical_pairs"]
    merged_df = identify_pleiotropic_studies(merged_df, canonical_pairs)

    # ==== Compute rates ====
    yearly_df = compute_pleiotropy_rates_by_year(merged_df)
    era_df = compute_pleiotropy_rates_by_era(merged_df)

    # ==== Statistical tests ====
    trend_test = test_temporal_trend(merged_df)
    presso_breakpoint = config["case_study_5"]["pleiotropy"][
        "mr_presso_breakpoint"
    ]
    presso_test = test_mr_presso_impact(merged_df, presso_breakpoint)
    era_tests = test_era_differences(merged_df)

    # ==== Generate summary ====
    summary = generate_summary(
        yearly_df, era_df, trend_test, presso_test, era_tests, config
    )

    # ==== Save outputs ====
    if not args.dry_run:
        # ---- CSVs ----
        yearly_df.to_csv(
            output_dir / "pleiotropy_rates_by_year.csv", index=False
        )
        logger.info(f"Saved: {output_dir / 'pleiotropy_rates_by_year.csv'}")

        era_df.to_csv(output_dir / "pleiotropy_rates_by_era.csv", index=False)
        logger.info(f"Saved: {output_dir / 'pleiotropy_rates_by_era.csv'}")

        # ---- Statistical tests ----
        trend_df = pd.DataFrame([trend_test])
        trend_df.to_csv(output_dir / "temporal_trend_tests.csv", index=False)
        logger.info(f"Saved: {output_dir / 'temporal_trend_tests.csv'}")

        presso_df = pd.DataFrame([presso_test])
        presso_df.to_csv(
            output_dir / "mr_presso_impact_analysis.csv", index=False
        )
        logger.info(f"Saved: {output_dir / 'mr_presso_impact_analysis.csv'}")

        era_tests.to_csv(output_dir / "era_comparison_tests.csv", index=False)
        logger.info(f"Saved: {output_dir / 'era_comparison_tests.csv'}")

        # ---- Summary ----
        summary_path = output_dir / "pleiotropy_awareness_summary.md"
        with open(summary_path, "w") as f:
            f.write(summary)
        logger.info(f"Saved: {summary_path}")

        # ---- Metadata ----
        save_metadata(
            config,
            merged_df["pmid"].nunique(),
            len(merged_df),
            merged_df["is_pleiotropic"].sum(),
            output_dir,
        )

        # ---- Figures ----
        formats = config["figures"]["format"]
        plot_pleiotropy_over_time(yearly_df, figures_dir, formats)
        plot_pleiotropy_by_era(era_df, figures_dir, formats)
        plot_mr_presso_impact(
            yearly_df, presso_breakpoint, figures_dir, formats
        )

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
