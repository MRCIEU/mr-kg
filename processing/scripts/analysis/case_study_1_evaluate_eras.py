"""Evaluate new temporal era definitions for CS1.

This script analyzes the sample size distribution and trends across
the new 5-era temporal stratification to determine if the granular
eras provide scientific value or should be collapsed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy import stats
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"


def make_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration file (default: {DEFAULT_CONFIG})",
    )

    # ---- --metrics-file ----
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Path to pair_reproducibility_metrics.csv",
    )

    res = parser.parse_args()
    return res


def load_config(config_path: Path):
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    """Evaluate temporal era stratification."""
    args = make_args()

    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    output_config = config["output"]["case_study_1"]

    if args.metrics_file:
        metrics_file = args.metrics_file
    else:
        metrics_dir = PROJECT_ROOT / output_config["metrics"]
        metrics_file = metrics_dir / "stratified_by_temporal_era.csv"

    if not metrics_file.exists():
        logger.error("Metrics file not found: {}", metrics_file)
        logger.error(
            "Please run case_study_1_reproducibility_metrics.py first"
        )
        return 1

    logger.info("Loading metrics from: {}", metrics_file)
    era_metrics = pd.read_csv(metrics_file)

    print("\n" + "=" * 70)
    print("TEMPORAL ERA EVALUATION")
    print("=" * 70)

    # ---- era sample sizes and key metrics ----
    print("\nEra Sample Sizes and Key Metrics:")
    print("-" * 70)

    era_names = [
        "early_mr",
        "mr_egger",
        "mr_presso",
        "within_family",
        "strobe_mr",
    ]

    era_stats = []
    for era in era_names:
        era_data = era_metrics[era_metrics["temporal_era"] == era]

        if len(era_data) == 0:
            print(f"\n{era}: NO DATA")
            continue

        n_pairs = era_data["n_pairs"].iloc[0]
        mean_conc = era_data["mean_concordance"].iloc[0]
        pct_high = era_data["pct_high"].iloc[0]

        era_range = config["case_study_1"]["temporal_eras"][era]
        era_label = f"{era} ({era_range[0]}-{era_range[1]})"

        print(f"\n{era_label}:")
        print(f"  n_pairs: {n_pairs}")
        print(f"  mean_concordance: {mean_conc:.3f}")
        print(f"  pct_high_tier: {pct_high:.1f}%")

        era_stats.append(
            {
                "era": era,
                "era_label": era_label,
                "n_pairs": n_pairs,
                "mean_concordance": mean_conc,
                "pct_high": pct_high,
            }
        )

    # ---- statistical tests ----
    print("\n" + "-" * 70)
    print("Statistical Tests:")
    print("-" * 70)

    era_df = pd.DataFrame(era_stats)

    if len(era_df) < 2:
        print("\nInsufficient eras with data for statistical testing")
        return 1

    # ---- check for monotonic trends ----
    print("\nMonotonic Trend Tests (Spearman correlation):")

    conc_corr, conc_p = stats.spearmanr(
        range(len(era_df)), era_df["mean_concordance"]
    )
    print(f"  Concordance vs. era: rho={conc_corr:.3f}, p={conc_p:.4f}")

    pct_corr, pct_p = stats.spearmanr(range(len(era_df)), era_df["pct_high"])
    print(f"  Pct high tier vs. era: rho={pct_corr:.3f}, p={pct_p:.4f}")

    # ---- load full pair-level data for ANOVA ----
    pair_metrics_file = (
        metrics_file.parent / "pair_reproducibility_metrics.csv"
    )
    if pair_metrics_file.exists():
        print("\nOne-way ANOVA (era effect on concordance):")

        pairs_df = pd.read_csv(pair_metrics_file)

        # ---- filter to eras with data ----
        valid_eras = era_df["era"].tolist()
        pairs_filtered = pairs_df[
            pairs_df["temporal_era"].isin(valid_eras)
        ].copy()

        # ---- perform ANOVA ----
        era_groups = [
            pairs_filtered[pairs_filtered["temporal_era"] == era][
                "mean_direction_concordance"
            ].dropna()
            for era in valid_eras
        ]

        f_stat, p_value = stats.f_oneway(*era_groups)

        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  P-value: {p_value:.4f}")
        print(
            f"  Significant difference between eras: {'Yes' if p_value < 0.05 else 'No'}"
        )

        # ---- pairwise comparisons if significant ----
        if p_value < 0.05:
            print("\nPairwise Mann-Whitney U tests:")
            for i in range(len(valid_eras)):
                for j in range(i + 1, len(valid_eras)):
                    era1, era2 = valid_eras[i], valid_eras[j]
                    u_stat, u_p = stats.mannwhitneyu(
                        era_groups[i], era_groups[j], alternative="two-sided"
                    )
                    print(f"  {era1} vs {era2}: U={u_stat:.1f}, p={u_p:.4f}")

    # ---- sample size warnings ----
    print("\n" + "-" * 70)
    print("Sample Size Warnings:")
    print("-" * 70)

    min_sample = 20
    small_eras = era_df[era_df["n_pairs"] < min_sample]

    if len(small_eras) > 0:
        print(
            f"\nEras with < {min_sample} pairs (insufficient for robust analysis):"
        )
        for _, row in small_eras.iterrows():
            print(f"  {row['era_label']}: n={row['n_pairs']}")
    else:
        print(f"\nAll eras have >= {min_sample} pairs")

    # ---- recommendation ----
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # ---- criteria for keeping granular eras ----
    sufficient_sample = all(era_df["n_pairs"] >= min_sample)
    significant_diff = p_value < 0.05 if pair_metrics_file.exists() else False
    interesting_trend = abs(conc_corr) > 0.5 or abs(pct_corr) > 0.5

    if sufficient_sample and (significant_diff or interesting_trend):
        print("\n✅ KEEP GRANULAR ERAS")
        print("\nJustification:")
        if sufficient_sample:
            print("  - All eras have sufficient sample sizes")
        if significant_diff:
            print("  - Statistically significant differences between eras")
        if interesting_trend:
            print("  - Strong monotonic trends observed")
    elif len(small_eras) <= 1 and (significant_diff or interesting_trend):
        print("\n⚠️ KEEP GRANULAR ERAS (with caution)")
        print("\nJustification:")
        print(
            f"  - Most eras have sufficient sample sizes (only {len(small_eras)} below threshold)"
        )
        if significant_diff:
            print("  - Statistically significant differences between eras")
        if interesting_trend:
            print("  - Strong monotonic trends observed")
        print("\nCaution:")
        print("  - Consider collapsing or excluding eras with small samples")
    else:
        print("\n❌ COLLAPSE TO 2-3 GROUPS")
        print("\nJustification:")
        if not sufficient_sample:
            print(
                f"  - {len(small_eras)} eras have insufficient sample sizes (< {min_sample})"
            )
        if not significant_diff:
            print("  - No statistically significant differences between eras")
        if not interesting_trend:
            print("  - No strong monotonic trends observed")
        print("\nSuggested grouping:")
        print("  - Early MR (2010-2017): early_mr + mr_egger")
        print("  - Advanced MR (2018-2019): mr_presso")
        print("  - Modern MR (2020-2024): within_family + strobe_mr")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
