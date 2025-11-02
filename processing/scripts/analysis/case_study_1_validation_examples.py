"""Generate validation example briefs for Case Study 1 canonical pairs.

This script creates detailed Markdown briefs for canonical trait pairs
defined in the configuration, as well as top concordant and discordant
examples. Each brief includes study details, reproducibility metrics,
and temporal context.

Input:
    - data/processed/case-study-cs1/metrics/
        pair_reproducibility_metrics.csv
    - config/case_studies.yml

Output:
    - .notes/analysis-notes/case-study-analysis/cs1-validation/
        canonical_pairs/
        top_concordant/
        top_discordant/
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from loguru import logger

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


def sanitize_filename(text: str) -> str:
    """Convert text to safe filename.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized filename string
    """
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    res = text.strip("_")
    return res


def generate_pair_brief(
    pair_data: pd.Series, output_path: Path, pair_type: str
) -> None:
    """Generate a Markdown brief for a single trait pair.

    Args:
        pair_data: Series containing pair metrics
        output_path: Path to write Markdown file
        pair_type: Type of pair (canonical, concordant, discordant)
    """
    brief_content = f"""# {pair_type.title()} Pair Validation Brief

## Study Information

- **PMID**: {pair_data["study1_pmid"]}
- **Model**: {pair_data["study1_model"]}
- **Title**: {pair_data["title"]}
- **Publication Year**: {int(pair_data["publication_year"])}
- **Temporal Era**: {pair_data["temporal_era"]}

## Reproducibility Metrics

- **Comparison Count**: {int(pair_data["study_count"])}
- **Reproducibility Tier**: {pair_data["reproducibility_tier"]}
- **Study Count Band**: {pair_data["study_count_band"]}

### Direction Concordance Statistics

- **Mean**: {pair_data["mean_direction_concordance"]:.4f}
- **Median**: {pair_data["median_direction_concordance"]:.4f}
- **Min**: {pair_data["min_direction_concordance"]:.4f}
- **Max**: {pair_data["max_direction_concordance"]:.4f}
- **Std**: {pair_data["std_direction_concordance"]:.4f}
- **Variance**: {pair_data["concordance_variance"]:.4f}

## Match Type Profile

- **Exact Matches**: {pair_data["has_exact_match"]}
- **Fuzzy Matches**: {pair_data["has_fuzzy_match"]}
- **EFO Matches**: {pair_data["has_efo_match"]}
- **Total Matched Pairs**: {int(pair_data["total_matched_pairs"])}

## Interpretation

"""

    if pair_data["mean_direction_concordance"] >= 0.7:
        brief_content += """This pair demonstrates high reproducibility across
independent studies with strong directional consistency. The findings
suggest robust evidence for the causal relationship.
"""
    elif pair_data["mean_direction_concordance"] >= 0.3:
        brief_content += """This pair shows moderate reproducibility with some
variability in effect directions. Additional investigation of study
heterogeneity may be warranted.
"""
    elif pair_data["mean_direction_concordance"] >= 0.0:
        brief_content += """This pair exhibits low reproducibility with
substantial directional inconsistency. Potential sources of heterogeneity
should be investigated.
"""
    else:
        brief_content += """This pair shows discordant findings across studies
with opposing effect directions. This may indicate context-dependent effects
or methodological differences.
"""

    # ---- Write brief to file ----
    with open(output_path, "w") as f:
        f.write(brief_content)

    logger.info(f"Generated brief: {output_path.name}")


def find_canonical_pairs(
    metrics_df: pd.DataFrame, canonical_list: List[List[str]]
) -> List[pd.Series]:
    """Find canonical trait pairs in the metrics data.

    Args:
        metrics_df: DataFrame with pair metrics
        canonical_list: List of canonical trait pair patterns

    Returns:
        List of Series for matched canonical pairs
    """
    logger.info(f"Searching for {len(canonical_list)} canonical pairs...")

    matched_pairs = []

    for exposure, outcome in canonical_list:
        # ---- Create case-insensitive pattern match ----
        exp_pattern = re.compile(re.escape(exposure), re.IGNORECASE)
        out_pattern = re.compile(re.escape(outcome), re.IGNORECASE)

        # ---- Search in title field ----
        title_matches = metrics_df[
            metrics_df["title"].str.contains(exp_pattern, na=False)
            & metrics_df["title"].str.contains(out_pattern, na=False)
        ]

        if len(title_matches) > 0:
            # ---- Take highest reproducibility match ----
            best_match = title_matches.nlargest(
                1, "mean_direction_concordance"
            ).iloc[0]
            matched_pairs.append(best_match)
            logger.info(f"Found canonical pair: {exposure} -> {outcome}")
        else:
            logger.warning(
                f"No match for canonical pair: {exposure} -> {outcome}"
            )

    logger.info(f"Matched {len(matched_pairs)} canonical pairs")
    res = matched_pairs
    return res


def select_top_examples(
    metrics_df: pd.DataFrame, n: int, high: bool
) -> List[pd.Series]:
    """Select top concordant or discordant examples.

    Args:
        metrics_df: DataFrame with pair metrics
        n: Number of examples to select
        high: If True, select high concordance; else select discordant

    Returns:
        List of Series for selected examples
    """
    if high:
        logger.info(f"Selecting top {n} concordant pairs...")
        sorted_df = metrics_df.nlargest(n, "mean_direction_concordance")
    else:
        logger.info(f"Selecting top {n} discordant pairs...")
        sorted_df = metrics_df.nsmallest(n, "mean_direction_concordance")

    res = [row for _, row in sorted_df.iterrows()]
    return res


def main() -> None:
    """Main execution function."""
    args = parse_args()

    logger.info(f"Loading configuration from: {args.config.resolve()}")
    config = load_config(args.config)

    validation_config = config["case_study_1"]["validation"]
    notes_base = PROJECT_ROOT / config["notes"]["case_study_1"]
    metrics_dir = PROJECT_ROOT / config["output"]["case_study_1"]["metrics"]

    if args.dry_run:
        logger.info("[DRY RUN] Would generate validation briefs")
        logger.info(
            f"[DRY RUN] Input: {metrics_dir / 'pair_reproducibility_metrics.csv'}"
        )
        logger.info(f"[DRY RUN] Output: {notes_base}/")
        logger.info(
            f"[DRY RUN] Canonical pairs: {len(validation_config['canonical_pairs'])}"
        )
        return

    # ---- Create output directories ----
    canonical_dir = notes_base / "canonical_pairs"
    concordant_dir = notes_base / "top_concordant"
    discordant_dir = notes_base / "top_discordant"

    for dir_path in [canonical_dir, concordant_dir, discordant_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {notes_base.resolve()}")

    # ---- Load metrics data ----
    input_path = metrics_dir / "pair_reproducibility_metrics.csv"
    logger.info(f"Loading metrics from: {input_path.resolve()}")
    metrics_df = pd.read_csv(input_path)

    logger.info(f"Loaded {len(metrics_df)} pair metrics")

    # ---- Generate canonical pair briefs ----
    canonical_pairs = find_canonical_pairs(
        metrics_df, validation_config["canonical_pairs"]
    )

    for i, pair_data in enumerate(canonical_pairs, 1):
        filename = f"canonical_{i:02d}_{sanitize_filename(pair_data['title'][:50])}.md"
        output_path = canonical_dir / filename
        generate_pair_brief(pair_data, output_path, "canonical")

    # ---- Generate top concordant briefs ----
    n_concordant = validation_config["n_top_concordant"]
    concordant_pairs = select_top_examples(metrics_df, n_concordant, high=True)

    for i, pair_data in enumerate(concordant_pairs, 1):
        filename = f"concordant_{i:02d}_{sanitize_filename(pair_data['title'][:50])}.md"
        output_path = concordant_dir / filename
        generate_pair_brief(pair_data, output_path, "concordant")

    # ---- Generate top discordant briefs ----
    n_discordant = validation_config["n_top_discordant"]
    discordant_pairs = select_top_examples(
        metrics_df, n_discordant, high=False
    )

    for i, pair_data in enumerate(discordant_pairs, 1):
        filename = f"discordant_{i:02d}_{sanitize_filename(pair_data['title'][:50])}.md"
        output_path = discordant_dir / filename
        generate_pair_brief(pair_data, output_path, "discordant")

    # ---- Summary output ----
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION BRIEFS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Canonical pairs: {len(canonical_pairs)}")
    logger.info(f"Top concordant: {len(concordant_pairs)}")
    logger.info(f"Top discordant: {len(discordant_pairs)}")
    logger.info(
        f"Total briefs generated: {len(canonical_pairs) + len(concordant_pairs) + len(discordant_pairs)}"
    )
    logger.info("=" * 60)
    logger.info("\nValidation brief generation complete!")


if __name__ == "__main__":
    main()
