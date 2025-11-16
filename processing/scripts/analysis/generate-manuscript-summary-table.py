"""Generate LaTeX-formatted summary tables for manuscript.

This script consolidates statistics from all three MR-KG databases
(vector store, trait profile, evidence profile) to generate publication-ready
LaTeX tables for manuscript inclusion.

Inputs:
- Overall database statistics (from generate-overall-database-stats.py)
- Trait profile statistics (from analyze-trait-summary-stats.py)
- Evidence profile statistics (from analyze-evidence-summary-stats.py)

Outputs:
- summary-table-full.tex: Complete table with all statistics
- summary-table-compact.tex: Condensed table for main text
- summary-table-data.json: Raw data for custom table generation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from loguru import logger
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_OVERALL_DIR = PROCESSED_DIR / "overall-stats"
DEFAULT_TRAIT_DIR = PROCESSED_DIR / "trait-profiles" / "analysis"
DEFAULT_EVIDENCE_DIR = PROCESSED_DIR / "evidence-profiles" / "analysis"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR / "manuscript-tables"


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
        help="Perform dry run without executing generation",
    )

    # ---- --overall-dir ----
    parser.add_argument(
        "--overall-dir",
        type=Path,
        default=DEFAULT_OVERALL_DIR,
        help=f"Directory with overall stats (default: {DEFAULT_OVERALL_DIR})",
    )

    # ---- --trait-dir ----
    parser.add_argument(
        "--trait-dir",
        type=Path,
        default=DEFAULT_TRAIT_DIR,
        help=f"Directory with trait stats (default: {DEFAULT_TRAIT_DIR})",
    )

    # ---- --evidence-dir ----
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help=f"Directory with evidence stats (default: {DEFAULT_EVIDENCE_DIR})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for tables (default: {DEFAULT_OUTPUT_DIR})",
    )

    res = parser.parse_args()
    return res


# ==== Data loading functions ====


def load_overall_statistics(overall_dir: Path) -> Dict[str, Any]:
    """Load overall database statistics.

    Args:
        overall_dir: Directory containing overall statistics

    Returns:
        Dictionary with overall statistics
    """
    json_file = overall_dir / "database-summary.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Overall statistics not found: {json_file}")

    with open(json_file, "r") as f:
        res = json.load(f)
    return res


def load_trait_statistics(trait_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load trait profile statistics.

    Args:
        trait_dir: Directory containing trait statistics

    Returns:
        Dictionary with DataFrames for different statistics
    """
    model_stats_file = trait_dir / "summary-stats-by-model.csv"
    distributions_file = trait_dir / "similarity-distributions.csv"
    correlations_file = trait_dir / "metric-correlations.csv"

    if not model_stats_file.exists():
        raise FileNotFoundError(
            f"Trait model stats not found: {model_stats_file}"
        )
    if not distributions_file.exists():
        raise FileNotFoundError(
            f"Trait distributions not found: {distributions_file}"
        )

    res = {
        "model_stats": pd.read_csv(model_stats_file),
        "distributions": pd.read_csv(distributions_file),
        "correlations": pd.read_csv(correlations_file)
        if correlations_file.exists()
        else None,
    }
    return res


def load_evidence_statistics(evidence_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load evidence profile statistics.

    Args:
        evidence_dir: Directory containing evidence statistics

    Returns:
        Dictionary with DataFrames for different statistics
    """
    model_stats_file = evidence_dir / "summary-stats-by-model.csv"
    distributions_file = evidence_dir / "similarity-distributions.csv"
    correlations_file = evidence_dir / "metric-correlations.csv"

    if not model_stats_file.exists():
        raise FileNotFoundError(
            f"Evidence model stats not found: {model_stats_file}"
        )
    if not distributions_file.exists():
        raise FileNotFoundError(
            f"Evidence distributions not found: {distributions_file}"
        )

    res = {
        "model_stats": pd.read_csv(model_stats_file),
        "distributions": pd.read_csv(distributions_file),
        "correlations": pd.read_csv(correlations_file)
        if correlations_file.exists()
        else None,
    }
    return res


# ==== Table generation functions ====


def generate_overall_section(stats: Dict[str, Any]) -> List[str]:
    """Generate LaTeX rows for overall MR-KG characteristics.

    Args:
        stats: Overall statistics dictionary

    Returns:
        List of LaTeX table row strings
    """
    res = [
        "\\multicolumn{2}{l}{\\textbf{Overall MR-KG Characteristics}} \\\\",
        "\\hline",
        f"Total unique papers (PMIDs) & {stats['total_unique_pmids']:,} \\\\",
        f"Total unique traits & {stats['total_unique_traits']:,} \\\\",
        f"Total model extraction records & {stats['total_model_results']:,} \\\\",
        f"Number of extraction models & {stats['total_unique_models']} \\\\",
        f"Temporal coverage & "
        f"{stats['temporal_range_start']}--{stats['temporal_range_end']} \\\\",
        f"Average results per paper & "
        f"{stats['avg_results_per_pmid']:.2f} \\\\",
    ]
    return res


def generate_trait_section(
    trait_stats: Dict[str, pd.DataFrame],
) -> List[str]:
    """Generate LaTeX rows for trait profile characteristics.

    Args:
        trait_stats: Dictionary with trait statistics DataFrames

    Returns:
        List of LaTeX table row strings
    """
    model_stats = trait_stats["model_stats"]
    distributions = trait_stats["distributions"]

    total_combinations = model_stats["total_combinations"].sum()
    total_pairs = model_stats["total_similarity_pairs"].sum()

    avg_semantic = distributions["mean_semantic_similarity"].mean()
    median_semantic = distributions["median_semantic_similarity"].median()

    avg_jaccard = distributions["mean_jaccard_similarity"].mean()
    median_jaccard = distributions["median_jaccard_similarity"].median()

    res = [
        "\\hline",
        "\\multicolumn{2}{l}{\\textbf{Trait Profile Similarity}} \\\\",
        "\\hline",
        f"Total PMID-model combinations & {total_combinations:,} \\\\",
        f"Total pairwise comparisons & {total_pairs:,} \\\\",
        f"Semantic similarity (mean) & {avg_semantic:.3f} \\\\",
        f"Semantic similarity (median) & {median_semantic:.3f} \\\\",
        f"Jaccard similarity (mean) & {avg_jaccard:.3f} \\\\",
        f"Jaccard similarity (median) & {median_jaccard:.3f} \\\\",
    ]

    if trait_stats["correlations"] is not None:
        correlations = trait_stats["correlations"]
        avg_corr = correlations["corr_semantic_jaccard"].mean()
        res.append(f"Semantic-Jaccard correlation & {avg_corr:.3f} \\\\")

    return res


def generate_evidence_section(
    evidence_stats: Dict[str, pd.DataFrame],
) -> List[str]:
    """Generate LaTeX rows for evidence profile characteristics.

    Args:
        evidence_stats: Dictionary with evidence statistics DataFrames

    Returns:
        List of LaTeX table row strings
    """
    model_stats = evidence_stats["model_stats"]
    distributions = evidence_stats["distributions"]

    total_combinations = model_stats["total_combinations"].sum()
    total_pairs = model_stats["total_similarity_pairs"].sum()

    avg_direction = distributions["mean_direction_concordance"].mean()
    median_direction = distributions["median_direction_concordance"].median()

    avg_composite = distributions["mean_composite_direction"].mean()
    median_composite = distributions["median_composite_direction"].median()

    res = [
        "\\hline",
        "\\multicolumn{2}{l}{\\textbf{Evidence Profile Similarity}} \\\\",
        "\\hline",
        f"Total PMID-model combinations & {total_combinations:,} \\\\",
        f"Total pairwise comparisons & {total_pairs:,} \\\\",
        f"Direction concordance (mean) & {avg_direction:.3f} \\\\",
        f"Direction concordance (median) & {median_direction:.3f} \\\\",
        f"Composite similarity (mean) & {avg_composite:.3f} \\\\",
        f"Composite similarity (median) & {median_composite:.3f} \\\\",
    ]

    return res


def generate_full_latex_table(
    overall_stats: Dict[str, Any],
    trait_stats: Dict[str, pd.DataFrame],
    evidence_stats: Dict[str, pd.DataFrame],
) -> str:
    """Generate complete LaTeX table with all statistics.

    Args:
        overall_stats: Overall database statistics
        trait_stats: Trait profile statistics
        evidence_stats: Evidence profile statistics

    Returns:
        Complete LaTeX table code as string
    """
    header = [
        "% MR-KG Summary Statistics Table (Full Version)",
        "% Generated by generate-manuscript-summary-table.py",
        "% Customizable: adjust column widths, fonts, spacing as needed",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{MR-KG Database Summary Statistics}",
        "\\label{tab:mrkg-summary-full}",
        "\\begin{tabular}{lr}",
        "\\hline",
    ]

    overall_rows = generate_overall_section(overall_stats)
    trait_rows = generate_trait_section(trait_stats)
    evidence_rows = generate_evidence_section(evidence_stats)

    footer = [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]

    res = "\n".join(
        header + overall_rows + trait_rows + evidence_rows + footer
    )
    return res


def generate_compact_latex_table(
    overall_stats: Dict[str, Any],
    trait_stats: Dict[str, pd.DataFrame],
    evidence_stats: Dict[str, pd.DataFrame],
) -> str:
    """Generate compact LaTeX table for main text.

    Args:
        overall_stats: Overall database statistics
        trait_stats: Trait profile statistics
        evidence_stats: Evidence profile statistics

    Returns:
        Compact LaTeX table code as string
    """
    model_stats_trait = trait_stats["model_stats"]
    distributions_trait = trait_stats["distributions"]
    model_stats_evidence = evidence_stats["model_stats"]
    distributions_evidence = evidence_stats["distributions"]

    header = [
        "% MR-KG Summary Statistics Table (Compact Version)",
        "% Generated by generate-manuscript-summary-table.py",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{MR-KG Database Characteristics}",
        "\\label{tab:mrkg-summary-compact}",
        "\\begin{tabular}{lr}",
        "\\hline",
    ]

    rows = [
        f"Papers (PMIDs) & {overall_stats['total_unique_pmids']:,} \\\\",
        f"Unique traits & {overall_stats['total_unique_traits']:,} \\\\",
        f"Temporal span & "
        f"{overall_stats['temporal_range_start']}--"
        f"{overall_stats['temporal_range_end']} \\\\",
        "\\hline",
        f"Trait comparisons & "
        f"{model_stats_trait['total_similarity_pairs'].sum():,} \\\\",
        f"Evidence comparisons & "
        f"{model_stats_evidence['total_similarity_pairs'].sum():,} \\\\",
        "\\hline",
    ]

    footer = [
        "\\end{tabular}",
        "\\end{table}",
    ]

    res = "\n".join(header + rows + footer)
    return res


def consolidate_data_for_json(
    overall_stats: Dict[str, Any],
    trait_stats: Dict[str, pd.DataFrame],
    evidence_stats: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """Consolidate all statistics into JSON-serializable structure.

    Args:
        overall_stats: Overall database statistics
        trait_stats: Trait profile statistics
        evidence_stats: Evidence profile statistics

    Returns:
        Dictionary with consolidated statistics
    """
    res = {
        "overall": overall_stats,
        "trait_profile": {
            "model_stats": trait_stats["model_stats"].to_dict(
                orient="records"
            ),
            "distributions_summary": {
                "mean_semantic": float(
                    trait_stats["distributions"][
                        "mean_semantic_similarity"
                    ].mean()
                ),
                "median_semantic": float(
                    trait_stats["distributions"][
                        "median_semantic_similarity"
                    ].median()
                ),
                "mean_jaccard": float(
                    trait_stats["distributions"][
                        "mean_jaccard_similarity"
                    ].mean()
                ),
                "median_jaccard": float(
                    trait_stats["distributions"][
                        "median_jaccard_similarity"
                    ].median()
                ),
            },
        },
        "evidence_profile": {
            "model_stats": evidence_stats["model_stats"].to_dict(
                orient="records"
            ),
            "distributions_summary": {
                "mean_direction": float(
                    evidence_stats["distributions"][
                        "mean_direction_concordance"
                    ].mean()
                ),
                "median_direction": float(
                    evidence_stats["distributions"][
                        "median_direction_concordance"
                    ].median()
                ),
                "mean_composite": float(
                    evidence_stats["distributions"][
                        "mean_composite_direction"
                    ].mean()
                ),
                "median_composite": float(
                    evidence_stats["distributions"][
                        "median_composite_direction"
                    ].median()
                ),
            },
        },
    }
    return res


# ==== Main execution ====


def main():
    """Execute manuscript table generation."""
    args = make_args()

    # ---- Validate paths ----

    if args.dry_run:
        logger.info("Dry run - validating paths")
        paths_to_check = [
            (args.overall_dir, "Overall stats directory"),
            (args.trait_dir, "Trait stats directory"),
            (args.evidence_dir, "Evidence stats directory"),
        ]
        for path, desc in paths_to_check:
            if not path.exists():
                logger.error(f"{desc} not found: {path}")
                return 1
            logger.info(f"{desc}: {path}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("Dry run complete - paths validated")
        return 0

    # ---- Setup ----

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # ---- Load statistics ----

    logger.info("Loading overall statistics...")
    try:
        overall_stats = load_overall_statistics(args.overall_dir)
        logger.info(
            f"Loaded overall stats: {overall_stats['total_unique_pmids']} "
            "papers"
        )
    except Exception as e:
        logger.error(f"Failed to load overall statistics: {e}")
        return 1

    logger.info("Loading trait profile statistics...")
    try:
        trait_stats = load_trait_statistics(args.trait_dir)
        logger.info(
            f"Loaded trait stats: {len(trait_stats['model_stats'])} models"
        )
    except Exception as e:
        logger.error(f"Failed to load trait statistics: {e}")
        return 1

    logger.info("Loading evidence profile statistics...")
    try:
        evidence_stats = load_evidence_statistics(args.evidence_dir)
        logger.info(
            f"Loaded evidence stats: "
            f"{len(evidence_stats['model_stats'])} models"
        )
    except Exception as e:
        logger.error(f"Failed to load evidence statistics: {e}")
        return 1

    # ---- Generate full LaTeX table ----

    logger.info("\nGenerating full LaTeX table...")
    full_table = generate_full_latex_table(
        overall_stats, trait_stats, evidence_stats
    )
    output_file = args.output_dir / "summary-table-full.tex"
    with open(output_file, "w") as f:
        f.write(full_table)
    logger.info(f"Saved full table: {output_file}")

    # ---- Generate compact LaTeX table ----

    logger.info("Generating compact LaTeX table...")
    compact_table = generate_compact_latex_table(
        overall_stats, trait_stats, evidence_stats
    )
    output_file = args.output_dir / "summary-table-compact.tex"
    with open(output_file, "w") as f:
        f.write(compact_table)
    logger.info(f"Saved compact table: {output_file}")

    # ---- Generate JSON data ----

    logger.info("Consolidating data to JSON...")
    consolidated_data = consolidate_data_for_json(
        overall_stats, trait_stats, evidence_stats
    )
    output_file = args.output_dir / "summary-table-data.json"
    with open(output_file, "w") as f:
        json.dump(consolidated_data, f, indent=2)
    logger.info(f"Saved JSON data: {output_file}")

    # ---- Print preview ----

    logger.info("\n" + "=" * 60)
    logger.info("COMPACT TABLE PREVIEW:")
    logger.info("=" * 60)
    logger.info(compact_table)
    logger.info("=" * 60)

    logger.info("\nTable generation complete!")
    logger.info(f"All outputs saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
