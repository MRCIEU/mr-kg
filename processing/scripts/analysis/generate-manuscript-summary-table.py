"""Generate LaTeX-formatted summary tables for manuscript.

This script consolidates statistics from all three MR-KG databases
(vector store, trait profile, evidence profile) to generate publication-ready
LaTeX tables for manuscript inclusion.

Inputs:
- Overall database statistics (from generate-overall-database-stats.py)
- Trait profile statistics (from analyze-trait-summary-stats.py)
- Evidence profile statistics (from analyze-evidence-summary-stats.py)

Outputs:
- summary-table-full.tex: Aggregated table across all models
- summary-table-full-{model}.tex: Per-model tables (6 files)
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
    completeness_file = evidence_dir / "completeness-by-model.csv"

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
        "completeness": pd.read_csv(completeness_file)
        if completeness_file.exists()
        else None,
    }
    return res


def load_model_specific_stats(overall_dir: Path) -> pd.DataFrame:
    """Load per-model statistics from model-statistics.csv.

    Args:
        overall_dir: Directory containing overall statistics

    Returns:
        DataFrame with per-model statistics or None if file doesn't exist
    """
    csv_file = overall_dir / "model-statistics.csv"
    if csv_file.exists():
        res = pd.read_csv(csv_file)
        return res
    return None


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


# ==== Per-model table generation functions ====


def get_available_models(
    trait_stats: Dict[str, pd.DataFrame],
    evidence_stats: Dict[str, pd.DataFrame],
) -> List[str]:
    """Get list of available models from trait and evidence statistics.

    Args:
        trait_stats: Trait profile statistics
        evidence_stats: Evidence profile statistics

    Returns:
        List of unique model names sorted alphabetically
    """
    trait_models = set(trait_stats["model_stats"]["model"].unique())
    evidence_models = set(evidence_stats["model_stats"]["model"].unique())

    all_models = trait_models | evidence_models
    res = sorted(list(all_models))
    return res


def filter_model_statistics(
    model: str,
    overall_stats: Dict[str, Any],
    trait_stats: Dict[str, pd.DataFrame],
    evidence_stats: Dict[str, pd.DataFrame],
    model_specific_stats: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Filter statistics for a specific model.

    Args:
        model: Model name to filter
        overall_stats: Overall database statistics
        trait_stats: Trait profile statistics
        evidence_stats: Evidence profile statistics
        model_specific_stats: Per-model statistics from model-statistics.csv

    Returns:
        Dictionary with filtered statistics for the model
    """
    res = {
        "model": model,
        "overall": {},
        "trait": {},
        "evidence": {},
    }

    # ---- Extract model-specific overall stats ----

    if model_specific_stats is not None:
        model_row = model_specific_stats[
            model_specific_stats["model"] == model
        ]
        if not model_row.empty:
            res["overall"] = {
                "unique_pmids": int(model_row["unique_pmids"].iloc[0]),
                "extraction_count": int(model_row["extraction_count"].iloc[0]),
                "total_results": int(
                    model_row["total_results_extracted"].iloc[0]
                ),
                "avg_results_per_extraction": float(
                    model_row["avg_results_per_extraction"].iloc[0]
                ),
                "total_traits": int(
                    model_row["total_traits_extracted"].iloc[0]
                ),
            }

    # ---- Filter trait statistics ----

    trait_model_stats = trait_stats["model_stats"][
        trait_stats["model_stats"]["model"] == model
    ]
    trait_distributions = trait_stats["distributions"][
        trait_stats["distributions"]["model"] == model
    ]

    res["trait"]["model_stats"] = trait_model_stats
    res["trait"]["distributions"] = trait_distributions

    if trait_stats["correlations"] is not None:
        trait_correlations = trait_stats["correlations"][
            trait_stats["correlations"]["model"] == model
        ]
        res["trait"]["correlations"] = trait_correlations
    else:
        res["trait"]["correlations"] = None

    # ---- Filter evidence statistics ----

    evidence_model_stats = evidence_stats["model_stats"][
        evidence_stats["model_stats"]["model"] == model
    ]
    evidence_distributions = evidence_stats["distributions"][
        evidence_stats["distributions"]["model"] == model
    ]

    res["evidence"]["model_stats"] = evidence_model_stats
    res["evidence"]["distributions"] = evidence_distributions

    if evidence_stats["correlations"] is not None:
        evidence_correlations = evidence_stats["correlations"][
            evidence_stats["correlations"]["model"] == model
        ]
        res["evidence"]["correlations"] = evidence_correlations
    else:
        res["evidence"]["correlations"] = None

    if evidence_stats["completeness"] is not None:
        evidence_completeness = evidence_stats["completeness"][
            evidence_stats["completeness"]["model"] == model
        ]
        res["evidence"]["completeness"] = evidence_completeness
    else:
        res["evidence"]["completeness"] = None

    return res


def format_model_name(model: str) -> str:
    """Format model name for display in table captions.

    Args:
        model: Raw model name (e.g., "gpt-4-1")

    Returns:
        Formatted model name (e.g., "GPT-4-1")
    """
    if model.startswith("gpt"):
        res = model.upper()
    elif model.startswith("deepseek"):
        parts = model.split("-")
        res = "-".join([p.capitalize() for p in parts])
    elif model.startswith("llama"):
        res = model.replace("llama", "LLaMA-")
    elif model.startswith("o4"):
        res = model.upper()
    else:
        res = model.capitalize()

    return res


def generate_model_overall_section(
    model: str, model_data: Dict[str, Any]
) -> List[str]:
    """Generate LaTeX rows for model-specific overall characteristics.

    Args:
        model: Model name
        model_data: Filtered model statistics

    Returns:
        List of LaTeX table row strings
    """
    formatted_name = format_model_name(model)
    res = [
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Model: {formatted_name}}}}} \\\\",
        "\\hline",
    ]

    overall = model_data["overall"]
    if overall:
        res.extend(
            [
                f"Papers processed (PMIDs) & {overall['unique_pmids']:,} \\\\",
                f"Total extraction results & {overall['total_results']:,} \\\\",
                f"Unique traits extracted & {overall['total_traits']:,} \\\\",
                f"Average results per paper & "
                f"{overall['avg_results_per_extraction']:.2f} \\\\",
            ]
        )
    else:
        res.extend(
            [
                "Papers processed (PMIDs) & N/A \\\\",
                "Total extraction results & N/A \\\\",
                "Unique traits extracted & N/A \\\\",
                "Average results per paper & N/A \\\\",
            ]
        )

    return res


def generate_model_trait_section(
    model_data: Dict[str, Any],
) -> List[str]:
    """Generate LaTeX rows for model-specific trait profile characteristics.

    Args:
        model_data: Filtered model statistics

    Returns:
        List of LaTeX table row strings
    """
    model_stats = model_data["trait"]["model_stats"]
    distributions = model_data["trait"]["distributions"]
    correlations = model_data["trait"]["correlations"]

    res = [
        "\\hline",
        "\\multicolumn{2}{l}{\\textbf{Trait Profile Similarity}} \\\\",
        "\\hline",
    ]

    if model_stats.empty or distributions.empty:
        res.extend(
            [
                "Total PMID-model combinations & N/A \\\\",
                "Total pairwise comparisons & N/A \\\\",
                "Semantic similarity (mean) & N/A \\\\",
                "Semantic similarity (median) & N/A \\\\",
                "Jaccard similarity (mean) & N/A \\\\",
                "Jaccard similarity (median) & N/A \\\\",
            ]
        )
        return res

    total_combinations = int(model_stats["total_combinations"].iloc[0])
    total_pairs = int(model_stats["total_similarity_pairs"].iloc[0])

    avg_semantic = float(distributions["mean_semantic_similarity"].iloc[0])
    median_semantic = float(
        distributions["median_semantic_similarity"].iloc[0]
    )
    avg_jaccard = float(distributions["mean_jaccard_similarity"].iloc[0])
    median_jaccard = float(distributions["median_jaccard_similarity"].iloc[0])

    res.extend(
        [
            f"Total PMID-model combinations & {total_combinations:,} \\\\",
            f"Total pairwise comparisons & {total_pairs:,} \\\\",
            f"Semantic similarity (mean) & {avg_semantic:.3f} \\\\",
            f"Semantic similarity (median) & {median_semantic:.3f} \\\\",
            f"Jaccard similarity (mean) & {avg_jaccard:.3f} \\\\",
            f"Jaccard similarity (median) & {median_jaccard:.3f} \\\\",
        ]
    )

    if (
        correlations is not None
        and not correlations.empty
        and "corr_semantic_jaccard" in correlations.columns
    ):
        avg_corr = float(correlations["corr_semantic_jaccard"].iloc[0])
        res.append(f"Semantic-Jaccard correlation & {avg_corr:.3f} \\\\")

    return res


def generate_model_evidence_section(
    model_data: Dict[str, Any],
) -> List[str]:
    """Generate LaTeX rows for model-specific evidence profile characteristics.

    Args:
        model_data: Filtered model statistics

    Returns:
        List of LaTeX table row strings
    """
    model_stats = model_data["evidence"]["model_stats"]
    distributions = model_data["evidence"]["distributions"]
    completeness = model_data["evidence"]["completeness"]

    res = [
        "\\hline",
        "\\multicolumn{2}{l}{\\textbf{Evidence Profile Similarity}} \\\\",
        "\\hline",
    ]

    if model_stats.empty or distributions.empty:
        res.extend(
            [
                "Total PMID-model combinations & N/A \\\\",
                "Total pairwise comparisons & N/A \\\\",
                "Direction concordance (mean) & N/A \\\\",
                "Direction concordance (median) & N/A \\\\",
                "Composite similarity (mean) & N/A \\\\",
                "Composite similarity (median) & N/A \\\\",
                "Data completeness (mean) & N/A \\\\",
            ]
        )
        return res

    total_combinations = int(model_stats["total_combinations"].iloc[0])
    total_pairs = int(model_stats["total_similarity_pairs"].iloc[0])

    avg_direction = float(distributions["mean_direction_concordance"].iloc[0])
    median_direction = float(
        distributions["median_direction_concordance"].iloc[0]
    )
    avg_composite = float(distributions["mean_composite_direction"].iloc[0])
    median_composite = float(
        distributions["median_composite_direction"].iloc[0]
    )

    res.extend(
        [
            f"Total PMID-model combinations & {total_combinations:,} \\\\",
            f"Total pairwise comparisons & {total_pairs:,} \\\\",
            f"Direction concordance (mean) & {avg_direction:.3f} \\\\",
            f"Direction concordance (median) & {median_direction:.3f} \\\\",
            f"Composite similarity (mean) & {avg_composite:.3f} \\\\",
            f"Composite similarity (median) & {median_composite:.3f} \\\\",
        ]
    )

    if completeness is not None and not completeness.empty:
        mean_completeness = float(completeness["mean_completeness"].iloc[0])
        res.append(f"Data completeness (mean) & {mean_completeness:.3f} \\\\")

    return res


def generate_model_latex_table(model: str, model_data: Dict[str, Any]) -> str:
    """Generate complete LaTeX table for a specific model.

    Args:
        model: Model name
        model_data: Filtered statistics for this model

    Returns:
        Complete LaTeX table code as string
    """
    formatted_name = format_model_name(model)
    label = f"tab:mrkg-summary-{model}"

    header = [
        f"% MR-KG Summary Statistics Table - Model: {model}",
        "% Generated by generate-manuscript-summary-table.py",
        "% Customizable: adjust column widths, fonts, spacing as needed",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{MR-KG Summary Statistics - {formatted_name} Model}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lr}",
        "\\hline",
    ]

    overall_rows = generate_model_overall_section(model, model_data)
    trait_rows = generate_model_trait_section(model_data)
    evidence_rows = generate_model_evidence_section(model_data)

    footer = [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]

    res = "\n".join(
        header + overall_rows + trait_rows + evidence_rows + footer
    )
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

    # ---- Load model-specific statistics ----

    logger.info("Loading model-specific statistics...")
    model_specific_stats = load_model_specific_stats(args.overall_dir)
    if model_specific_stats is not None:
        logger.info(
            f"Loaded model-specific stats: {len(model_specific_stats)} models"
        )
    else:
        logger.warning("Model-specific stats not found, using defaults")

    # ---- Generate aggregated full LaTeX table ----

    logger.info("\nGenerating aggregated LaTeX table...")
    full_table = generate_full_latex_table(
        overall_stats, trait_stats, evidence_stats
    )
    output_file = args.output_dir / "summary-table-full.tex"
    with open(output_file, "w") as f:
        f.write(full_table)
    logger.info(f"Saved aggregated table: {output_file}")

    # ---- Generate per-model LaTeX tables ----

    logger.info("\nGenerating per-model LaTeX tables...")
    models = get_available_models(trait_stats, evidence_stats)
    logger.info(f"Found {len(models)} models: {', '.join(models)}")

    for model in models:
        logger.info(f"Generating table for model: {model}")

        try:
            # Filter statistics for this model
            model_data = filter_model_statistics(
                model,
                overall_stats,
                trait_stats,
                evidence_stats,
                model_specific_stats,
            )

            # Generate LaTeX table
            model_table = generate_model_latex_table(model, model_data)

            # Save to file
            output_file = args.output_dir / f"summary-table-full-{model}.tex"
            with open(output_file, "w") as f:
                f.write(model_table)
            logger.info(f"Saved model table: {output_file}")

        except Exception as e:
            logger.error(f"Failed to generate table for {model}: {e}")
            continue

    # ---- Generate JSON data ----

    logger.info("\nConsolidating data to JSON...")
    consolidated_data = consolidate_data_for_json(
        overall_stats, trait_stats, evidence_stats
    )
    output_file = args.output_dir / "summary-table-data.json"
    with open(output_file, "w") as f:
        json.dump(consolidated_data, f, indent=2)
    logger.info(f"Saved JSON data: {output_file}")

    # ---- Summary ----

    logger.info("\nTable generation complete!")
    logger.info(
        f"Generated 1 aggregated table + {len(models)} per-model tables"
    )
    logger.info(f"All outputs saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
