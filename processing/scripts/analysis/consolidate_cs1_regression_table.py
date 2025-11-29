#!/usr/bin/env python3

"""
Generate regression coefficients table for Case Study 1.

Produces LaTeX-formatted table for the linear regression model predicting
direction concordance from study count, publication year, and match type.
This table documents the study count dilution effect (beta = -0.024).

Input:
    - data/processed/case-study-cs1/models/temporal_model_coefficients.csv
    - data/processed/case-study-cs1/models/temporal_model_diagnostics.json

Output:
    - data/artifacts/manuscript-tables/cs1_regression_coefficients.csv
    - data/artifacts/manuscript-tables/cs1_regression_coefficients.tex
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger
from yiutils.project_utils import find_project_root

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "case-study-cs1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "artifacts" / "manuscript-tables"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- --input-dir ----
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    res = parser.parse_args()
    return res


def load_coefficients(input_dir: Path) -> pd.DataFrame:
    """Load model coefficients from CSV.

    Args:
        input_dir: Directory containing model outputs

    Returns:
        DataFrame with model coefficients
    """
    coef_path = input_dir / "models" / "temporal_model_coefficients.csv"
    if not coef_path.exists():
        logger.error(f"Coefficients file not found: {coef_path}")
        sys.exit(1)

    res = pd.read_csv(coef_path)
    logger.info(f"Loaded coefficients from: {coef_path}")
    return res


def load_diagnostics(input_dir: Path) -> Dict[str, Any]:
    """Load model diagnostics from JSON.

    Args:
        input_dir: Directory containing model outputs

    Returns:
        Dictionary with model diagnostics
    """
    diag_path = input_dir / "models" / "temporal_model_diagnostics.json"
    if not diag_path.exists():
        logger.error(f"Diagnostics file not found: {diag_path}")
        sys.exit(1)

    with open(diag_path, "r") as f:
        res = json.load(f)
    logger.info(f"Loaded diagnostics from: {diag_path}")
    return res


def format_pvalue(p: float) -> str:
    """Format p-value for display.

    Args:
        p: P-value to format

    Returns:
        Formatted p-value string
    """
    if p < 0.001:
        return "<0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


def format_coefficient(coef: float, decimals: int = 3) -> str:
    """Format coefficient for display.

    Args:
        coef: Coefficient value
        decimals: Number of decimal places

    Returns:
        Formatted coefficient string
    """
    return f"{coef:.{decimals}f}"


def create_regression_table(
    coef_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
) -> pd.DataFrame:
    """Create formatted regression coefficients table.

    Args:
        coef_df: DataFrame with raw coefficients
        diagnostics: Dictionary with model diagnostics

    Returns:
        Formatted DataFrame for manuscript table
    """
    logger.info("Creating regression coefficients table...")

    # ---- Map variable names to readable labels ----
    variable_labels = {
        "const": "Intercept",
        "publication_year": "Publication year",
        "study_count": "Study count",
        "match_type_exact": "Match type (exact)",
    }

    # ---- Create formatted table ----
    table_data = []
    for _, row in coef_df.iterrows():
        var_name = row["variable"]
        label = variable_labels.get(var_name, var_name)

        # ---- Format 95% CI ----
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"

        table_data.append(
            {
                "Variable": label,
                "Coefficient": format_coefficient(row["coefficient"]),
                "Std. Error": format_coefficient(row["std_error"]),
                "95% CI": ci_str,
                "P-value": format_pvalue(row["p_value"]),
            }
        )

    table_df = pd.DataFrame(table_data)

    # ---- Order rows ----
    row_order = [
        "Intercept",
        "Publication year",
        "Study count",
        "Match type (exact)",
    ]
    table_df["sort_key"] = table_df["Variable"].apply(
        lambda x: row_order.index(x) if x in row_order else len(row_order)
    )
    table_df = table_df.sort_values("sort_key").drop(columns=["sort_key"])

    logger.info(f"Created table with {len(table_df)} rows")
    return table_df


def generate_latex_table(
    table_df: pd.DataFrame,
    diagnostics: Dict[str, Any],
) -> str:
    """Generate LaTeX table string.

    Args:
        table_df: Formatted table DataFrame
        diagnostics: Model diagnostics for footer

    Returns:
        LaTeX table string
    """
    # ---- Build LaTeX manually for precise control ----
    n_obs = diagnostics["n_observations"]
    r_sq = diagnostics["r_squared"]
    adj_r_sq = diagnostics["adj_r_squared"]
    f_stat = diagnostics["f_statistic"]
    f_pval = diagnostics["f_pvalue"]

    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variable & Coefficient & Std. Error & 95\\% CI & P-value \\\\",
        "\\midrule",
    ]

    for _, row in table_df.iterrows():
        line = (
            f"{row['Variable']} & "
            f"{row['Coefficient']} & "
            f"{row['Std. Error']} & "
            f"{row['95% CI']} & "
            f"{row['P-value']} \\\\"
        )
        lines.append(line)

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    # ---- Add footer with model diagnostics ----
    footer = (
        f"\\\\[0.5em]\n"
        f"\\footnotesize\\textit{{Note:}} "
        f"N = {n_obs:,} trait pairs. "
        f"$R^2$ = {r_sq:.4f}, "
        f"Adjusted $R^2$ = {adj_r_sq:.4f}. "
        f"F-statistic = {f_stat:.2f} (p {'<0.001' if f_pval < 0.001 else f'= {f_pval:.3f}'})."
    )

    latex_str = "\n".join(lines) + "\n" + footer
    return latex_str


def main() -> int:
    """Main function.

    Returns:
        Exit code (0 for success)
    """
    args = parse_args()

    # ---- Setup logging ----
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.dry_run:
        logger.info("[DRY RUN] Would process:")
        logger.info(f"  Input dir: {args.input_dir}")
        logger.info(f"  Output dir: {args.output_dir}")
        return 0

    # ---- Create output directory ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    coef_df = load_coefficients(args.input_dir)
    diagnostics = load_diagnostics(args.input_dir)

    # ---- Create table ----
    table_df = create_regression_table(coef_df, diagnostics)

    # ---- Save CSV ----
    csv_path = args.output_dir / "cs1_regression_coefficients.csv"
    table_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")

    # ---- Generate and save LaTeX ----
    latex_str = generate_latex_table(table_df, diagnostics)
    tex_path = args.output_dir / "cs1_regression_coefficients.tex"
    with open(tex_path, "w") as f:
        f.write(latex_str)
    logger.info(f"Saved LaTeX: {tex_path}")

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("REGRESSION TABLE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"N observations: {diagnostics['n_observations']:,}")
    logger.info(f"R-squared: {diagnostics['r_squared']:.4f}")
    logger.info(f"Adjusted R-squared: {diagnostics['adj_r_squared']:.4f}")
    logger.info("\nCoefficients:")
    for _, row in table_df.iterrows():
        logger.info(
            f"  {row['Variable']:25s}: "
            f"beta = {row['Coefficient']:>8s}, "
            f"p = {row['P-value']}"
        )
    logger.info("=" * 60)

    logger.info("\nOutput files:")
    logger.info(f"  {csv_path}")
    logger.info(f"  {tex_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
