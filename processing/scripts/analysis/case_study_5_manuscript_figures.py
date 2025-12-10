"""Generate manuscript-ready figures for Case Study 5 using Altair.

This script creates publication-quality faceted time series figures for CS5
temporal evolution analysis. All figures are exported as PNG for manuscript
inclusion.

Figures generated:
- Faceted time series plot with four subplots:
  A. MR Papers by Publication Year (log scale)
  B. Trait Diversity Over Time
  C. Reporting Completeness Over Time
  D. Reporting Completeness by Field Type

All subplots use year range 2003-2025 with era boundary markers.

Outputs:
- data/artifacts/manuscript-figures/cs5_fig_temporal_trends.png
"""

import argparse
from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
import yaml
from loguru import logger
from yiutils.project_utils import find_project_root

from color_schemes import GRUVBOX_DARK, get_field_type_colors

# ==== Project configuration ====

PROJECT_ROOT = find_project_root("docker-compose.yml")
CONFIG_DIR = PROJECT_ROOT / "processing" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "case_studies.yml"
MANUSCRIPT_FIGURES_DIR = (
    PROJECT_ROOT / "data" / "artifacts" / "manuscript-figures"
)

# Year range for all plots
YEAR_MIN = 2003
YEAR_MAX = 2025

# Plot dimensions (wider for time series)
PLOT_WIDTH = 900  # 1.5x wider than original 600
PLOT_HEIGHT = 200


# ==== Argument parsing ====


def parse_args():
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
        help="Perform dry run without generating figures",
    )

    # ---- --config ----
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Configuration file (default: {DEFAULT_CONFIG})",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MANUSCRIPT_FIGURES_DIR,
        help=(
            "Override output directory from default "
            f"({MANUSCRIPT_FIGURES_DIR})"
        ),
    )

    res = parser.parse_args()
    return res


# ==== Configuration loading ====


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    return config


# ==== Era processing utilities ====


def get_era_transitions(era_defs: Dict[str, List[int]]) -> List[Dict]:
    """Extract era transition years and labels.

    Args:
        era_defs: Dictionary mapping era names to [start_year, end_year]

    Returns:
        List of dicts with year, era label, and numeric label
    """
    transitions = []
    sorted_eras = sorted(era_defs.items(), key=lambda x: x[1][0])

    for idx, (era_name, years) in enumerate(sorted_eras, start=1):
        transitions.append(
            {"year": years[0], "era": era_name, "era_num": str(idx)}
        )

    return transitions


# ==== Data loading ====


def load_temporal_stats(data_dir: Path) -> pd.DataFrame:
    """Load temporal statistics from CSV.

    Args:
        data_dir: Data directory path

    Returns:
        DataFrame with publication year statistics
    """
    temporal_stats_file = (
        data_dir / "processed" / "overall-stats" / "temporal-statistics.csv"
    )

    if not temporal_stats_file.exists():
        msg = (
            f"Temporal statistics file not found: {temporal_stats_file}\n"
            "Run summary statistics pipeline first."
        )
        raise FileNotFoundError(msg)

    logger.info(f"Loading temporal statistics from: {temporal_stats_file}")
    df = pd.read_csv(temporal_stats_file)

    # ---- Remove outlier years ----
    df = df[df["publication_year"] <= 2025].copy()

    # ---- Sort and filter for log scale ----
    df = df.sort_values("publication_year")
    df = df[df["paper_count"] > 0].copy()

    logger.info(f"Loaded {len(df)} years of temporal data")

    res = df
    return res


def calculate_log_ticks(ymax: float) -> List[int]:
    """Calculate logarithmic tick values for y-axis.

    Generates ticks like 1, 3, 10, 30, 100, 300, etc.

    Args:
        ymax: Maximum y value

    Returns:
        List of tick values
    """
    log_ticks = []
    k = 0
    while (10**k) <= ymax * 1.2:
        base = 10**k
        for m in (1, 3):
            val = base * m
            if val >= 1:
                log_ticks.append(val)
        k += 1
    res = log_ticks
    return res


def load_diversity_data(cs5_output_dir: Path) -> pd.DataFrame:
    """Load trait diversity yearly statistics.

    Args:
        cs5_output_dir: CS5 output base directory

    Returns:
        DataFrame with trait diversity by year
    """
    diversity_dir = cs5_output_dir / "diversity"
    csv_path = diversity_dir / "trait_counts_by_year.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Diversity data not found: {csv_path}")

    logger.info(f"Loading trait diversity data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} years of diversity data")

    res = df
    return res


def load_completeness_data(cs5_output_dir: Path) -> pd.DataFrame:
    """Load reporting completeness yearly statistics.

    Args:
        cs5_output_dir: CS5 output base directory

    Returns:
        DataFrame with completeness by year
    """
    completeness_dir = cs5_output_dir / "completeness"
    csv_path = completeness_dir / "field_completeness_by_year.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Completeness data not found: {csv_path}")

    logger.info(f"Loading completeness data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} years of completeness data")

    res = df
    return res


# ==== Data preparation ====


def prepare_field_type_yearly_data(
    completeness_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare yearly field type completeness data.

    Args:
        completeness_df: DataFrame with yearly completeness statistics

    Returns:
        Long-format DataFrame with year, field_type, completeness_pct
    """
    logger.info("Preparing yearly field type data...")

    # ---- Field type mappings ----
    field_mapping = {
        "confidence_interval_pct": "Confidence Interval",
        "direction_pct": "Direction",
        "effect_size_beta_pct": "Effect Size",
        "p_value_pct": "Statistical",
    }

    # ---- Combine effect size columns ----
    completeness_df["effect_size_pct"] = completeness_df[
        ["effect_size_beta_pct", "effect_size_or_pct", "effect_size_hr_pct"]
    ].max(axis=1)

    # ---- Reshape to long format ----
    plot_data = []
    for _, row in completeness_df.iterrows():
        for field_col, field_label in field_mapping.items():
            if field_col == "effect_size_beta_pct":
                # Use combined effect size
                pct_val = row["effect_size_pct"]
            else:
                pct_val = row[field_col]

            plot_data.append(
                {
                    "pub_year": row["pub_year"],
                    "field_type": field_label,
                    "completeness_pct": pct_val,
                }
            )

    res = pd.DataFrame(plot_data)
    logger.info(f"Prepared {len(res)} field type x year combinations")
    return res


# ==== Subplot creation ====


def create_era_markers(
    era_transitions: List[Dict],
) -> alt.LayerChart:
    """Create era boundary markers with labels.

    Args:
        era_transitions: List of era transition dicts

    Returns:
        Altair layer with vertical lines and text labels
    """
    transitions_df = pd.DataFrame(era_transitions)

    # ---- Vertical lines ----
    lines = (
        alt.Chart(transitions_df)
        .mark_rule(color=GRUVBOX_DARK["red"], strokeWidth=1)
        .encode(x=alt.X("year:Q"))
    )

    # ---- Era labels (numeric) ----
    labels = (
        alt.Chart(transitions_df)
        .mark_text(
            align="left",
            dx=3,
            dy=5,
            fontSize=12,
            color=GRUVBOX_DARK["red"],
            fontWeight="bold",
        )
        .encode(
            x=alt.X("year:Q"),
            y=alt.value(0),
            text=alt.Text("era_num:N"),
        )
    )

    res = lines + labels
    return res


def create_temporal_distribution_subplot(
    temporal_df: pd.DataFrame,
    era_transitions: List[Dict],
) -> alt.Chart:
    """Create MR papers by publication year subplot with log scale.

    Args:
        temporal_df: DataFrame with publication_year and paper_count columns
        era_transitions: List of era transition dicts

    Returns:
        Altair chart for temporal distribution subplot
    """
    logger.info("Creating temporal distribution subplot...")

    ymax = float(temporal_df["paper_count"].max())
    log_ticks = calculate_log_ticks(ymax)

    # ---- Line chart with points ----
    line = (
        alt.Chart(temporal_df)
        .mark_line(point=True, color=GRUVBOX_DARK["blue"])
        .encode(
            x=alt.X(
                "publication_year:Q",
                title="Publication Year",
                scale=alt.Scale(domain=[YEAR_MIN, YEAR_MAX]),
                axis=alt.Axis(format="d", labelAngle=0),
            ),
            y=alt.Y(
                "paper_count:Q",
                title="Number of Papers (log scale)",
                scale=alt.Scale(type="log", base=10, nice=False, domainMin=1),
                axis=alt.Axis(values=log_ticks, format=","),
            ),
            tooltip=[
                alt.Tooltip("publication_year:Q", title="Year", format="d"),
                alt.Tooltip("paper_count:Q", title="Papers", format=","),
            ],
        )
    )

    # ---- Text labels for all points ----
    labels = (
        alt.Chart(temporal_df)
        .mark_text(
            align="center",
            dy=26,
            stroke="black",
            strokeWidth=1,
            fontSize=10,
        )
        .encode(
            x=alt.X("publication_year:Q"),
            y=alt.Y("paper_count:Q"),
            text=alt.Text("paper_count:Q", format=","),
        )
    )

    # ---- Era markers ----
    era_markers = create_era_markers(era_transitions)

    # ---- Combine layers ----
    combined = (line + labels + era_markers).properties(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        title="A. MR Papers by Publication Year",
    )

    return combined


def create_diversity_subplot(
    diversity_df: pd.DataFrame,
    era_transitions: List[Dict],
) -> alt.Chart:
    """Create trait diversity over time subplot.

    Args:
        diversity_df: DataFrame with yearly diversity statistics
        era_transitions: List of era transition dicts

    Returns:
        Altair chart for diversity subplot
    """
    logger.info("Creating diversity subplot...")

    # ---- Dashed line connecting points ----
    line = (
        alt.Chart(diversity_df)
        .mark_line(strokeDash=[5, 5], color=GRUVBOX_DARK["blue"])
        .encode(
            x=alt.X(
                "pub_year:Q",
                title="Publication Year",
                scale=alt.Scale(domain=[YEAR_MIN, YEAR_MAX]),
                axis=alt.Axis(format="d", labelAngle=0),
            ),
            y=alt.Y(
                "mean_trait_count:Q",
                title="Mean Traits per Study",
            ),
        )
    )

    # ---- Points ----
    points = (
        alt.Chart(diversity_df)
        .mark_point(size=60, color=GRUVBOX_DARK["blue"], filled=True)
        .encode(
            x=alt.X("pub_year:Q"),
            y=alt.Y("mean_trait_count:Q"),
            tooltip=[
                alt.Tooltip("pub_year:Q", title="Year", format="d"),
                alt.Tooltip(
                    "mean_trait_count:Q",
                    title="Mean Traits",
                    format=".2f",
                ),
                alt.Tooltip("n_studies:Q", title="Studies"),
            ],
        )
    )

    # ---- Error bands ----
    error_band = (
        alt.Chart(diversity_df)
        .mark_errorband(extent="stderr", opacity=0.3)
        .encode(
            x=alt.X("pub_year:Q"),
            y=alt.Y("mean_trait_count:Q"),
            yError=alt.YError("std_trait_count:Q"),
        )
    )

    # ---- Era markers ----
    era_markers = create_era_markers(era_transitions)

    # ---- Combine layers ----
    combined = (error_band + line + points + era_markers).properties(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        title="B. Trait Diversity Over Time",
    )

    return combined


def create_completeness_subplot(
    completeness_df: pd.DataFrame,
    era_transitions: List[Dict],
) -> alt.Chart:
    """Create completeness over time subplot.

    Args:
        completeness_df: DataFrame with yearly completeness statistics
        era_transitions: List of era transition dicts

    Returns:
        Altair chart for completeness subplot
    """
    logger.info("Creating completeness subplot...")

    # ---- Prepare data for key fields ----
    fields_to_plot = [
        "p_value_pct",
        "confidence_interval_pct",
        "standard_error_pct",
    ]
    field_labels = {
        "p_value_pct": "P-value",
        "confidence_interval_pct": "95% CI",
        "standard_error_pct": "SE",
    }

    plot_data = []
    for _, row in completeness_df.iterrows():
        for field in fields_to_plot:
            plot_data.append(
                {
                    "pub_year": row["pub_year"],
                    "field": field_labels[field],
                    "completeness_pct": row[field],
                }
            )

    plot_df = pd.DataFrame(plot_data)

    # ---- Get field type colors ----
    field_colors = get_field_type_colors(use_gruvbox=True)

    # ---- Main time series ----
    line = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "pub_year:Q",
                title="Publication Year",
                scale=alt.Scale(domain=[YEAR_MIN, YEAR_MAX]),
                axis=alt.Axis(format="d", labelAngle=0),
            ),
            y=alt.Y(
                "completeness_pct:Q",
                title="Reporting Completeness (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "field:N",
                title="Field",
                scale=alt.Scale(
                    domain=list(field_labels.values()),
                    range=[field_colors[f] for f in field_labels.values()],
                ),
            ),
            tooltip=[
                alt.Tooltip("pub_year:Q", title="Year", format="d"),
                alt.Tooltip("field:N", title="Field"),
                alt.Tooltip(
                    "completeness_pct:Q",
                    title="Completeness (%)",
                    format=".1f",
                ),
            ],
        )
    )

    # ---- Era markers ----
    era_markers = create_era_markers(era_transitions)

    # ---- Combine layers ----
    combined = (line + era_markers).properties(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        title="C. Overall Reporting Completeness Over Time",
    )

    return combined


def create_field_type_subplot(
    field_type_df: pd.DataFrame,
    era_transitions: List[Dict],
) -> alt.Chart:
    """Create completeness by field type subplot.

    Args:
        field_type_df: DataFrame with yearly field type completeness
        era_transitions: List of era transition dicts

    Returns:
        Altair chart for field type subplot
    """
    logger.info("Creating field type subplot...")

    # ---- Get field type colors ----
    field_colors = get_field_type_colors(use_gruvbox=True)
    field_types = field_type_df["field_type"].unique().tolist()

    # ---- Create line chart with year on x-axis ----
    line = (
        alt.Chart(field_type_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "pub_year:Q",
                title="Publication Year",
                scale=alt.Scale(domain=[YEAR_MIN, YEAR_MAX]),
                axis=alt.Axis(format="d", labelAngle=0),
            ),
            y=alt.Y(
                "completeness_pct:Q",
                title="Reporting Completeness (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "field_type:N",
                title="Field Type",
                scale=alt.Scale(
                    domain=field_types,
                    range=[field_colors[f] for f in field_types],
                ),
            ),
            tooltip=[
                alt.Tooltip("pub_year:Q", title="Year", format="d"),
                alt.Tooltip("field_type:N", title="Field Type"),
                alt.Tooltip(
                    "completeness_pct:Q",
                    title="Completeness (%)",
                    format=".1f",
                ),
            ],
        )
    )

    # ---- Era markers ----
    era_markers = create_era_markers(era_transitions)

    # ---- Combine layers ----
    combined = (line + era_markers).properties(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        title="D. Reporting Completeness Over Time by Field Type",
    )

    return combined


# ==== Main figure generation ====


def create_faceted_figure(
    temporal_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    completeness_df: pd.DataFrame,
    era_defs: Dict[str, List[int]],
    output_dir: Path,
    dry_run: bool = False,
):
    """Create faceted time series figure with four subplots.

    Args:
        temporal_df: DataFrame with publication year statistics
        diversity_df: DataFrame with trait diversity data
        completeness_df: DataFrame with completeness data
        era_defs: Dictionary of era definitions
        output_dir: Directory to save figure
        dry_run: If True, show what would be done without executing
    """
    logger.info("Creating faceted temporal trends figure...")

    if dry_run:
        logger.info("DRY RUN - Would generate faceted figure")
        return

    # ---- Prepare era transitions ----
    era_transitions = get_era_transitions(era_defs)

    # ---- Prepare field type yearly data ----
    field_type_df = prepare_field_type_yearly_data(completeness_df)

    # ---- Create subplots ----
    subplot_a = create_temporal_distribution_subplot(
        temporal_df, era_transitions
    )
    subplot_b = create_diversity_subplot(diversity_df, era_transitions)
    subplot_c = create_completeness_subplot(completeness_df, era_transitions)
    subplot_d = create_field_type_subplot(field_type_df, era_transitions)

    # ---- Combine vertically ----
    combined = (
        alt.vconcat(
            subplot_a,
            subplot_b,
            subplot_c,
            subplot_d,
            spacing=40,
        )
        .resolve_scale(color="independent")
        .properties(
            title=alt.TitleParams(
                "Temporal Trends in MR Evidence",
                fontSize=18,
                anchor="middle",
            )
        )
    )

    # ---- Save figure ----
    output_file = output_dir / "cs5_fig_temporal_trends.png"
    combined.save(str(output_file), ppi=300)
    logger.info(f"Saved figure: {output_file}")


# ==== Main execution ====


def main():
    """Execute manuscript figure generation for Case Study 5."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Case Study 5: Manuscript figure generation")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    # ---- Load configuration ----
    logger.info("Loading configuration from: {}", args.config)
    config = load_config(args.config)

    cs5_config = config["case_study_5"]
    output_config = config["output"]["case_study_5"]
    cs5_output_dir = PROJECT_ROOT / output_config["base"]

    # ---- Determine output directory ----
    output_dir = args.output_dir

    if args.dry_run:
        logger.info("Dry run - validating configuration and paths")
        logger.info("Output directory: {}", output_dir)
        logger.info("Dry run complete - configuration validated")
        return 0

    # ---- Create output directory ----
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", output_dir)

    # ---- Load data ----
    data_dir = PROJECT_ROOT / "data"
    temporal_df = load_temporal_stats(data_dir)
    diversity_df = load_diversity_data(cs5_output_dir)
    completeness_df = load_completeness_data(cs5_output_dir)

    # ---- Get era definitions ----
    era_defs = cs5_config["temporal_eras"]

    # ---- Generate figure ----
    create_faceted_figure(
        temporal_df,
        diversity_df,
        completeness_df,
        era_defs,
        output_dir,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("Case Study 5 manuscript figures generation complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
