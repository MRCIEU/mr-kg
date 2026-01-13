"""Generate manuscript figures for summary statistics.

This script creates publication-ready figures for the MR literature
temporal distribution analysis using Altair. Figures are exported as PNG
at 300 DPI for manuscript inclusion.

Data sources:
    - temporal-statistics.csv: Publication year statistics

Output:
    - summary_temporal_distribution.png: Log-scale line plot of papers
      by year
"""

import argparse
from pathlib import Path
from typing import List

import altair as alt
import pandas as pd
from yiutils.project_utils import find_project_root

from color_schemes import GRUVBOX_DARK


# ==== Path setup ====


def get_project_paths() -> tuple[Path, Path, Path]:
    """Get standard project paths.

    Returns:
        Tuple of (project_root, data_dir, output_dir)
    """
    project_root = find_project_root("docker-compose.yml")
    data_dir = project_root / "data"
    output_dir = data_dir / "artifacts" / "manuscript-figures"
    return project_root, data_dir, output_dir


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

    df = pd.read_csv(temporal_stats_file)

    # ---- Remove incomplete years ----
    df = df[df["publication_year"] <= 2024].copy()

    # ---- Sort and filter for log scale ----
    df = df.sort_values("publication_year")
    df = df[df["paper_count"] > 0].copy()

    return df


# ==== Figure generation ====


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
    return log_ticks


def create_temporal_distribution_figure(df: pd.DataFrame) -> alt.LayerChart:
    """Create temporal distribution line chart with log scale.

    Args:
        df: DataFrame with publication_year and paper_count columns

    Returns:
        Altair layered chart with line and text labels
    """
    min_year = int(df["publication_year"].min())
    ymax = float(df["paper_count"].max())
    log_ticks = calculate_log_ticks(ymax)

    base_chart = alt.Chart(df).properties(width=800, height=420)

    # ---- Line chart with points ----
    line_raw = base_chart.mark_line(
        point=True, color=GRUVBOX_DARK["blue"]
    ).encode(
        x=alt.X(
            "publication_year:Q",
            title="Publication Year",
            scale=alt.Scale(domain=[min_year, 2024]),
            axis=alt.Axis(tickMinStep=1, format="d"),
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

    # ---- Text labels for all points ----
    labels_all = base_chart.mark_text(
        align="center",
        dy=26,
        stroke="black",
        strokeWidth=1,
        fontSize=10,
    ).encode(
        x="publication_year:Q",
        y="paper_count:Q",
        text=alt.Text("paper_count:Q", format=","),
    )

    # ---- Enable x-axis zoom/pan ----
    xzoom = alt.selection_interval(bind="scales", encodings=["x"])

    res = (
        alt.layer(line_raw, labels_all)
        .add_params(xzoom)
        .properties(
            title="MR papers by publication year (log scale)",
        )
        .configure_axis(gridColor="#e6e6e6", gridOpacity=0.7)
    )

    return res


def create_temporal_distribution_normal_scale(
    df: pd.DataFrame,
) -> alt.LayerChart:
    """Create temporal distribution line chart with normal scale.

    Args:
        df: DataFrame with publication_year and paper_count columns

    Returns:
        Altair layered chart with line and text labels
    """
    min_year = int(df["publication_year"].min())
    max_year = int(df["publication_year"].max())

    base_chart = alt.Chart(df).properties(width=800, height=420)

    # ---- Line chart with points ----
    line_raw = base_chart.mark_line(
        point=True, color=GRUVBOX_DARK["blue"]
    ).encode(
        x=alt.X(
            "publication_year:Q",
            title="Publication Year",
            scale=alt.Scale(domain=[min_year, max_year]),
            axis=alt.Axis(tickMinStep=1, format="d"),
        ),
        y=alt.Y(
            "paper_count:Q",
            title="Number of Papers",
            scale=alt.Scale(zero=True),
            axis=alt.Axis(format=","),
        ),
        tooltip=[
            alt.Tooltip("publication_year:Q", title="Year", format="d"),
            alt.Tooltip("paper_count:Q", title="Papers", format=","),
        ],
    )

    # ---- Text labels for selected points ----
    # Only show labels for years after 2015 to avoid clutter
    df_labeled = df[df["publication_year"] >= 2015].copy()
    labels_recent = (
        alt.Chart(df_labeled)
        .mark_text(
            align="center",
            dy=-10,
            fontSize=10,
        )
        .encode(
            x="publication_year:Q",
            y="paper_count:Q",
            text=alt.Text("paper_count:Q", format=","),
        )
    )

    # ---- Enable x-axis zoom/pan ----
    xzoom = alt.selection_interval(bind="scales", encodings=["x"])

    res = (
        alt.layer(line_raw, labels_recent)
        .add_params(xzoom)
        .properties(
            title="MR papers by publication year",
        )
        .configure_axis(gridColor="#e6e6e6", gridOpacity=0.7)
    )

    return res


# ==== Main execution ====


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    res = parser.parse_args()
    return res


def main() -> None:
    """Generate summary manuscript figures."""
    args = parse_args()

    # ---- Get paths ----
    project_root, data_dir, output_dir = get_project_paths()

    print("Summary Manuscript Figures Generator")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    if args.dry_run:
        print("[DRY RUN] No files will be created")
        print()

    # ---- Load data ----
    print("Loading data...")
    temporal_stats = load_temporal_stats(data_dir)
    print(
        f"  Loaded temporal statistics: {len(temporal_stats)} "
        f"years ({temporal_stats['publication_year'].min()}-"
        f"{temporal_stats['publication_year'].max()})"
    )
    print()

    # ---- Configure Altair for PNG export ----
    alt.data_transformers.enable("default", max_rows=None)

    # ---- Generate figures ----
    print("Generating figures...")

    # ---- Create output directories ----
    lit_figures_dir = data_dir / "processed" / "figures" / "literature"

    # ---- Figure: Temporal distribution (log scale) ----
    print("  Creating temporal distribution figure (log scale)...")
    temporal_fig_log = create_temporal_distribution_figure(temporal_stats)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "summary_temporal_distribution.png"
        temporal_fig_log.save(str(output_file), ppi=300)
        print(f"    Saved: {output_file}")

        # ---- Save PNG to literature figures directory ----
        lit_figures_dir.mkdir(parents=True, exist_ok=True)
        output_png = lit_figures_dir / "temporal_distribution_line.png"
        temporal_fig_log.save(str(output_png), ppi=300)
        print(f"    Saved: {output_png}")
    else:
        print("    [DRY RUN] Would save: summary_temporal_distribution.png")
        print(
            "    [DRY RUN] Would save: "
            "temporal_distribution_line.png (log scale)"
        )

    # ---- Figure: Temporal distribution (normal scale) ----
    print("  Creating temporal distribution figure (normal scale)...")
    temporal_fig_normal = create_temporal_distribution_normal_scale(
        temporal_stats
    )

    if not args.dry_run:
        output_file_normal = (
            output_dir / "summary_temporal_distribution_normal.png"
        )
        temporal_fig_normal.save(str(output_file_normal), ppi=300)
        print(f"    Saved: {output_file_normal}")

        # ---- Save PNG to literature figures directory ----
        output_png_normal = (
            lit_figures_dir / "temporal_distribution_line_normal.png"
        )
        temporal_fig_normal.save(str(output_png_normal), ppi=300)
        print(f"    Saved: {output_png_normal}")
    else:
        print(
            "    [DRY RUN] Would save: "
            "summary_temporal_distribution_normal.png"
        )
        print(
            "    [DRY RUN] Would save: "
            "temporal_distribution_line_normal.png (normal scale)"
        )

    print()
    print("Summary manuscript figures generation complete!")


if __name__ == "__main__":
    main()
