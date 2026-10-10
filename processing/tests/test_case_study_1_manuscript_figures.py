"""Display and numerical-preservation checks for the four-panel CS1 figures."""

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import pytest
import vl_convert as vlc

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))

import case_study_1_manuscript_figures as figures  # noqa: E402


@pytest.fixture
def metrics() -> pd.DataFrame:
    """Supply several categories, both match groups and excluded categories."""
    return pd.DataFrame(
        {
            "outcome_category": [
                "cardiovascular",
                "cardiovascular",
                "cancer",
                "cancer",
                "other",
                "psychiatric",
                "uncategorized",
                None,
            ],
            "reproducibility_tier": [
                "high",
                "low",
                "high",
                "moderate",
                "high",
                "high",
                "discordant",
                "moderate",
            ],
            "mean_direction_concordance": [
                0.9,
                0.1,
                0.7,
                0.5,
                1.0,
                0.8,
                -1.0,
                0.4,
            ],
            "has_exact_match": [
                True,
                False,
                False,
                True,
                False,
                True,
                True,
                False,
            ],
            # Include mixed any-exact and EFO-only no-exact summaries.
            "has_fuzzy_match": [
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
            ],
            "has_efo_match": [
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
            ],
            "study_count_band": [
                "2-3",
                "2-3",
                "4-6",
                "4-6",
                "7-10",
                "11+",
                "2-3",
                "4-6",
            ],
        }
    )


@pytest.fixture
def captured_specs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[dict[str, Any]]:
    """Capture charts without writing PNGs or reading retained data."""
    tier_path = tmp_path / "cs1_tier_distribution.csv"
    tier_path.write_text(
        "Reproducibility Tier,Pairs (n),Percentage (%)\n"
        "High,10,52.6\nModerate,2,10.5\nLow,3,15.8\n"
        "Discordant,4,21.1\nTotal,19,100.0\n"
    )
    monkeypatch.setattr(figures, "MANUSCRIPT_TABLES_DIR", tmp_path)
    specs: list[dict[str, Any]] = []

    def capture(
        chart: alt.TopLevelMixin, filename: Any, **kwargs: Any
    ) -> None:
        """Keep the spec and check the existing export resolution."""
        assert kwargs == {"ppi": 300}
        assert Path(filename).parent == tmp_path
        specs.append(chart.to_dict())

    monkeypatch.setattr(alt.TopLevelMixin, "save", capture)
    return specs


def rows(spec: dict[str, Any], chart: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve an inline chart dataset without changing its fields."""
    return spec["datasets"][chart["data"]["name"]]


def test_figure_1_keeps_four_panels_and_separate_populations(
    metrics: pd.DataFrame,
    captured_specs: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Panel A uses its own input; B/C/D retain categorised summaries."""
    original = metrics.copy(deep=True)
    tier_bytes = (tmp_path / "cs1_tier_distribution.csv").read_bytes()
    figures.create_figure_1(metrics, tmp_path)
    spec = captured_specs[0]
    left, right = spec["hconcat"]
    assert len(left["vconcat"]) == len(right["vconcat"]) == 3
    panel_a, panel_b = left["vconcat"][1:]
    panel_c, panel_d = right["vconcat"][1:]
    assert spec["title"]["text"] == "Reproducibility"
    assert panel_a["title"]["text"] == (
        "A. Reproducibility tiers: all summaries"
    )
    assert panel_b["title"]["text"] == (
        "B. Reproducibility tiers by outcome category"
    )
    for panel in [panel_a, panel_b]:
        assert panel["layer"][0]["encoding"]["color"]["title"] == (
            "Reproducibility tier"
        )
    assert panel_a["title"]["subtitle"] == "Concordance summaries (n=19)"
    for panel in [panel_b, panel_c, panel_d]:
        assert panel["title"]["subtitle"] == (
            "Categorised concordance summaries (n=6)"
        )
    assert panel_c["title"]["text"] == (
        "C. Concordance by exact-match presence"
    )
    tier_rows = rows(spec, panel_a)
    assert [row["Pairs (n)"] for row in tier_rows] == [10, 2, 3, 4]
    assert [row["Percentage (%)"] for row in tier_rows] == [
        52.6,
        10.5,
        15.8,
        21.1,
    ]
    category_rows = rows(spec, panel_b["layer"][0])
    assert sum(row["count"] for row in category_rows) == 6
    assert (
        next(
            row for row in category_rows if row["outcome_category"] == "other"
        )["count"]
        == 1
    )
    assert [
        (row["match_type"], row["mean_direction_concordance"])
        for row in rows(spec, panel_c)
    ] == [
        ("exact", 0.9),
        ("fuzzy", 0.1),
        ("fuzzy", 0.7),
        ("exact", 0.5),
        ("fuzzy", 1.0),
        ("exact", 0.8),
    ]
    assert len(rows(spec, panel_d)) == 6
    pd.testing.assert_frame_equal(metrics, original)
    assert (tmp_path / "cs1_tier_distribution.csv").read_bytes() == tier_bytes


def test_figure_1_changes_legend_text_not_match_keys_or_densities(
    metrics: pd.DataFrame,
    captured_specs: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    """The legend names any-exact/no-exact while keeping original groups."""
    figures.create_figure_1(metrics, tmp_path)
    spec = captured_specs[0]
    panel_c, panel_d = spec["hconcat"][1]["vconcat"][1:]
    for chart, groups in [
        (panel_c, ["match_type"]),
        (panel_d["spec"], ["outcome_category", "match_type"]),
    ]:
        assert chart["transform"] == [
            {
                "density": "mean_direction_concordance",
                "bandwidth": 0.15,
                "groupby": groups,
                "as": ["concordance", "density"],
            }
        ]
        assert chart["encoding"]["color"]["scale"]["domain"] == [
            "exact",
            "fuzzy",
        ]
        assert chart["encoding"]["x"]["title"] == (
            "Mean direction concordance"
        )
    legend = panel_d["spec"]["encoding"]["color"]["legend"]
    assert legend["title"] == "Exact-match presence"
    assert legend["labelExpr"] == (
        "datum.label === 'exact' ? 'Any exact' : 'No exact'"
    )
    compiled = vlc.vegalite_to_vega(spec, vl_version="6.1")
    assert legend["labelExpr"] in str(compiled)
    assert len(rows(spec, panel_d)) == 6


def test_figure_2_preserves_bands_scores_means_and_threshold(
    metrics: pd.DataFrame,
    captured_specs: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    """Change units, not the saved band selection or plotted values."""
    data = pd.concat(
        [
            metrics,
            pd.DataFrame(
                {
                    "study_count_band": ["invalid", "2-3"],
                    "mean_direction_concordance": [0.6, None],
                }
            ),
        ],
        ignore_index=True,
    )
    figures.create_figure_2(data, tmp_path)
    spec = captured_specs[0]
    facet = spec["vconcat"][0]
    plotted = rows(spec, facet)
    assert spec["title"]["text"] == (
        "Concordance by number of compared studies"
    )
    assert spec["title"]["subtitle"] == [
        "Concordance summaries (n=8)",
        "Focal study plus retained comparisons within the same model",
    ]
    assert facet["facet"]["sort"] == ["2-3", "4-6", "7-10", "11+"]
    assert Counter(row["study_count_band"] for row in plotted) == {
        "2-3": 3,
        "4-6": 3,
        "7-10": 1,
        "11+": 1,
    }
    assert [row["mean_direction_concordance"] for row in plotted] == (
        metrics["mean_direction_concordance"].tolist()
    )
    expected_means = {"2-3": 0.0, "4-6": 1.6 / 3, "7-10": 1.0, "11+": 0.8}
    for row in plotted:
        assert row["mean_conc"] == pytest.approx(
            expected_means[row["study_count_band"]]
        )
        assert row["threshold"] == 0.7
    histogram = facet["spec"]["layer"][0]
    assert histogram["encoding"]["x"] == {
        "bin": {"maxbins": 20},
        "field": "mean_direction_concordance",
        "scale": {"domain": [-1, 1]},
        "title": "Mean direction concordance",
        "type": "quantitative",
    }
    assert histogram["encoding"]["y"]["title"] == ("Concordance summaries (n)")
    vlc.vegalite_to_vega(spec, vl_version="6.1")


def test_dry_run_does_not_read_tier_table_or_save(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep the existing no-write dry-run path."""

    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Dry run must not read or save figure inputs")

    monkeypatch.setattr(pd, "read_csv", fail)
    monkeypatch.setattr(alt.TopLevelMixin, "save", fail)
    figures.create_figure_1(pd.DataFrame(), tmp_path, dry_run=True)
    figures.create_figure_2(pd.DataFrame(), tmp_path, dry_run=True)
    assert not list(tmp_path.iterdir())
