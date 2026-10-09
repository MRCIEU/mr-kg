"""Presentation and supplied-bound checks for the combined CS5 figure."""

import sys
from pathlib import Path

import pandas as pd
import vl_convert as vlc

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))

import case_study_5_manuscript_figures as figures


def test_diversity_axis_title_and_supplied_sd_bounds() -> None:
    """Use a human-readable axis without changing explicit SD offsets."""
    data = pd.DataFrame(
        {
            "pub_year": [2005, 2007],
            "mean_trait_count": [2.5, 4.0],
            "std_trait_count": [3.0, 2.0],
            "n_studies": [11, 16],
        }
    )
    eras = [{"year": 2003, "era_num": "1"}]
    spec = figures.create_diversity_subplot(data, eras).to_dict()
    band = spec["layer"][0]
    assert band["mark"]["extent"] == "stderr"
    assert band["encoding"]["yError"]["field"] == "std_trait_count"
    vega = vlc.vegalite_to_vega(spec, vl_version="6.1")
    titles = [
        axis["title"]
        for axis in vega["axes"]
        if axis.get("scale") == "y" and "title" in axis
    ]
    assert titles == ["Mean Traits per Study"]
    formulas = {
        transform["as"]: transform["expr"]
        for dataset in vega["data"]
        for transform in dataset.get("transform", [])
        if transform["type"] == "formula"
        and transform["as"].endswith("_mean_trait_count")
    }
    assert formulas == {
        "upper_mean_trait_count": (
            "datum['mean_trait_count'] + datum['std_trait_count']"
        ),
        "lower_mean_trait_count": (
            "datum['mean_trait_count'] - datum['std_trait_count']"
        ),
    }


def test_completeness_labels_preserve_series_values_and_colors() -> None:
    """Expand the SE label without changing its data or original color."""
    data = pd.DataFrame(
        {
            "pub_year": [2005],
            "p_value_pct": [50.0],
            "confidence_interval_pct": [40.0],
            "standard_error_pct": [30.0],
        }
    )
    eras = [{"year": 2003, "era_num": "1"}]
    spec = figures.create_completeness_subplot(data, eras).to_dict()
    assert spec["title"] == (
        "C. Reporting completeness over time by field"
    )
    line = spec["layer"][0]
    colors = figures.get_field_type_colors(use_gruvbox=True)
    assert line["encoding"]["color"]["scale"] == {
        "domain": ["P-value", "95% CI", "Standard error"],
        "range": [colors["P-value"], colors["95% CI"], colors["SE"]],
    }
    rows = spec["datasets"][line["data"]["name"]]
    assert [(row["field"], row["completeness_pct"]) for row in rows] == [
        ("P-value", 50.0),
        ("95% CI", 40.0),
        ("Standard error", 30.0),
    ]
