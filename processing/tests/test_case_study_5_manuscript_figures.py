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
    """Consolidate five fields with preserved marginals and local C height."""
    data = pd.DataFrame(
        {
            "pub_year": [2005],
            "p_value_pct": [50.0],
            "confidence_interval_pct": [40.0],
            "standard_error_pct": [30.0],
            "direction_pct": [80.0],
            "effect_size_beta_pct": [20.0],
            "effect_size_or_pct": [60.0],
            "effect_size_hr_pct": [10.0],
        }
    )
    eras = [{"year": 2003, "era_num": "1"}]
    spec = figures.create_completeness_subplot(data, eras).to_dict()
    assert spec["title"] == "C. Reporting completeness by Field Type"
    assert spec["height"] == 400
    assert figures.PLOT_HEIGHT == 200
    line = spec["layer"][0]
    colors = figures.get_completeness_field_colors()
    assert line["encoding"]["color"]["scale"] == {
        "domain": list(colors),
        "range": list(colors.values()),
    }
    assert list(colors) == [
        "Direction",
        "Effect size",
        "Standard error",
        "95% CI",
        "P-Value",
    ]
    assert len(set(colors.values())) == 5
    rows = spec["datasets"][line["data"]["name"]]
    assert [(row["field_type"], row["completeness_pct"]) for row in rows] == [
        ("Direction", 80.0),
        ("Effect size", 60.0),
        ("Standard error", 30.0),
        ("95% CI", 40.0),
        ("P-Value", 50.0),
    ]
    assert len(spec["layer"]) == 2  # line plus nested era-marker layer
    assert len(spec["layer"][1]["layer"]) == 2
    assert line["mark"]["interpolate"] == "linear"
    assert "effect_size_pct" not in data
