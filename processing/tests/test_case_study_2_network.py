"""Unit tests for case study 2 network utilities."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "processing"
    / "scripts"
    / "analysis"
    / "case_study_2_network_analysis.py"
)
OVERLAY_MODULE_PATH = (
    REPO_ROOT
    / "processing"
    / "scripts"
    / "analysis"
    / "case_study_2_concordance_overlay.py"
)
HOTSPOT_MODULE_PATH = (
    REPO_ROOT
    / "processing"
    / "scripts"
    / "analysis"
    / "case_study_2_hotspot_profiles.py"
)
CONFIG_PATH = REPO_ROOT / "processing" / "config" / "case_studies.yml"
COOCCURRENCE_PATH = (
    REPO_ROOT / "data" / "processed" / "case-study-cs2" / "cooccurrence"
)


def load_filter_pair_records():
    module_ns = runpy.run_path(str(MODULE_PATH))
    return module_ns["filter_pair_records"]


def load_overlay_stats_func():
    module_ns = runpy.run_path(str(HOTSPOT_MODULE_PATH))
    return module_ns["load_overlay_stats"]


def test_filter_pair_records_thresholds() -> None:
    filter_pair_records = load_filter_pair_records()
    pair_df = pd.DataFrame(
        {
            "trait_a": ["a", "b", "c"],
            "trait_b": ["b", "c", "d"],
            "shared_studies": [5, 1, 3],
            "semantic_similarity": [0.7, 0.4, 0.8],
            "jaccard": [0.3, 0.05, 0.2],
        }
    )

    parameters = {
        "edge_weight_threshold": 2,
        "semantic_similarity_threshold": 0.5,
        "jaccard_threshold": 0.1,
    }

    filtered = filter_pair_records(pair_df, parameters)
    assert len(filtered) == 2
    assert set(filtered["trait_a"]) == {"a", "c"}
    assert set(filtered["trait_b"]) == {"b", "d"}


def test_load_overlay_stats_from_pairs(tmp_path: Path) -> None:
    load_overlay_stats = load_overlay_stats_func()
    pair_df = pd.DataFrame(
        {
            "trait_a": ["Trait 1", "Trait 2"],
            "trait_b": ["Trait 3", "Trait 4"],
            "pair_direction_agreement": [1, -1],
            "pair_match_type": ["exact", "fuzzy"],
        }
    )
    overlay_path = tmp_path / "trait_similarity_concordance_pairs.csv"
    pair_df.to_csv(overlay_path, index=False)

    stats = load_overlay_stats(overlay_path)
    assert "Trait 1" in stats
    trait_stats = stats["Trait 1"]
    assert trait_stats["concordance_mean"] == 1.0
    assert trait_stats["concordance_count"] == 1.0
    assert trait_stats["direction_positive"] == 1.0


def test_network_cli_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--dry-run",
            "--config",
            str(CONFIG_PATH),
            "--cooccurrence-dir",
            str(COOCCURRENCE_PATH),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_overlay_cli_outputs_pairs(tmp_path: Path) -> None:
    output_dir = tmp_path / "overlays"
    figure_dir = tmp_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(OVERLAY_MODULE_PATH),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
            "--figures-dir",
            str(figure_dir),
            "--limit",
            "5",
            "--min-shared-pairs",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    pair_path = output_dir / "trait_similarity_concordance_pairs.csv"
    assert pair_path.exists()
    pair_df = pd.read_csv(pair_path)
    required_cols = {"trait_a", "trait_b", "pair_direction_agreement"}
    assert required_cols.issubset(set(pair_df.columns))
