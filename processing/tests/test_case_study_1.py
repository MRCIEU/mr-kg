"""Tests for Case Study 1 reproducibility analysis scripts."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))

try:
    from case_study_1_reproducibility_metrics import assign_study_count_band
except ImportError:
    assign_study_count_band = None


@pytest.fixture
def config_path():
    """Path to case studies configuration file."""
    return Path(__file__).parent.parent / "config" / "case_studies.yml"


@pytest.fixture
def config(config_path):
    """Load case studies configuration."""
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def test_config_exists(config_path):
    """Test that configuration file exists."""
    assert config_path.exists(), f"Config file not found: {config_path}"


def test_config_structure(config):
    """Test that configuration has required structure."""
    assert "case_study_1" in config
    assert "databases" in config
    assert "output" in config

    cs1_config = config["case_study_1"]
    assert "min_study_count" in cs1_config
    assert "reproducibility_tiers" in cs1_config
    assert "study_count_bands" in cs1_config
    assert "temporal_eras" in cs1_config
    assert "validation" in cs1_config


def test_config_values(config):
    """Test that configuration values are sensible."""
    cs1_config = config["case_study_1"]

    assert cs1_config["min_study_count"] >= 2
    assert isinstance(cs1_config["reproducibility_tiers"], dict)

    tiers = cs1_config["reproducibility_tiers"]
    assert tiers["high"] > tiers["moderate"]
    assert tiers["moderate"] > tiers["low"]
    assert tiers["low"] >= 0.0
    assert tiers["high"] <= 1.0


def test_temporal_eras(config):
    """Test temporal era definitions."""
    cs1_config = config["case_study_1"]
    eras = cs1_config["temporal_eras"]

    assert "early" in eras
    assert "recent" in eras

    early = eras["early"]
    recent = eras["recent"]

    assert len(early) == 2
    assert len(recent) == 2
    assert early[0] < early[1]
    assert recent[0] < recent[1]
    assert early[1] <= recent[0]


def test_study_count_bands(config):
    """Test study count stratification bands."""
    cs1_config = config["case_study_1"]
    bands = cs1_config["study_count_bands"]

    assert len(bands) > 0

    for band in bands:
        assert len(band) == 2
        assert band[0] <= band[1]

    for i in range(len(bands) - 1):
        assert (
            bands[i][1] < bands[i + 1][0] or bands[i][1] == bands[i + 1][0] - 1
        )


def test_canonical_pairs(config):
    """Test canonical validation pairs."""
    cs1_config = config["case_study_1"]
    canonical = cs1_config["validation"]["canonical_pairs"]

    assert len(canonical) > 0

    for pair in canonical:
        assert len(pair) == 2
        assert isinstance(pair[0], str)
        assert isinstance(pair[1], str)
        assert len(pair[0]) > 0
        assert len(pair[1]) > 0


def test_database_paths(config):
    """Test that database paths are defined."""
    db_config = config["databases"]

    assert "evidence_profile" in db_config
    assert "trait_profile" in db_config
    assert "vector_store" in db_config

    for db_path in db_config.values():
        assert db_path.startswith("data/db/")
        assert db_path.endswith(".db")


def test_output_paths(config):
    """Test that output directory structure is defined."""
    cs1_output = config["output"]["case_study_1"]

    required_dirs = ["base", "raw_pairs", "metrics", "models", "figures"]

    for dir_key in required_dirs:
        assert dir_key in cs1_output
        assert cs1_output[dir_key].startswith("data/processed/case-study-cs1")


@pytest.mark.skipif(
    assign_study_count_band is None,
    reason="assign_study_count_band function not available",
)
def test_assign_study_count_band_within_ranges(config):
    """Test study count band assignment for values within defined ranges."""
    cs1_config = config["case_study_1"]
    bands = cs1_config["study_count_bands"]

    # ---- Test each band ----
    assert assign_study_count_band(2, bands) == "2-3"
    assert assign_study_count_band(3, bands) == "2-3"
    assert assign_study_count_band(4, bands) == "4-6"
    assert assign_study_count_band(6, bands) == "4-6"
    assert assign_study_count_band(7, bands) == "7-10"
    assert assign_study_count_band(10, bands) == "7-10"
    assert assign_study_count_band(11, bands) == "11+"
    assert assign_study_count_band(50, bands) == "11+"
    assert assign_study_count_band(999, bands) == "11+"


@pytest.mark.skipif(
    assign_study_count_band is None,
    reason="assign_study_count_band function not available",
)
def test_assign_study_count_band_out_of_range(config):
    """Test that study count band assignment raises error for out-of-range."""
    cs1_config = config["case_study_1"]
    bands = cs1_config["study_count_bands"]

    # ---- Values below minimum should raise ValueError ----
    with pytest.raises(ValueError, match="falls outside all defined bands"):
        assign_study_count_band(0, bands)

    with pytest.raises(ValueError, match="falls outside all defined bands"):
        assign_study_count_band(1, bands)

    # ---- Values above maximum (999) should also raise ValueError ----
    with pytest.raises(ValueError, match="falls outside all defined bands"):
        assign_study_count_band(1000, bands)
