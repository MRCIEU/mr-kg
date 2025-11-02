"""Tests for Case Study 5 temporal evolution analysis scripts."""

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "analysis"))


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
    """Test that configuration has required Case Study 5 structure."""
    assert "case_study_5" in config
    assert "databases" in config
    assert "output" in config

    cs5_config = config["case_study_5"]
    assert "models_included" in cs5_config
    assert "temporal_eras" in cs5_config
    assert "trait_diversity" in cs5_config
    assert "evidence_consistency" in cs5_config
    assert "reporting_completeness" in cs5_config
    assert "statistical" in cs5_config


def test_temporal_eras(config):
    """Test that temporal eras are properly defined."""
    eras = config["case_study_5"]["temporal_eras"]

    assert isinstance(eras, dict)
    assert len(eras) == 5

    expected_eras = [
        "early_mr",
        "mr_egger",
        "mr_presso",
        "within_family",
        "strobe_mr",
    ]
    for era in expected_eras:
        assert era in eras
        assert isinstance(eras[era], list)
        assert len(eras[era]) == 2
        assert eras[era][0] <= eras[era][1]


def test_reporting_completeness_config(config):
    """Test reporting completeness configuration."""
    rc_config = config["case_study_5"]["reporting_completeness"]

    assert "required_fields" in rc_config
    assert "composite_completeness_definition" in rc_config
    assert "strobe_breakpoint" in rc_config

    assert isinstance(rc_config["required_fields"], list)
    assert len(rc_config["required_fields"]) > 0
    assert rc_config["strobe_breakpoint"] == 2021


def test_trait_diversity_config(config):
    """Test trait diversity configuration."""
    td_config = config["case_study_5"]["trait_diversity"]

    assert "min_studies_per_year" in td_config
    assert "min_traits_per_study" in td_config

    assert td_config["min_studies_per_year"] > 0
    assert td_config["min_traits_per_study"] >= 1


def test_evidence_consistency_config(config):
    """Test evidence consistency configuration."""
    ec_config = config["case_study_5"]["evidence_consistency"]

    assert "min_pairs_per_era" in ec_config
    assert "concordance_improvement_threshold" in ec_config

    assert ec_config["min_pairs_per_era"] > 0
    assert 0 <= ec_config["concordance_improvement_threshold"] <= 1


def test_output_paths(config):
    """Test that output paths are defined."""
    cs5_output = config["output"]["case_study_5"]

    expected_dirs = [
        "base",
        "temporal",
        "diversity",
        "consistency",
        "completeness",
        "figures",
    ]

    for dir_name in expected_dirs:
        assert dir_name in cs5_output
        assert isinstance(cs5_output[dir_name], str)
        assert len(cs5_output[dir_name]) > 0


def test_database_paths(config):
    """Test that database paths are defined."""
    databases = config["databases"]

    assert "vector_store" in databases
    assert "trait_profile" in databases
    assert "evidence_profile" in databases


def test_models_included(config):
    """Test that models_included is properly defined."""
    models = config["case_study_5"]["models_included"]

    assert isinstance(models, list)
    assert len(models) > 0
    assert "gpt-5" in models


def test_statistical_config(config):
    """Test statistical configuration."""
    stats_config = config["case_study_5"]["statistical"]

    assert "confidence_level" in stats_config
    assert "significance_threshold" in stats_config
    assert "min_observations_for_regression" in stats_config

    assert 0 < stats_config["confidence_level"] < 1
    assert 0 < stats_config["significance_threshold"] < 1
    assert stats_config["min_observations_for_regression"] > 0


def test_covid_era_years(config):
    """Test COVID era years are defined."""
    covid_years = config["case_study_5"]["covid_era_years"]

    assert isinstance(covid_years, list)
    assert len(covid_years) == 2
    assert 2020 in covid_years
    assert 2021 in covid_years


def test_fashionable_traits_config(config):
    """Test fashionable traits configuration."""
    ft_config = config["case_study_5"]["fashionable_traits"]

    assert "top_k_per_year" in ft_config
    assert "hype_cycle_growth_threshold" in ft_config
    assert "hype_cycle_decline_threshold" in ft_config
    assert "min_occurrences_for_trend" in ft_config

    assert ft_config["top_k_per_year"] > 0
    assert ft_config["hype_cycle_growth_threshold"] > 0
    assert 0 < ft_config["hype_cycle_decline_threshold"] < 1
    assert ft_config["min_occurrences_for_trend"] >= 2


def test_pleiotropy_config(config):
    """Test pleiotropy awareness configuration."""
    p_config = config["case_study_5"]["pleiotropy"]

    assert "canonical_pairs" in p_config
    assert "cs2_hotspots" in p_config
    assert "mr_presso_breakpoint" in p_config

    assert isinstance(p_config["canonical_pairs"], list)
    assert len(p_config["canonical_pairs"]) > 0
    assert p_config["mr_presso_breakpoint"] > 2000


def test_winners_curse_config(config):
    """Test winner's curse configuration."""
    wc_config = config["case_study_5"]["winners_curse"]

    assert "min_studies_per_pair" in wc_config
    assert "min_effect_size_availability" in wc_config
    assert "subgroup_categories" in wc_config
    assert "effect_size_fields" in wc_config
    assert "expected_decline_range" in wc_config

    assert wc_config["min_studies_per_pair"] >= 2
    assert 0 <= wc_config["min_effect_size_availability"] <= 1
    assert isinstance(wc_config["effect_size_fields"], list)
    assert len(wc_config["expected_decline_range"]) == 2


# ==== Phase 0: Temporal preparation tests ====


def test_temporal_preparation_output_exists(config):
    """Test that temporal preparation outputs exist."""
    base_path = Path(config["output"]["case_study_5"]["temporal"])

    expected_files = [
        "temporal_metadata.csv",
        "era_statistics.csv",
        "temporal_metadata.json",
    ]

    for file_name in expected_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_temporal_metadata_structure(config):
    """Test temporal metadata CSV has expected structure."""
    temporal_path = (
        Path(config["output"]["case_study_5"]["temporal"])
        / "temporal_metadata.csv"
    )

    if not temporal_path.exists():
        pytest.skip(f"Temporal metadata not found: {temporal_path}")

    df = pd.read_csv(temporal_path)

    expected_columns = [
        "pmid",
        "pub_date",
        "pub_year",
        "era",
        "covid_era_flag",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["pub_year"].notna().all()
    assert df["era"].notna().all()
    assert df["pmid"].is_unique


def test_era_statistics_structure(config):
    """Test era statistics CSV has expected structure."""
    era_stats_path = (
        Path(config["output"]["case_study_5"]["temporal"])
        / "era_statistics.csv"
    )

    if not era_stats_path.exists():
        pytest.skip(f"Era statistics not found: {era_stats_path}")

    df = pd.read_csv(era_stats_path)

    expected_columns = [
        "era",
        "year_range",
        "n_studies",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) >= 5
    assert df["n_studies"].sum() > 0


# ==== Phase 1: Trait diversity tests ====


def test_trait_diversity_output_exists(config):
    """Test that trait diversity outputs exist."""
    base_path = Path(config["output"]["case_study_5"]["diversity"])

    expected_files = [
        "trait_counts_by_year.csv",
        "trait_counts_by_era.csv",
        "temporal_trend_model.csv",
        "era_comparison_tests.csv",
        "diversity_metadata.json",
    ]

    for file_name in expected_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_trait_counts_by_year_structure(config):
    """Test trait counts by year CSV structure."""
    file_path = (
        Path(config["output"]["case_study_5"]["diversity"])
        / "trait_counts_by_year.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Trait counts file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "pub_year",
        "n_studies",
        "mean_trait_count",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["n_studies"].sum() > 0
    assert df["mean_trait_count"].min() > 0


def test_temporal_trend_model_results(config):
    """Test temporal trend model has valid results."""
    file_path = (
        Path(config["output"]["case_study_5"]["diversity"])
        / "temporal_trend_model.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Trend model not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "slope",
        "intercept",
        "p_value",
        "r_squared",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["p_value"].between(0, 1).all()
    assert df["r_squared"].between(0, 1).all()


# ==== Phase 3: Evidence consistency tests ====


def test_evidence_consistency_output_exists(config):
    """Test that evidence consistency outputs exist."""
    base_path = Path(config["output"]["case_study_5"]["consistency"])

    expected_files = [
        "concordance_by_year.csv",
        "concordance_by_era.csv",
        "concordance_by_match_type_era.csv",
        "era_comparison_tests.csv",
        "strobe_impact_analysis.csv",
        "consistency_metadata.json",
    ]

    for file_name in expected_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_concordance_by_era_structure(config):
    """Test concordance by era CSV structure."""
    file_path = (
        Path(config["output"]["case_study_5"]["consistency"])
        / "concordance_by_era.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Concordance file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "era",
        "mean_concordance",
        "median_concordance",
        "std_concordance",
        "n_pairs",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) >= 2
    assert df["mean_concordance"].between(-1, 1).all()
    assert df["n_pairs"].min() > 0


def test_strobe_impact_analysis(config):
    """Test STROBE-MR impact analysis results."""
    file_path = (
        Path(config["output"]["case_study_5"]["consistency"])
        / "strobe_impact_analysis.csv"
    )

    if not file_path.exists():
        pytest.skip(f"STROBE impact file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "period",
        "mean_concordance",
        "n_pairs",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) == 2
    assert df["mean_concordance"].between(-1, 1).all()


# ==== Phase 4: Reporting completeness tests ====


def test_reporting_completeness_output_exists(config):
    """Test that reporting completeness outputs exist."""
    base_path = Path(config["output"]["case_study_5"]["completeness"])

    expected_files = [
        "field_completeness_by_year.csv",
        "field_completeness_by_era.csv",
        "field_type_by_era.csv",
        "strobe_impact_on_reporting.csv",
        "completeness_metadata.json",
    ]

    for file_name in expected_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_field_completeness_structure(config):
    """Test field completeness CSV structure."""
    file_path = (
        Path(config["output"]["case_study_5"]["completeness"])
        / "field_completeness_by_era.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Completeness file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "era",
        "n_studies",
        "effect_size_beta_pct",
        "p_value_pct",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["effect_size_beta_pct"].between(0, 100).all()
    assert df["n_studies"].min() > 0


def test_strobe_reporting_impact(config):
    """Test STROBE-MR reporting impact analysis."""
    file_path = (
        Path(config["output"]["case_study_5"]["completeness"])
        / "strobe_impact_on_reporting.csv"
    )

    if not file_path.exists():
        pytest.skip(f"STROBE reporting impact not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "field",
        "pre_pct",
        "post_pct",
        "change_pct",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["pre_pct"].between(0, 100).all()
    assert df["post_pct"].between(0, 100).all()


# ==== Phase 5: Fashionable traits tests ====


def test_fashionable_traits_output_exists(config):
    """Test that fashionable traits outputs exist."""
    base_path = Path(config["output"]["case_study_5"]["fashionable"])

    if not base_path.exists():
        pytest.skip(f"Fashionable traits directory not found: {base_path}")

    expected_files = [
        "top_traits_by_year.csv",
        "trait_popularity_trends.csv",
        "hype_cycle_candidates.csv",
        "fashionable_metadata.json",
    ]

    for file_name in expected_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_top_traits_structure(config):
    """Test top traits by year CSV structure."""
    file_path = (
        Path(config["output"]["case_study_5"]["fashionable"])
        / "top_traits_by_year.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Top traits file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "pub_year",
        "trait",
        "occurrence_count",
        "rank",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["occurrence_count"].min() > 0
    assert df["rank"].min() == 1


def test_hype_cycle_candidates(config):
    """Test hype cycle candidates identification."""
    file_path = (
        Path(config["output"]["case_study_5"]["fashionable"])
        / "hype_cycle_candidates.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Hype cycle file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "trait",
        "peak_year",
        "peak_count",
        "decline_rate",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    if len(df) > 0:
        assert df["peak_count"].min() > 0


# ==== Phase 6: Pleiotropy awareness tests ====


def test_pleiotropy_awareness_output_exists(config):
    """Test that pleiotropy awareness outputs exist."""
    base_path = Path(config["output"]["case_study_5"]["pleiotropy"])

    if not base_path.exists():
        pytest.skip(f"Pleiotropy directory not found: {base_path}")

    expected_files = [
        "canonical_pair_trends.csv",
        "hotspot_trait_trends.csv",
        "mr_presso_impact.csv",
        "pleiotropy_metadata.json",
    ]

    for file_name in expected_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_canonical_pair_trends_structure(config):
    """Test canonical pair trends CSV structure."""
    file_path = (
        Path(config["output"]["case_study_5"]["pleiotropy"])
        / "canonical_pair_trends.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Canonical pairs file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "exposure",
        "outcome",
        "era",
        "study_count",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["study_count"].min() >= 0


def test_mr_presso_impact_structure(config):
    """Test MR-PRESSO impact analysis structure."""
    file_path = (
        Path(config["output"]["case_study_5"]["pleiotropy"])
        / "mr_presso_impact.csv"
    )

    if not file_path.exists():
        pytest.skip(f"MR-PRESSO impact file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "period",
        "avg_outcomes_per_study",
        "study_count",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert "pre_mr_presso" in df["period"].values
    assert "post_mr_presso" in df["period"].values


# ==== Phase 7: Winner's curse tests ====


def test_winners_curse_output_exists(config):
    """Test that winner's curse outputs exist."""
    base_path = Path(config["output"]["case_study_5"]["winners_curse"])

    if not base_path.exists():
        pytest.skip(f"Winner's curse directory not found: {base_path}")

    expected_files = [
        "stage1_summary_statistics.csv",
        "stage2_effect_size_decline.csv",
        "stage2_temporal_model.csv",
        "stage2_subgroup_analysis.csv",
        "winners_curse_metadata.json",
    ]

    for file_name in expected_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Missing file: {file_path}"


def test_stage1_summary_statistics(config):
    """Test Stage 1 summary statistics structure."""
    file_path = (
        Path(config["output"]["case_study_5"]["winners_curse"])
        / "stage1_summary_statistics.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Stage 1 summary not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "exposure",
        "outcome",
        "total_studies",
        "effect_size_availability",
        "avg_effect_size",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert df["total_studies"].min() >= 2
    assert df["effect_size_availability"].between(0, 1).all()


def test_stage2_effect_size_decline(config):
    """Test Stage 2 effect size decline analysis."""
    file_path = (
        Path(config["output"]["case_study_5"]["winners_curse"])
        / "stage2_effect_size_decline.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Stage 2 decline file not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "exposure",
        "outcome",
        "initial_effect_size",
        "final_effect_size",
        "percent_decline",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    if len(df) > 0:
        assert df["initial_effect_size"].notna().any()


def test_stage2_temporal_model(config):
    """Test Stage 2 temporal regression model."""
    file_path = (
        Path(config["output"]["case_study_5"]["winners_curse"])
        / "stage2_temporal_model.csv"
    )

    if not file_path.exists():
        pytest.skip(f"Stage 2 model not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = [
        "coefficient",
        "std_err",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert "study_sequence" in df["variable"].values
    assert df["p_value"].between(0, 1).all()


# ==== Figure generation tests ====


def test_figures_exist(config):
    """Test that expected figures are generated."""
    figures_path = Path(config["output"]["case_study_5"]["figures"])

    if not figures_path.exists():
        pytest.skip(f"Figures directory not found: {figures_path}")

    expected_figures = [
        "trait_diversity_over_time",
        "trait_diversity_by_era",
        "concordance_over_time",
        "concordance_by_match_type",
        "completeness_over_time",
        "completeness_by_field_type",
        "strobe_impact",
        "strobe_reporting_impact",
    ]

    for fig_name in expected_figures:
        png_path = figures_path / f"{fig_name}.png"
        svg_path = figures_path / f"{fig_name}.svg"

        assert png_path.exists() or svg_path.exists(), (
            f"Missing figure: {fig_name}"
        )


def test_figure_formats(config):
    """Test that figures are generated in multiple formats."""
    figures_path = Path(config["output"]["case_study_5"]["figures"])

    if not figures_path.exists():
        pytest.skip(f"Figures directory not found: {figures_path}")

    png_files = list(figures_path.glob("*.png"))
    svg_files = list(figures_path.glob("*.svg"))

    assert len(png_files) > 0, "No PNG figures generated"
    assert len(svg_files) > 0, "No SVG figures generated"
