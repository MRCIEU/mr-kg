"""Consolidated five-series contracts, frozen guards and candidate promotion."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import vl_convert

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts/analysis"))
import case_study_5_consolidated_figures as figures


def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create source marginals and valid A/B inputs without database access."""
    rows = []
    for year in (2003, 2005):
        row = {"pub_year": year, "n_studies": 10}
        for field, k in zip(figures.accepted.FIELDS, (2, 4, 1, 8, 0, 5, 9)):
            row[f"{field}_n"] = k
            row[f"{field}_pct"] = 10.0 * k
        rows.append(row)
    temporal = pd.DataFrame(
        {"publication_year": [2003, 2005], "paper_count": [2, 3]}
    )
    diversity = pd.DataFrame(
        {
            "pub_year": [2005],
            "mean_trait_count": [2.5],
            "std_trait_count": [1.2],
            "n_studies": [10],
        }
    )
    res = (pd.DataFrame(rows), temporal, diversity)
    return res


def baseline_fixture(
    root: Path,
) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Prepare a real guard baseline with synthetic audited data/specifications."""
    counts, temporal, diversity = synthetic_inputs()
    config = {"case_study_5": {"temporal_eras": {"early_mr": [2003, 2025]}}}
    eras = figures.main_figures.get_era_transitions(
        config["case_study_5"]["temporal_eras"]
    )
    for path in (
        figures.MAIN_PATH,
        figures.COUNTS_PATH,
        figures.CONFIG_PATH,
        figures.AUDIT_PATH,
        figures.BASELINE_PATH,
        figures.PRIOR_CSV_PATH,
    ):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / figures.MAIN_PATH).write_bytes(b"original-main")
    counts.to_csv(root / figures.COUNTS_PATH, index=False)
    (root / figures.CONFIG_PATH).write_text("config")
    audit = {
        "status": "passed",
        "inputs_sha256": {
            str(p): figures.file_hash(root / p)
            for p in (
                figures.MAIN_PATH,
                figures.COUNTS_PATH,
                figures.CONFIG_PATH,
            )
        },
    }
    (root / figures.AUDIT_PATH).write_text(json.dumps(audit))
    prior = figures.accepted.prepare_interval_data(counts)
    prior.to_csv(root / figures.PRIOR_CSV_PATH, index=False)
    (
        root / figures.TABLE_DIR / "cs5_completeness_ci_provenance.json"
    ).write_text(json.dumps({"methods": {"limits": "synthetic test"}}))
    baseline = {
        "original_main_sha256": figures.file_hash(root / figures.MAIN_PATH),
        "preserved_delivery_sha256": {
            str(figures.PRIOR_CSV_PATH): figures.file_hash(
                root / figures.PRIOR_CSV_PATH
            )
        },
        "subplot_A": figures.main_figures.create_temporal_distribution_subplot(
            temporal, eras
        ).to_dict(),
        "subplot_B": figures.main_figures.create_diversity_subplot(
            diversity, eras
        ).to_dict(),
    }
    source = ast.parse(Path(figures.main_figures.__file__).read_text())
    baseline["protected_function_ast_sha256"] = {
        n.name: hashlib.sha256(
            ast.dump(n, include_attributes=False).encode()
        ).hexdigest()
        for n in source.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "create_temporal_distribution_subplot",
            "create_diversity_subplot",
            "create_era_markers",
            "calculate_log_ticks",
        )
    }
    for companion in ("C", "D"):
        column = "field" if companion == "C" else "field_type"
        rows = [
            {
                "pub_year": r.pub_year,
                column: "95% CI"
                if r.metric == "Confidence interval"
                else r.metric,
                "completeness_pct": r.completeness_pct,
            }
            for r in prior[prior.companion == companion].itertuples()
        ]
        baseline[f"old_{companion}"] = {
            "layer": [{"data": {"name": "old"}}],
            "datasets": {"old": rows},
        }
    (root / figures.BASELINE_PATH).write_text(json.dumps(baseline))
    res = (baseline, audit, counts, temporal, diversity, config)
    return res


def test_five_series_point_and_interval_invariance(tmp_path: Path) -> None:
    """Match all old C/D source points, interval rows and A/B specifications."""
    baseline, _, counts, temporal, diversity, config = baseline_fixture(
        tmp_path
    )
    eras = figures.main_figures.get_era_transitions(
        config["case_study_5"]["temporal_eras"]
    )
    data = figures.verify_invariance(
        tmp_path, baseline, counts, temporal, diversity, eras
    )
    assert len(data) == 10
    assert data.metric.drop_duplicates().tolist() == list(figures.SELECTION)
    assert data[data.metric == "Effect size"].completeness_pct.tolist() == [
        40.0,
        40.0,
    ]
    assert data[data.metric == "Effect size"].n_reported.tolist() == [4, 4]
    assert data[
        data.metric == "Effect size"
    ].interval_method.unique().tolist() == [
        "bonferroni_clopper_pearson_max_envelope"
    ]
    assert "effect_size_pct" not in counts
    baseline["subplot_B"]["height"] = 201
    with pytest.raises(ValueError, match="Protected subplot_B"):
        figures.verify_invariance(
            tmp_path, baseline, counts, temporal, diversity, eras
        )


def test_historical_point_changes_rejected(tmp_path: Path) -> None:
    """The proof compares frozen original plotting datasets, not only new code."""
    baseline, _, counts, temporal, diversity, config = baseline_fixture(
        tmp_path
    )
    next(
        r
        for r in baseline["old_D"]["datasets"]["old"]
        if r["field_type"] == "Direction"
    )["completeness_pct"] = 12.34
    eras = figures.main_figures.get_era_transitions(
        config["case_study_5"]["temporal_eras"]
    )
    with pytest.raises(AssertionError):
        figures.verify_invariance(
            tmp_path, baseline, counts, temporal, diversity, eras
        )


def test_five_facet_filter_palette_and_linear_bounds() -> None:
    """Inherit facet data, use common palette and compile a 2x3 layout."""
    counts, _, _ = synthetic_inputs()
    data = figures.select_combined_rows(
        figures.accepted.prepare_interval_data(counts)
    )
    spec = figures.create_combined_companion(data, [{"year": 2003}]).to_dict()
    assert spec["columns"] == 2
    # Put axes on actual facets, never in the unoccupied sixth cell.
    assert spec["resolve"]["axis"]["x"] == "independent"
    assert spec["facet"]["sort"] == list(figures.SELECTION)
    colors = figures.main_figures.get_completeness_field_colors()
    for layer in (spec["spec"]["layer"][0], spec["spec"]["layer"][2]):
        assert "data" not in layer
        assert layer["mark"]["interpolate"] == "linear"
        assert layer["mark"]["clip"] is True
        assert layer["encoding"]["color"]["scale"] == {
            "domain": list(colors),
            "range": list(colors.values()),
        }
    assert spec["spec"]["layer"][0]["encoding"]["y2"]["field"] == "upper_pct"
    assert spec["spec"]["layer"][0]["encoding"]["y"]["scale"]["domain"] == [
        0,
        100,
    ]
    vega = vl_convert.vegalite_to_vega(spec)
    assert '"loess"' not in json.dumps(vega)
    assert '"regression"' not in json.dumps(vega)


def test_candidate_publication_and_successor_guard(tmp_path: Path) -> None:
    """Exercise actual PNG/CSV writes, preserved history and a guarded rerun."""
    baseline, audit, counts, temporal, diversity, config = baseline_fixture(
        tmp_path
    )
    figures.validate_baseline(tmp_path)
    eras = figures.main_figures.get_era_transitions(
        config["case_study_5"]["temporal_eras"]
    )
    data = figures.verify_invariance(
        tmp_path, baseline, counts, temporal, diversity, eras
    )
    original_audit = (tmp_path / figures.AUDIT_PATH).read_bytes()
    historical_csv = (tmp_path / figures.PRIOR_CSV_PATH).read_bytes()
    figures.publish_candidates(
        tmp_path,
        tmp_path / "candidates",
        data,
        temporal,
        diversity,
        counts,
        config,
        audit,
    )
    assert (tmp_path / figures.AUDIT_PATH).read_bytes() == original_audit
    assert (tmp_path / figures.PRIOR_CSV_PATH).read_bytes() == historical_csv
    assert len(pd.read_csv(tmp_path / figures.COMBINED_CSV_PATH)) == 10
    assert (tmp_path / figures.MAIN_PATH).read_bytes().startswith(b"\x89PNG")
    provenance = json.loads((tmp_path / figures.PROVENANCE_PATH).read_text())
    for path, digest in provenance["outputs_sha256"].items():
        assert figures.file_hash(tmp_path / path) == digest
    figures.validate_baseline(
        tmp_path
    )  # explicit successor hash, not original
    (tmp_path / figures.MAIN_PATH).write_bytes(b"unapproved-main")
    with pytest.raises(ValueError, match="approved successor"):
        figures.validate_baseline(tmp_path)


def test_guard_rejects_scientific_and_historical_changes(
    tmp_path: Path,
) -> None:
    """Reject corpus/config drift and unrecorded main/old delivery changes."""
    baseline_fixture(tmp_path)
    for path in (
        figures.COUNTS_PATH,
        figures.CONFIG_PATH,
        figures.PRIOR_CSV_PATH,
        figures.MAIN_PATH,
    ):
        before = (tmp_path / path).read_bytes()
        (tmp_path / path).write_bytes(b"changed")
        with pytest.raises(ValueError):
            figures.validate_baseline(tmp_path)
        (tmp_path / path).write_bytes(before)


def test_main_has_three_panels_and_no_C_intervals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Assembly preserves the two 200-high panels and one locally taller C."""
    counts, temporal, diversity = synthetic_inputs()
    specs = []
    monkeypatch.setattr(
        figures.alt.VConcatChart,
        "save",
        lambda self, *a, **kw: specs.append(self.to_dict()),
    )
    figures.main_figures.create_faceted_figure(
        temporal, diversity, counts, {"early_mr": [2003, 2025]}, tmp_path
    )
    spec = specs[0]
    assert len(spec["vconcat"]) == 3
    assert [s["height"] for s in spec["vconcat"]] == [200, 200, 400]
    assert (
        spec["vconcat"][2]["title"]
        == "C. Reporting completeness by Field Type"
    )
    assert len(spec["vconcat"][2]["layer"]) == 2
    assert len(spec["vconcat"][2]["layer"][1]["layer"]) == 2


def test_render_failure_does_not_promote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed companion renderer cannot replace the canonical main."""
    baseline, audit, counts, temporal, diversity, config = baseline_fixture(
        tmp_path
    )
    data = figures.select_combined_rows(
        figures.accepted.prepare_interval_data(counts)
    )
    before = (tmp_path / figures.MAIN_PATH).read_bytes()

    def fail(*args: object) -> None:
        raise RuntimeError("simulated renderer failure")

    monkeypatch.setattr(figures, "create_combined_companion", fail)
    with pytest.raises(RuntimeError, match="simulated renderer failure"):
        figures.publish_candidates(
            tmp_path,
            tmp_path / "candidates",
            data,
            temporal,
            diversity,
            counts,
            config,
            audit,
        )
    assert (tmp_path / figures.MAIN_PATH).read_bytes() == before
    assert not (tmp_path / figures.PROVENANCE_PATH).exists()
    assert (
        figures.file_hash(tmp_path / figures.MAIN_PATH)
        == baseline["original_main_sha256"]
    )


def test_dry_run_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate real guards/invariance without creating candidate artifacts."""
    _, _, counts, temporal, diversity, config = baseline_fixture(tmp_path)
    config["output"] = {"case_study_5": {"base": "unused"}}
    monkeypatch.setattr(figures, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(figures.main_figures, "load_config", lambda p: config)
    monkeypatch.setattr(
        figures.main_figures, "load_temporal_stats", lambda p: temporal
    )
    monkeypatch.setattr(
        figures.main_figures, "load_diversity_data", lambda p: diversity
    )
    monkeypatch.setattr(
        figures.main_figures, "load_completeness_data", lambda p: counts
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["script", "--dry-run", "--candidate-dir", str(tmp_path / "never")],
    )
    before = {
        str(p.relative_to(tmp_path)): figures.file_hash(p)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    figures.main()
    after = {
        str(p.relative_to(tmp_path)): figures.file_hash(p)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert before == after
    assert not (tmp_path / "never").exists()
