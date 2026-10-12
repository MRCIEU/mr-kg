"""Check supplemental counts, coverage construction and presentation contracts."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import vl_convert

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts/analysis"))
import case_study_5_supplementary_completeness_figures as figures


def counts_frame(years: tuple[int, ...] = (2003, 2005)) -> pd.DataFrame:
    """Build complete source columns with correlated marginal field counts."""
    rows = []
    for year in years:
        row = {"pub_year": year, "n_studies": 10}
        for field, k in zip(figures.FIELDS, (2, 4, 1, 8, 0, 5, 9)):
            row[f"{field}_n"] = k
            row[f"{field}_pct"] = k * 10.0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "k,n,expected",
    [
        (5, 10, (0.2365930905, 0.7634069095)),
        (0, 10, (0.0, 0.2775327999)),
        (10, 10, (0.7224672001, 1.0)),
    ],
)
def test_wilson_known_cases(k: int, n: int, expected: tuple) -> None:
    """Wilson has nonzero uncertainty at zero and all-success boundaries."""
    assert figures.wilson_interval(k, n) == pytest.approx(expected)


@pytest.mark.parametrize("n", [1, 2, 10])
def test_exact_boundaries(n: int) -> None:
    """Exact intervals implement alpha/2 tails, including very small years."""
    alpha = 0.05 / 3
    assert figures.exact_interval(0, n) == pytest.approx(
        (0, 1 - (alpha / 2) ** (1 / n))
    )
    assert figures.exact_interval(n, n) == pytest.approx(
        ((alpha / 2) ** (1 / n), 1)
    )


@pytest.mark.parametrize(
    "counts,n",
    [((0, 0, 0), 1), ((1, 1, 0), 1), ((2, 2, 2), 10), ((2, 4, 1), 10)],
)
def test_max_envelope_and_ties(counts: tuple, n: int) -> None:
    """The max envelope reselects both endpoints and is permutation invariant."""
    intervals = [figures.exact_interval(k, n) for k in counts]
    expected = (max(i[0] for i in intervals), max(i[1] for i in intervals))
    assert figures.max_effect_interval(counts, n) == expected
    assert figures.max_effect_interval(counts[::-1], n) == expected
    assert 0 <= expected[0] <= max(counts) / n <= expected[1] <= 1


@pytest.mark.parametrize(
    "k,n",
    [
        (0, 0),
        (-1, 10),
        (11, 10),
        (1.5, 10),
        (1, 10.5),
        (float("nan"), 10),
        (True, 10),
    ],
)
def test_invalid_interval_counts(k: float, n: float) -> None:
    """Reject invalid events and denominators before statistical calculation."""
    with pytest.raises(ValueError):
        figures.wilson_interval(k, n)
    with pytest.raises(ValueError):
        figures.exact_interval(k, n)


def test_max_requires_three_components() -> None:
    """The multiplicity correction applies to exactly the three source fields."""
    with pytest.raises(ValueError):
        figures.max_effect_interval((2, 4), 10)


def test_prepare_preserves_values_and_duplicate_metrics() -> None:
    """C/D use the frozen definitions, not effect unions or statistical unions."""
    source = counts_frame()
    data = figures.prepare_interval_data(source)
    assert len(data) == 14
    assert data.n_studies.tolist() == [10] * 14
    for year in source.pub_year:
        c = data[(data.companion == "C") & (data.pub_year == year)].set_index(
            "metric"
        )
        d = data[(data.companion == "D") & (data.pub_year == year)].set_index(
            "metric"
        )
        assert c.completeness_pct.to_dict() == {
            "P-value": 50,
            "Confidence interval": 80,
            "Standard error": 0,
        }
        assert d.completeness_pct.to_dict() == {
            "Confidence Interval": 80,
            "Direction": 90,
            "Effect Size": 40,
            "Statistical": 50,
        }
        for c_label, d_label in [
            ("P-value", "Statistical"),
            ("Confidence interval", "Confidence Interval"),
        ]:
            for column in ("lower_pct", "upper_pct", "n_reported"):
                assert c.loc[c_label, column] == d.loc[d_label, column]
        assert d.loc["Effect Size", "n_reported"] == 4
        assert (
            d.loc["Effect Size", "interval_method"]
            == "bonferroni_clopper_pearson_max_envelope"
        )
        assert d.loc["Effect Size", "effect_size_beta_n"] == 2
        assert d.loc["Effect Size", "effect_size_or_n"] == 4
        assert d.loc["Effect Size", "effect_size_hr_n"] == 1
    pd.testing.assert_frame_equal(source, counts_frame())


def test_zero_denominator_and_gaps_are_absent_not_zero() -> None:
    """Do not calculate a percentage for empty years or impute absent years."""
    source = counts_frame((2003, 2004, 2006))
    source.loc[1, "n_studies"] = 0
    for f in figures.FIELDS:
        source.loc[1, f"{f}_n"] = 0
        source.loc[1, f"{f}_pct"] = float("nan")
    data = figures.prepare_interval_data(source)
    assert sorted(data.pub_year.unique()) == [2003, 2006]
    assert 2005 not in data.pub_year.tolist()


@pytest.mark.parametrize(
    "column,value",
    [
        ("n_studies", -1),
        ("n_studies", 1.5),
        ("effect_size_beta_n", 1.5),
        ("effect_size_beta_n", 11),
        ("pub_year", 2026),
        ("pub_year", 2003.5),
        ("p_value_pct", 51),
    ],
)
def test_source_validation(column: str, value: float) -> None:
    """Reject impossible integer counts and altered source percentages."""
    source = counts_frame()
    source[column] = source[column].astype(float)
    source.loc[0, column] = value
    with pytest.raises(ValueError):
        figures.prepare_interval_data(source)


def test_missing_duplicate_and_empty_sources() -> None:
    """Require the full source contract and at least one annual observation."""
    for source in [
        counts_frame().drop(columns="p_value_n"),
        counts_frame((2003, 2003)),
        counts_frame().iloc[:0],
    ]:
        with pytest.raises(ValueError):
            figures.prepare_interval_data(source)
    source = counts_frame()
    for col in source.columns:
        if col != "pub_year":
            source[col] = 0
    with pytest.raises(ValueError):
        figures.prepare_interval_data(source)


@pytest.mark.parametrize(
    "companion,columns,n_metrics", [("C", 1, 3), ("D", 2, 4)]
)
def test_facet_contract_and_compilation(
    companion: str, columns: int, n_metrics: int
) -> None:
    """Bound layers compile without statistical smoothing or imputation."""
    data = figures.prepare_interval_data(counts_frame())
    chart = figures.create_companion(
        data, companion, [{"year": 2003}, {"year": 2021}]
    )
    spec = chart.to_dict()
    assert spec["columns"] == columns
    # Explicit data on child layers would bypass the per-metric facet filter.
    assert "data" not in spec["spec"]["layer"][0]
    assert "data" not in spec["spec"]["layer"][2]
    assert spec["spec"]["layer"][0]["mark"]["interpolate"] == "linear"
    assert spec["spec"]["layer"][2]["mark"]["interpolate"] == "linear"
    assert spec["spec"]["layer"][2]["mark"]["strokeJoin"] == "round"
    assert spec["spec"]["layer"][2]["mark"]["clip"] is True
    if companion == "C":
        original_colors = figures.get_field_type_colors(use_gruvbox=True)
        assert spec["spec"]["layer"][0]["encoding"]["color"]["scale"][
            "range"
        ] == [
            original_colors["P-value"],
            original_colors["95% CI"],
            original_colors["SE"],
        ]
    assert len(spec["facet"]["sort"]) == n_metrics
    assert spec["spec"]["layer"][0]["encoding"]["y"]["field"] == "lower_pct"
    assert spec["spec"]["layer"][0]["encoding"]["y2"]["field"] == "upper_pct"
    assert (
        spec["spec"]["layer"][2]["encoding"]["y"]["field"]
        == "completeness_pct"
    )
    assert spec["spec"]["layer"][0]["encoding"]["y"]["scale"]["domain"] == [
        0,
        100,
    ]
    compiled = vl_convert.vegalite_to_vega(spec, vl_version="6.1")
    assert isinstance(compiled, dict)
    assert '"loess"' not in json.dumps(compiled)
    assert '"regression"' not in json.dumps(compiled)
    assert "n=10" in " ".join(spec["title"]["subtitle"])


def audited_root(tmp_path: Path) -> dict:
    """Create a synthetic, explicitly audited input baseline, with no DB access."""
    for path in (figures.COUNTS_PATH, figures.CONFIG_PATH, figures.MAIN_PATH):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("baseline")
    counts_frame().to_csv(tmp_path / figures.COUNTS_PATH, index=False)
    (tmp_path / figures.CONFIG_PATH).write_text(
        "case_study_5:\n  temporal_eras:\n    early_mr: [2003, 2025]\n"
    )
    res = {
        "status": "passed",
        "checks": {
            "annual_and_era_counts_match_frozen_predicate": True,
            "retained_png_reproduced_byte_identically": True,
            "installed_manuscript_png_matches": True,
        },
        "inputs_sha256": {
            str(p): figures.file_hash(tmp_path / p)
            for p in (
                figures.COUNTS_PATH,
                figures.CONFIG_PATH,
                figures.MAIN_PATH,
            )
        },
    }
    return res


def test_audit_fails_on_baseline_change(tmp_path: Path) -> None:
    """Changed counts/config/main PNG require explicit baseline reconciliation."""
    audit = audited_root(tmp_path)
    figures.verify_audit(audit, tmp_path)
    for path in (figures.COUNTS_PATH, figures.CONFIG_PATH, figures.MAIN_PATH):
        original = (tmp_path / path).read_bytes()
        (tmp_path / path).write_text("changed")
        with pytest.raises(ValueError, match="Audited input changed"):
            figures.verify_audit(audit, tmp_path)
        (tmp_path / path).write_bytes(original)
    with pytest.raises(ValueError, match="passed baseline audit"):
        figures.verify_audit({"status": "failed"}, tmp_path)


def test_dry_run_is_no_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate counts without creating directories, assets or provenance."""
    audit = audited_root(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(figures, "PROJECT_ROOT", tmp_path)
    out = tmp_path / "uncreated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "script",
            "--dry-run",
            "--audit-json",
            str(audit_path),
            "--output-dir",
            str(out),
            "--table-dir",
            str(out),
        ],
    )
    main_hash = figures.file_hash(tmp_path / figures.MAIN_PATH)
    figures.main()
    assert not out.exists()
    assert figures.file_hash(tmp_path / figures.MAIN_PATH) == main_hash


def test_write_path_preserves_main_and_records_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render real temporary PNGs and verify the CSV/provenance contract."""
    audit = audited_root(tmp_path)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(figures, "PROJECT_ROOT", tmp_path)
    images = tmp_path / "figures"
    tables = tmp_path / "tables"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "script",
            "--audit-json",
            str(audit_path),
            "--output-dir",
            str(images),
            "--table-dir",
            str(tables),
        ],
    )
    before = figures.file_hash(tmp_path / figures.MAIN_PATH)
    figures.main()
    assert figures.file_hash(tmp_path / figures.MAIN_PATH) == before
    for filename in figures.FILENAMES.values():
        assert (
            (images / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        )
    csv = tables / "cs5_completeness_annual_ci.csv"
    assert len(pd.read_csv(csv)) == 14
    provenance = json.loads(
        (tables / "cs5_completeness_ci_provenance.json").read_text()
    )
    assert provenance["main_png_sha256_before_and_after"] == before
    assert provenance["baseline_audit_sha256"] == figures.file_hash(audit_path)
    assert provenance["source_counts_sha256"] == figures.file_hash(
        tmp_path / figures.COUNTS_PATH
    )
    for relative_path, digest in provenance["outputs_sha256"].items():
        assert figures.file_hash(tmp_path / relative_path) == digest
