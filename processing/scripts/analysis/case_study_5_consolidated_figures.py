"""Consolidate Figure 4C/D and emit one five-facet CI companion.

Render candidates first and promote only after frozen scientific input,
A/B specification, annual point and interval invariance checks pass.
Accepted interval calculations are imported, never reimplemented here.
"""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import vl_convert

import case_study_5_manuscript_figures as main_figures
import case_study_5_supplementary_completeness_figures as accepted

PROJECT_ROOT = main_figures.PROJECT_ROOT
MAIN_PATH = accepted.MAIN_PATH
COUNTS_PATH = accepted.COUNTS_PATH
CONFIG_PATH = accepted.CONFIG_PATH
TABLE_DIR = Path("data/artifacts/manuscript-tables")
FIGURE_DIR = Path("data/artifacts/manuscript-figures")
BASELINE_PATH = TABLE_DIR / "cs5_consolidated_presentation_baseline.json"
AUDIT_PATH = TABLE_DIR / "cs5_completeness_baseline_audit.json"
PRIOR_CSV_PATH = TABLE_DIR / "cs5_completeness_annual_ci.csv"
PROVENANCE_PATH = TABLE_DIR / "cs5_consolidated_presentation_provenance.json"
COMBINED_FIGURE_PATH = (
    FIGURE_DIR / "cs5_fig_completeness_by_field_type_combined_ci.png"
)
COMBINED_CSV_PATH = TABLE_DIR / "cs5_completeness_combined_annual_ci.csv"
SELECTION = {
    "Direction": ("D", "Direction"),
    "Effect size": ("D", "Effect Size"),
    "Standard error": ("C", "Standard error"),
    "95% CI": ("C", "Confidence interval"),
    "P-Value": ("C", "P-value"),
}


# ==== Frozen scientific and presentation guards ====


def file_hash(path: Path) -> str:
    """Hash large scientific inputs without loading them into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    res = h.hexdigest()
    return res


def validate_baseline(root: Path) -> tuple[dict, dict]:
    """Accept only the original or an explicitly recorded successor main PNG."""
    baseline = json.loads((root / BASELINE_PATH).read_text())
    audit = json.loads((root / AUDIT_PATH).read_text())
    if audit.get("status") != "passed":
        raise ValueError("A passed original scientific audit is required")
    # Original audit/source paths remain frozen, except authorized main edits.
    for relative, expected in audit["inputs_sha256"].items():
        if (
            relative == str(MAIN_PATH)
            or relative
            == "processing/scripts/analysis/case_study_5_manuscript_figures.py"
        ):
            continue
        if file_hash(root / relative) != expected:
            raise ValueError(f"Scientific baseline changed: {relative}")
    for relative, expected in baseline["preserved_delivery_sha256"].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"Historical delivery changed: {relative}")
    current = file_hash(root / MAIN_PATH)
    if current != baseline["original_main_sha256"]:
        path = root / PROVENANCE_PATH
        if not path.exists():
            raise ValueError(
                "Changed main requires successor presentation record"
            )
        successor = json.loads(path.read_text())
        if (
            successor.get("validation", {}).get("invariance_passed")
            is not True
            or successor.get("original_audit_sha256")
            != file_hash(root / AUDIT_PATH)
            or successor.get("presentation_baseline_sha256")
            != file_hash(root / BASELINE_PATH)
            or successor.get("outputs_sha256", {}).get(str(MAIN_PATH))
            != current
        ):
            raise ValueError(
                "Main does not match approved successor presentation"
            )
    res = (baseline, audit)
    return res


# ==== Five-series data and invariance ====


def select_combined_rows(prior: pd.DataFrame) -> pd.DataFrame:
    """Select five unique source metrics; retain every numerical column."""
    frames = []
    for label, (companion, metric) in SELECTION.items():
        frame = prior[
            (prior.companion == companion) & (prior.metric == metric)
        ].copy()
        if frame.empty or frame.pub_year.duplicated().any():
            raise ValueError(f"Missing or duplicate source series: {metric}")
        frame["companion"] = "combined"
        frame["metric"] = label
        frames.append(frame)
    res = pd.concat(frames, ignore_index=True)
    return res


def verify_invariance(
    root: Path,
    baseline: dict,
    counts: pd.DataFrame,
    temporal: pd.DataFrame,
    diversity: pd.DataFrame,
    eras: list[dict],
) -> pd.DataFrame:
    """Check exact A/B specs and 115 source points/previous interval rows."""
    source = ast.parse(Path(main_figures.__file__).read_text())
    for node in source.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in baseline["protected_function_ast_sha256"]
        ):
            digest = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            if digest != baseline["protected_function_ast_sha256"][node.name]:
                raise ValueError(f"Protected helper changed: {node.name}")
    current_specs = {
        "subplot_A": main_figures.create_temporal_distribution_subplot(
            temporal, eras
        ).to_dict(),
        "subplot_B": main_figures.create_diversity_subplot(
            diversity, eras
        ).to_dict(),
    }
    for key, current in current_specs.items():
        if current != baseline[key]:
            raise ValueError(f"Protected {key} specification changed")
        if vl_convert.vegalite_to_vega(current) != vl_convert.vegalite_to_vega(
            baseline[key]
        ):
            raise ValueError(f"Protected {key} compiled Vega changed")
    data = select_combined_rows(accepted.prepare_interval_data(counts))
    prior = select_combined_rows(pd.read_csv(root / PRIOR_CSV_PATH))
    pd.testing.assert_frame_equal(
        data, prior, check_exact=False, atol=1e-10, rtol=0
    )
    points = main_figures.prepare_field_type_yearly_data(counts)
    historical_points = []
    for label, (companion, metric) in SELECTION.items():
        spec = baseline[f"old_{companion}"]
        dataset = spec["datasets"][spec["layer"][0]["data"]["name"]]
        column = "field" if companion == "C" else "field_type"
        old_label = "95% CI" if metric == "Confidence interval" else metric
        historical_points.extend(
            {
                "pub_year": row["pub_year"],
                "field_type": label,
                "completeness_pct": row["completeness_pct"],
            }
            for row in dataset
            if row[column] == old_label
        )
    keys = ["pub_year", "field_type"]
    pd.testing.assert_frame_equal(
        points.sort_values(keys).reset_index(drop=True),
        pd.DataFrame(historical_points)
        .sort_values(keys)
        .reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    joined = points.merge(
        data[["pub_year", "metric", "completeness_pct"]],
        left_on=["pub_year", "field_type"],
        right_on=["pub_year", "metric"],
        validate="one_to_one",
        suffixes=("_main", "_supp"),
    )
    if (
        len(joined) != len(data)
        or not (
            joined.completeness_pct_main == joined.completeness_pct_supp
        ).all()
    ):
        raise ValueError("Main/companion annual estimates differ")
    res = data
    return res


# ==== Single five-facet companion ====


def create_combined_companion(
    data: pd.DataFrame, eras: list[dict]
) -> alt.FacetChart:
    """Use two columns/three rows, inheriting metric-filtered parent data."""
    colors = main_figures.get_completeness_field_colors()
    base = alt.Chart().encode(
        x=alt.X(
            "pub_year:Q",
            title="Publication year",
            scale=alt.Scale(domain=[2003, 2025], nice=False),
            axis=alt.Axis(values=[2003, 2008, 2013, 2018, 2025], format="d"),
        ),
        color=alt.Color(
            "metric:N",
            legend=None,
            scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        ),
    )
    band = base.mark_area(
        opacity=0.18, interpolate="linear", clip=True
    ).encode(
        y=alt.Y(
            "lower_pct:Q",
            title="Reporting (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        y2="upper_pct:Q",
    )
    line = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, size=25),
        interpolate="linear",
        strokeJoin="round",
        clip=True,
    ).encode(
        y=alt.Y(
            "completeness_pct:Q",
            title="Reporting (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        tooltip=[
            "metric:N",
            "pub_year:Q",
            "n_studies:Q",
            "n_reported:Q",
            alt.Tooltip("completeness_pct:Q", format=".2f"),
            alt.Tooltip("lower_pct:Q", format=".2f"),
            alt.Tooltip("upper_pct:Q", format=".2f"),
        ],
    )
    rules = (
        alt.Chart(pd.DataFrame(eras))
        .mark_rule(
            color="#888888",
            strokeDash=[3, 3],
            opacity=0.6,
        )
        .encode(x="year:Q")
    )
    sparse = data.drop_duplicates("pub_year")
    sparse = sparse[sparse.n_studies < 20]
    n_note = "; ".join(
        f"{int(r.pub_year)}: n={int(r.n_studies)}" for r in sparse.itertuples()
    )
    subtitles = [
        "Annual observations and pointwise 95% intervals (no fitted smoother)."
    ]
    if n_note:
        subtitles.append(n_note)
    subtitles.append(
        "Effect size: max(beta, OR, HR); conservative exact max envelope."
    )
    res = (
        alt.layer(band, rules, line, data=data)
        .properties(
            width=260,
            height=150,
        )
        .facet(
            facet=alt.Facet(
                "metric:N",
                sort=list(colors),
                header=alt.Header(
                    title=None, labelFontSize=15, labelFontWeight="bold"
                ),
            ),
            columns=2,
            spacing=20,
        )
        .properties(
            title=alt.TitleParams(
                text="Reporting completeness by Field Type",
                subtitle=subtitles,
                anchor="start",
                fontSize=17,
                subtitleFontSize=12,
            )
        )
        .resolve_axis(x="independent")
        .configure_axis(
            labelFontSize=13, titleFontSize=14, titlePadding=10, labelAngle=0
        )
        .configure_view(stroke=None)
    )
    return res


# ==== Candidate rendering and guarded publication ====


def parse_args() -> argparse.Namespace:
    """Parse no-write preview and an optional retained candidate directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    # ---- --dry-run ----
    parser.add_argument("-n", "--dry-run", action="store_true")
    # ---- --candidate-dir ----
    parser.add_argument("--candidate-dir", type=Path)
    res = parser.parse_args()
    return res


def publish_candidates(
    root: Path,
    candidate_dir: Path,
    data: pd.DataFrame,
    temporal: pd.DataFrame,
    diversity: pd.DataFrame,
    counts: pd.DataFrame,
    config: dict,
    audit: dict,
) -> None:
    """Render all candidates, then publish an explicit successor record."""
    before = file_hash(root / MAIN_PATH)
    era_defs = config["case_study_5"]["temporal_eras"]
    eras = main_figures.get_era_transitions(era_defs)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    main_figures.create_faceted_figure(
        temporal, diversity, counts, era_defs, candidate_dir
    )
    supp = candidate_dir / COMBINED_FIGURE_PATH.name
    create_combined_companion(data, eras).save(str(supp), ppi=300)
    csv = candidate_dir / COMBINED_CSV_PATH.name
    data.to_csv(csv, index=False)
    # Generation must not affect canonical main before the promotion boundary.
    if file_hash(root / MAIN_PATH) != before:
        raise RuntimeError("Canonical main changed during candidate rendering")
    # Recheck protected inputs/delivery immediately before publication.
    baseline, _ = validate_baseline(root)
    verify_invariance(root, baseline, counts, temporal, diversity, eras)
    outputs = {
        MAIN_PATH: candidate_dir / MAIN_PATH.name,
        COMBINED_FIGURE_PATH: supp,
        COMBINED_CSV_PATH: csv,
    }
    provenance = {
        "schema_version": 1,
        "producer_head": subprocess.check_output(
            ["git", "-C", str(main_figures.PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "git_index_sha256": hashlib.sha256(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(main_figures.PROJECT_ROOT),
                    "ls-files",
                    "--stage",
                ]
            )
        ).hexdigest(),
        "original_audit_sha256": file_hash(root / AUDIT_PATH),
        "presentation_baseline_sha256": file_hash(root / BASELINE_PATH),
        "original_main_sha256": baseline["original_main_sha256"],
        "canonical_main_sha256_before_promotion": before,
        "scientific_inputs_sha256": {
            p: h
            for p, h in audit["inputs_sha256"].items()
            if p.startswith("data/") and p != str(MAIN_PATH)
        },
        "validation": {
            "invariance_passed": True,
            "A_B_specs_exact": True,
            "A_B_era_log_helper_ASTs_exact": True,
            "A_B_compiled_vega_exact": True,
            "point_and_interval_rows_match_prior_delivery": len(data),
            "interval_comparison_absolute_tolerance": 1e-10,
            "main_point_rows_match_companion": len(data),
        },
        "outputs_sha256": {str(p): file_hash(c) for p, c in outputs.items()},
        "source_sha256": {
            str(Path(p).relative_to(main_figures.PROJECT_ROOT)): file_hash(
                Path(p)
            )
            for p in (__file__, main_figures.__file__, accepted.__file__)
        },
        "runtime": {
            "python": accepted.platform.python_version(),
            "scipy": accepted.scipy.__version__,
            "altair": alt.__version__,
            "pandas": pd.__version__,
            "vl_convert": vl_convert.__version__,
        },
        "methods_and_limits": json.loads(
            (
                root / TABLE_DIR / "cs5_completeness_ci_provenance.json"
            ).read_text()
        )["methods"],
        "labels": list(SELECTION),
        "palette": main_figures.get_completeness_field_colors(),
        "presentation": "A/B unchanged; main C local height400 without CIs; one five-facet CI companion. 95% CI is the extracted schema key, not a validated extracted interval level.",
        "original_audit_unchanged": True,
    }
    # Copy only after all candidates and invariance checks have succeeded.
    for relative, candidate in outputs.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, destination)
        if (
            file_hash(destination)
            != provenance["outputs_sha256"][str(relative)]
        ):
            raise RuntimeError(f"Published output hash differs: {relative}")
    (root / PROVENANCE_PATH).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    for relative in [*outputs, PROVENANCE_PATH]:
        print(root / relative)


def main() -> None:
    """Validate original scientific baseline, generate and publish successors."""
    args = parse_args()
    baseline, audit = validate_baseline(PROJECT_ROOT)
    config = main_figures.load_config(PROJECT_ROOT / CONFIG_PATH)
    cs5_dir = PROJECT_ROOT / config["output"]["case_study_5"]["base"]
    temporal = main_figures.load_temporal_stats(PROJECT_ROOT / "data")
    diversity = main_figures.load_diversity_data(cs5_dir)
    counts = main_figures.load_completeness_data(cs5_dir)
    eras = main_figures.get_era_transitions(
        config["case_study_5"]["temporal_eras"]
    )
    data = verify_invariance(
        PROJECT_ROOT, baseline, counts, temporal, diversity, eras
    )
    if args.dry_run:
        print(
            f"Validated A/B invariance and {len(data)} point/interval rows; no writes"
        )
        return
    if args.candidate_dir:
        if (
            args.candidate_dir.resolve()
            == (PROJECT_ROOT / FIGURE_DIR).resolve()
        ):
            raise ValueError(
                "Candidate directory must not be canonical figure directory"
            )
        publish_candidates(
            PROJECT_ROOT,
            args.candidate_dir,
            data,
            temporal,
            diversity,
            counts,
            config,
            audit,
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="cs5-consolidated-candidates-"
        ) as tmp:
            publish_candidates(
                PROJECT_ROOT,
                Path(tmp),
                data,
                temporal,
                diversity,
                counts,
                config,
                audit,
            )


if __name__ == "__main__":
    main()
