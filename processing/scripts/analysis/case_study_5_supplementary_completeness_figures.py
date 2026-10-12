"""Generate audited supplementary companions to Figure 4C and 4D.

Annual observations are connected without a fitted trend model.
Ordinary fields use pointwise 95% Wilson intervals.
Effect Size retains max(beta, OR, HR), with a conservative envelope of
Bonferroni-adjusted exact component intervals, not an any-field union.
The original combined figure is never written by this entry point.
"""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import altair as alt
import pandas as pd
import scipy
import vl_convert
from scipy.stats import beta, norm

from case_study_5_manuscript_figures import PROJECT_ROOT, get_era_transitions
from color_schemes import get_field_type_colors

# ==== Input and output contracts ====

COUNTS_PATH = Path(
    "data/processed/case-study-cs5/completeness/field_completeness_by_year.csv"
)
CONFIG_PATH = Path("processing/config/case_studies.yml")
MAIN_PATH = Path(
    "data/artifacts/manuscript-figures/cs5_fig_temporal_trends.png"
)
TABLE_DIR = PROJECT_ROOT / "data/artifacts/manuscript-tables"
FIGURE_DIR = PROJECT_ROOT / "data/artifacts/manuscript-figures"
AUDIT_PATH = TABLE_DIR / "cs5_completeness_baseline_audit.json"
FIELDS = (
    "effect_size_beta",
    "effect_size_or",
    "effect_size_hr",
    "confidence_interval",
    "standard_error",
    "p_value",
    "direction",
)
METRICS = {
    "C": {
        "P-value": "p_value",
        "Confidence interval": "confidence_interval",
        "Standard error": "standard_error",
    },
    "D": {
        "Confidence Interval": "confidence_interval",
        "Direction": "direction",
        "Effect Size": "effect_size",
        "Statistical": "p_value",
    },
}
FILENAMES = {
    "C": "cs5_fig_completeness_by_field_ci.png",
    "D": "cs5_fig_completeness_by_field_type_ci.png",
}


# ==== Interval calculations ====


def validate_count(k: int, n: int) -> None:
    """Reject noninteger counts, invalid denominators and impossible counts."""
    if (
        isinstance(k, bool)
        or isinstance(n, bool)
        or not math.isfinite(k)
        or not math.isfinite(n)
        or int(k) != k
        or int(n) != n
        or n <= 0
        or not 0 <= k <= n
    ):
        raise ValueError("Counts require integer 0 <= k <= n and n > 0")


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson score interval as proportions."""
    validate_count(k, n)
    z = norm.ppf(0.975)
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    radius /= denominator
    res = (max(0.0, center - radius), min(1.0, center + radius))
    return res


def exact_interval(
    k: int,
    n: int,
    alpha: float = 0.05 / 3,
) -> tuple[float, float]:
    """Return a Clopper-Pearson interval with explicit boundary handling."""
    validate_count(k, n)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    res = (lower, upper)
    return res


def max_effect_interval(
    counts: tuple[int, int, int],
    n: int,
) -> tuple[float, float]:
    """Propagate three simultaneously covered exact intervals through max."""
    if len(counts) != 3:
        raise ValueError("Effect Size requires exactly beta, OR and HR counts")
    intervals = [exact_interval(k, n) for k in counts]
    res = (max(x[0] for x in intervals), max(x[1] for x in intervals))
    return res


# ==== Validation and long-form data ====


def prepare_interval_data(counts: pd.DataFrame) -> pd.DataFrame:
    """Preserve annual percentages; attach counts and pointwise bounds.

    Zero-denominator years are absent observations, never zero percentages.
    Missing years are not imputed; lines connect successive available years.
    """
    required = {"pub_year", "n_studies"}
    required.update(f"{f}_{suffix}" for f in FIELDS for suffix in ("n", "pct"))
    if not required.issubset(counts.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(counts))}")
    if counts.empty or counts["pub_year"].duplicated().any():
        raise ValueError("Require nonempty counts and unique annual rows")
    rows = []
    for _, row in counts.sort_values("pub_year").iterrows():
        year, n = row["pub_year"], row["n_studies"]
        if (
            not math.isfinite(year)
            or int(year) != year
            or not 2003 <= year <= 2025
        ):
            raise ValueError("Publication years must be integers in 2003-2025")
        if not math.isfinite(n) or int(n) != n or n < 0:
            raise ValueError(
                "Annual denominators must be nonnegative integers"
            )
        for field in FIELDS:
            k = row[f"{field}_n"]
            validate_count(k, max(n, 1))
            if k > n:
                raise ValueError("Reported count exceeds annual denominator")
            pct = row[f"{field}_pct"]
            if n and (
                not math.isfinite(pct) or abs(pct - 100 * k / n) > 1e-10
            ):
                raise ValueError("Stored percentage does not match count/n")
        if n == 0:
            continue
        n = int(n)
        component_counts = tuple(int(row[f"{f}_n"]) for f in FIELDS[:3])
        for companion, mappings in METRICS.items():
            for metric, field in mappings.items():
                if field == "effect_size":
                    k = max(component_counts)
                    pct = max(row[f"{f}_pct"] for f in FIELDS[:3])
                    lower, upper = max_effect_interval(component_counts, n)
                    method = "bonferroni_clopper_pearson_max_envelope"
                else:
                    k = int(row[f"{field}_n"])
                    pct = row[f"{field}_pct"]
                    lower, upper = wilson_interval(k, n)
                    method = "wilson"
                rows.append(
                    {
                        "companion": companion,
                        "metric": metric,
                        "pub_year": int(year),
                        "n_studies": n,
                        "n_reported": k,
                        "completeness_pct": pct,
                        "lower_pct": 100 * lower,
                        "upper_pct": 100 * upper,
                        "confidence_level": 0.95,
                        "interval_method": method,
                        **{f"{f}_n": int(row[f"{f}_n"]) for f in FIELDS},
                    }
                )
    if not rows:
        raise ValueError("No positive-denominator annual observations")
    res = pd.DataFrame(rows)
    return res


def file_hash(path: Path) -> str:
    """Calculate a file digest without exposing its contents."""
    res = hashlib.sha256(path.read_bytes()).hexdigest()
    return res


def verify_audit(audit: dict, root: Path = PROJECT_ROOT) -> None:
    """Require the frozen audit and unchanged inputs/main PNG before writing."""
    checks = audit.get("checks", {})
    if audit.get("status") != "passed" or not all(
        checks.get(key) is True
        for key in (
            "annual_and_era_counts_match_frozen_predicate",
            "retained_png_reproduced_byte_identically",
            "installed_manuscript_png_matches",
        )
    ):
        raise ValueError("Require a passed baseline audit before generation")
    for path in (COUNTS_PATH, CONFIG_PATH, MAIN_PATH):
        expected = audit.get("inputs_sha256", {}).get(str(path))
        if not expected or file_hash(root / path) != expected:
            raise ValueError(
                f"Audited input changed: {path}; reconcile baseline"
            )


# ==== Faceted figures ====


def create_companion(
    data: pd.DataFrame,
    companion: str,
    eras: list[dict],
) -> alt.FacetChart:
    """Build readable C (three rows) or D (2x2) annual trajectories."""
    frame = data[data["companion"] == companion].copy()
    colors = get_field_type_colors(use_gruvbox=True)
    palette = [
        colors[
            {"Standard error": "SE", "Confidence interval": "95% CI"}.get(m, m)
        ]
        for m in METRICS[companion]
    ]
    # Layers must inherit facet-filtered data, not bind the full frame again.
    base = alt.Chart().encode(
        x=alt.X(
            "pub_year:Q",
            title="Publication year",
            scale=alt.Scale(domain=[2003, 2025], nice=False),
            axis=alt.Axis(
                values=(
                    [2003, 2008, 2013, 2018, 2023, 2025]
                    if companion == "C"
                    else [2003, 2008, 2013, 2018, 2025]
                ),
                format="d",
            ),
        ),
        color=alt.Color(
            "metric:N",
            scale=alt.Scale(domain=list(METRICS[companion]), range=palette),
            legend=None,
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
    # Era markers are context, not interventions or tests of causal impact.
    rules = (
        alt.Chart(pd.DataFrame(eras))
        .mark_rule(
            color="#888888",
            strokeDash=[3, 3],
            opacity=0.6,
        )
        .encode(x="year:Q")
    )
    layered = alt.layer(band, rules, line, data=frame).properties(
        width=560 if companion == "C" else 260,
        height=140 if companion == "C" else 185,
    )
    sparse = frame.drop_duplicates("pub_year")
    sparse = sparse[sparse["n_studies"] < 20]
    n_note = "; ".join(
        f"{int(r.pub_year)}: n={int(r.n_studies)}" for r in sparse.itertuples()
    )
    subtitle = [
        "Annual observations and pointwise 95% intervals (no fitted smoother)."
    ]
    if n_note:
        subtitle.append(n_note)
    if companion == "D":
        subtitle.append(
            "Effect Size: max(beta, OR, HR); conservative exact max envelope."
        )
    res = (
        layered.facet(
            facet=alt.Facet(
                "metric:N",
                sort=list(METRICS[companion]),
                header=alt.Header(
                    title=None, labelFontSize=15, labelFontWeight="bold"
                ),
            ),
            columns=1 if companion == "C" else 2,
            spacing=20,
        )
        .properties(
            title=alt.TitleParams(
                text=f"Reporting completeness: Figure 4{companion} companion",
                subtitle=subtitle,
                anchor="start",
                fontSize=17,
                subtitleFontSize=12,
            )
        )
        .configure_axis(
            labelFontSize=13,
            titleFontSize=14,
            titlePadding=10,
            labelAngle=0,
        )
        .configure_view(stroke=None)
    )
    return res


# ==== Command-line entry point ====


def parse_args() -> argparse.Namespace:
    """Parse supplementary-only output paths and no-write preview mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    # ---- --dry-run ----
    parser.add_argument("-n", "--dry-run", action="store_true")
    # ---- --output-dir ----
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    # ---- --table-dir ----
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    # ---- --audit-json ----
    parser.add_argument("--audit-json", type=Path, default=AUDIT_PATH)
    res = parser.parse_args()
    return res


def main() -> None:
    """Validate the frozen baseline and write only supplementary artifacts."""
    import yaml

    args = parse_args()
    audit = json.loads(args.audit_json.read_text())
    verify_audit(audit, PROJECT_ROOT)
    data = prepare_interval_data(pd.read_csv(PROJECT_ROOT / COUNTS_PATH))
    config = yaml.safe_load((PROJECT_ROOT / CONFIG_PATH).read_text())
    eras = get_era_transitions(config["case_study_5"]["temporal_eras"])
    if args.dry_run:
        print(
            f"Validated {len(data)} interval rows; would write two companions"
        )
        return
    main_hash = file_hash(PROJECT_ROOT / MAIN_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for companion, filename in FILENAMES.items():
        path = args.output_dir / filename
        create_companion(data, companion, eras).save(str(path), ppi=300)
        outputs.append(path)
    csv = args.table_dir / "cs5_completeness_annual_ci.csv"
    data.to_csv(csv, index=False)
    outputs.append(csv)
    if file_hash(PROJECT_ROOT / MAIN_PATH) != main_hash:
        raise RuntimeError("Original main figure changed during generation")
    provenance = {
        "schema_version": 1,
        "baseline_audit_sha256": file_hash(args.audit_json),
        "baseline_audit": audit,
        "generator_sha256": file_hash(Path(__file__)),
        "source_counts_sha256": file_hash(PROJECT_ROOT / COUNTS_PATH),
        "main_png_sha256_before_and_after": main_hash,
        "runtime": {
            "python": platform.python_version(),
            "altair": alt.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "vl_convert": vl_convert.__version__,
        },
        "outputs_sha256": {
            str(p.relative_to(PROJECT_ROOT))
            if p.is_relative_to(PROJECT_ROOT)
            else str(p): file_hash(p)
            for p in outputs
        },
        "methods": {
            "ordinary_fields": "Two-sided pointwise 95% Wilson score intervals",
            "effect_size": "max of three marginal proportions; max of component lower/upper CP bounds, each alpha=0.05/3, two-sided tails=0.05/6",
            "n_reported": "For Effect Size this is max of component counts, NOT an any-effect-size event count",
            "trajectory": "Lines connect available annual observations; no imputation, smoother or trend test",
            "coverage": "Pointwise per year; not simultaneous across years or plotted metrics",
            "assumptions": "Binomial reporting process conditional on included/extracted corpus and independent publication rows",
            "limits": "Not extraction-accuracy, representativeness, full-paper reporting or causal STROBE uncertainty; collection cutoff is unverified",
        },
    }
    path = args.table_dir / "cs5_completeness_ci_provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    for path in [*outputs, path]:
        print(path)


if __name__ == "__main__":
    main()
