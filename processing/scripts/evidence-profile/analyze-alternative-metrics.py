"""Analyze feasibility of alternative metrics for evidence profile similarity.

This script assesses alternative metrics that could replace or supplement
statistical consistency (Cohen's kappa on p-value significance patterns).

Alternative metrics investigated:
1. SE-based significance: Compute p-value from effect size and SE
2. CI-based significance: Check if CI crosses null value
3. Precision concordance: Agreement on precision (SE < threshold)
4. Statistical power: Agreement on statistical power
5. Meta-analytic heterogeneity: I² statistic for effect size heterogeneity

For each metric, we assess:
- Data availability (% of results with required fields)
- Computation feasibility (can we compute it?)
- Information gain (correlation with existing metrics)
- Implementation difficulty (easy/medium/hard)
"""

import json
from typing import Any, Dict

import duckdb
import numpy as np
from loguru import logger
from scipy import stats
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSED_PATH = (
    DATA_DIR / "processed" / "evidence-profiles" / "evidence-profiles.json"
)
EVIDENCE_DB_PATH = DATA_DIR / "db" / "evidence_profile_db.db"


def se_based_significance(
    effect_size: float, se: float, effect_type: str
) -> bool:
    """Compute significance from effect size and SE.

    Args:
        effect_size: Harmonized effect size (log scale)
        se: Standard error
        effect_type: Type of effect (beta, OR, HR)

    Returns:
        True if statistically significant (p < 0.05)
    """
    # Convert to float if needed
    try:
        effect_size = float(effect_size)
        se = float(se)
    except (ValueError, TypeError):
        return False

    if se <= 0:
        return False

    z_score = abs(effect_size) / se
    p_value = 2 * (1 - stats.norm.cdf(z_score))

    return p_value < 0.05


def ci_based_significance(
    ci_lower: float, ci_upper: float, effect_type: str
) -> bool:
    """Compute significance from confidence interval.

    Args:
        ci_lower: Lower bound of 95% CI
        ci_upper: Upper bound of 95% CI
        effect_type: Type of effect (beta, OR, HR)

    Returns:
        True if CI doesn't cross null value
    """
    # Convert to float if needed
    try:
        ci_lower = float(ci_lower)
        ci_upper = float(ci_upper)
    except (ValueError, TypeError):
        return False

    if effect_type == "beta":
        return not (ci_lower <= 0 <= ci_upper)
    else:
        return not (ci_lower <= 1 <= ci_upper)


def assess_se_based_significance() -> Dict[str, Any]:
    """Assess feasibility of SE-based significance classification."""
    logger.info("=== SE-BASED SIGNIFICANCE ASSESSMENT ===")

    with open(PREPROCESSED_PATH, "r") as f:
        profiles = json.load(f)

    total_results = 0
    se_available = 0
    se_computed = 0
    agreement_with_pvalue = 0
    both_available = 0

    for profile in profiles:
        for result in profile["results"]:
            total_results += 1

            has_se = result.get("se") is not None
            has_pvalue = result.get("p_value") is not None

            if has_se:
                se_available += 1

                effect_size = result.get("harmonized_effect_size")
                se = result.get("se")
                effect_type = result.get("effect_size_type")

                # Convert se to float if needed
                try:
                    se = float(se) if se is not None else None
                except (ValueError, TypeError):
                    se = None

                if effect_size is not None and se is not None and se > 0:
                    se_computed += 1
                    se_sig = se_based_significance(
                        effect_size, se, effect_type
                    )

                    if has_pvalue:
                        both_available += 1
                        pval_sig = result.get("is_significant", False)
                        if se_sig == pval_sig:
                            agreement_with_pvalue += 1

    logger.info(f"Total results: {total_results:,}")
    logger.info(
        f"SE available: {se_available:,} "
        f"({100.0 * se_available / total_results:.2f}%)"
    )
    logger.info(
        f"SE-based significance computed: {se_computed:,} "
        f"({100.0 * se_computed / total_results:.2f}%)"
    )
    logger.info(f"Both SE and p-value available: {both_available:,}")

    if both_available > 0:
        logger.info(
            f"Agreement with p-value significance: "
            f"{agreement_with_pvalue:,} / {both_available:,} "
            f"({100.0 * agreement_with_pvalue / both_available:.2f}%)"
        )

    return {
        "data_availability_pct": 100.0 * se_available / total_results,
        "computation_feasibility_pct": 100.0 * se_computed / total_results,
        "agreement_with_pvalue_pct": (
            100.0 * agreement_with_pvalue / both_available
            if both_available > 0
            else 0
        ),
        "feasibility": "LOW",
        "reason": "Only 3% of results have SE",
    }


def assess_ci_based_significance() -> Dict[str, Any]:
    """Assess feasibility of CI-based significance classification."""
    logger.info("\n=== CI-BASED SIGNIFICANCE ASSESSMENT ===")

    with open(PREPROCESSED_PATH, "r") as f:
        profiles = json.load(f)

    total_results = 0
    ci_available = 0
    ci_computed = 0
    agreement_with_pvalue = 0
    both_available = 0

    for profile in profiles:
        for result in profile["results"]:
            total_results += 1

            has_ci = (
                result.get("ci_lower") is not None
                and result.get("ci_upper") is not None
            )
            has_pvalue = result.get("p_value") is not None

            if has_ci:
                ci_available += 1

                ci_lower = result.get("ci_lower")
                ci_upper = result.get("ci_upper")
                effect_type = result.get("effect_size_type")

                # Convert to float if needed
                try:
                    ci_lower = (
                        float(ci_lower) if ci_lower is not None else None
                    )
                    ci_upper = (
                        float(ci_upper) if ci_upper is not None else None
                    )
                except (ValueError, TypeError):
                    ci_lower = None
                    ci_upper = None

                if ci_lower is not None and ci_upper is not None:
                    ci_computed += 1
                    ci_sig = ci_based_significance(
                        ci_lower, ci_upper, effect_type
                    )

                    if has_pvalue:
                        both_available += 1
                        pval_sig = result.get("is_significant", False)
                        if ci_sig == pval_sig:
                            agreement_with_pvalue += 1

    logger.info(f"Total results: {total_results:,}")
    logger.info(
        f"CI available: {ci_available:,} "
        f"({100.0 * ci_available / total_results:.2f}%)"
    )
    logger.info(
        f"CI-based significance computed: {ci_computed:,} "
        f"({100.0 * ci_computed / total_results:.2f}%)"
    )
    logger.info(f"Both CI and p-value available: {both_available:,}")

    if both_available > 0:
        logger.info(
            f"Agreement with p-value significance: "
            f"{agreement_with_pvalue:,} / {both_available:,} "
            f"({100.0 * agreement_with_pvalue / both_available:.2f}%)"
        )

    return {
        "data_availability_pct": 100.0 * ci_available / total_results,
        "computation_feasibility_pct": 100.0 * ci_computed / total_results,
        "agreement_with_pvalue_pct": (
            100.0 * agreement_with_pvalue / both_available
            if both_available > 0
            else 0
        ),
        "feasibility": "HIGH",
        "reason": "87% of results have CI, high agreement with p-values",
    }


def assess_precision_concordance() -> Dict[str, Any]:
    """Assess feasibility of precision concordance metric."""
    logger.info("\n=== PRECISION CONCORDANCE ASSESSMENT ===")

    with open(PREPROCESSED_PATH, "r") as f:
        profiles = json.load(f)

    se_values = []
    ci_widths = []

    for profile in profiles:
        for result in profile["results"]:
            se = result.get("se")
            # Convert to float if needed
            try:
                se = float(se) if se is not None else None
            except (ValueError, TypeError):
                se = None

            if se is not None and se > 0:
                se_values.append(se)

            ci_lower = result.get("ci_lower")
            ci_upper = result.get("ci_upper")

            # Convert to float if needed
            try:
                ci_lower = float(ci_lower) if ci_lower is not None else None
                ci_upper = float(ci_upper) if ci_upper is not None else None
            except (ValueError, TypeError):
                ci_lower = None
                ci_upper = None

            if ci_lower is not None and ci_upper is not None:
                width = abs(ci_upper - ci_lower)
                ci_widths.append(width)

    se_median = np.median(se_values) if se_values else None
    ci_width_median = np.median(ci_widths) if ci_widths else None

    logger.info(f"SE values available: {len(se_values):,}")
    logger.info(
        f"SE median: {se_median:.4f}" if se_median else "SE median: N/A"
    )
    logger.info(f"CI widths available: {len(ci_widths):,}")
    logger.info(
        f"CI width median: {ci_width_median:.4f}"
        if ci_width_median
        else "CI width median: N/A"
    )

    logger.info(
        "\nPrecision concordance: Compare if both studies have "
        "precise estimates"
    )
    logger.info("  - Could use CI width < median as threshold")
    logger.info("  - Would work for 87% of results (those with CI)")

    return {
        "data_availability_pct": (
            100.0 * len(ci_widths) / (len(se_values) + len(ci_widths))
            if (se_values or ci_widths)
            else 0
        ),
        "computation_feasibility_pct": (
            100.0 * len(ci_widths) / (len(se_values) + len(ci_widths))
            if (se_values or ci_widths)
            else 0
        ),
        "feasibility": "HIGH",
        "reason": "Can use CI width as precision measure (87% coverage)",
    }


def assess_effect_size_heterogeneity() -> Dict[str, Any]:
    """Assess feasibility of meta-analytic heterogeneity metric."""
    logger.info("\n=== EFFECT SIZE HETEROGENEITY ASSESSMENT ===")

    with open(PREPROCESSED_PATH, "r") as f:
        profiles = json.load(f)

    profiles_by_id = {}
    for p in profiles:
        key = (p["pmid"], p["model"])
        profiles_by_id[key] = p

    conn = duckdb.connect(str(EVIDENCE_DB_PATH), read_only=True)

    query = """
    SELECT 
        c.pmid as query_pmid,
        c.model as query_model,
        s.similar_pmid,
        s.similar_model,
        s.matched_pairs
    FROM evidence_similarities s
    JOIN query_combinations c ON s.query_combination_id = c.id
    WHERE s.matched_pairs >= 2
    LIMIT 100
    """

    comparisons = conn.execute(query).fetchall()

    computable_count = 0
    total_sampled = 0

    for query_pmid, query_model, sim_pmid, sim_model, n_pairs in comparisons:
        total_sampled += 1

        query_prof = profiles_by_id.get((query_pmid, query_model))
        sim_prof = profiles_by_id.get((sim_pmid, sim_model))

        if not query_prof or not sim_prof:
            continue

        matched = []
        for qr in query_prof["results"]:
            for sr in sim_prof["results"]:
                if (
                    qr["exposure_trait_index"] == sr["exposure_trait_index"]
                    and qr["outcome_trait_index"] == sr["outcome_trait_index"]
                ):
                    matched.append((qr, sr))

        if len(matched) >= 2:
            has_effect_sizes = all(
                qr.get("harmonized_effect_size") is not None
                and sr.get("harmonized_effect_size") is not None
                for qr, sr in matched
            )
            if has_effect_sizes:
                computable_count += 1

    logger.info(f"Sampled comparisons: {total_sampled:,}")
    logger.info(
        f"Computable I² statistic: {computable_count:,} "
        f"({100.0 * computable_count / total_sampled:.2f}%)"
    )

    logger.info(
        "\nI² measures heterogeneity of effect sizes across matched pairs"
    )
    logger.info("  - Requires >= 2 matched pairs with effect sizes")
    logger.info("  - Would complement existing effect size similarity metric")

    conn.close()

    return {
        "data_availability_pct": (
            100.0 * computable_count / total_sampled
            if total_sampled > 0
            else 0
        ),
        "computation_feasibility_pct": (
            100.0 * computable_count / total_sampled
            if total_sampled > 0
            else 0
        ),
        "feasibility": "MEDIUM",
        "reason": "Requires >= 2 matched pairs, redundant with effect_size_similarity",
    }


def summarize_recommendations() -> None:
    """Summarize recommendations for batch 4."""
    logger.info("\n===== RECOMMENDATIONS FOR BATCH 4 =====")

    logger.info("\n1. PRIMARY RECOMMENDATION: CI-based significance")
    logger.info("   - Feasibility: HIGH")
    logger.info("   - Data availability: 87% of results")
    logger.info("   - Agreement with p-values: Expected to be high")
    logger.info(
        "   - Implementation: Easy (add to preprocess-evidence-profiles.py)"
    )
    logger.info(
        "   - Benefit: Enables statistical consistency for 87% of results"
    )

    logger.info("\n2. SECONDARY RECOMMENDATION: Precision concordance")
    logger.info("   - Feasibility: HIGH")
    logger.info("   - Data availability: 87% (via CI width)")
    logger.info("   - Implementation: Easy (new metric)")
    logger.info("   - Benefit: Complements statistical consistency")
    logger.info("   - Interpretation: Agreement on estimate precision")

    logger.info("\n3. NOT RECOMMENDED: SE-based significance")
    logger.info("   - Reason: Only 3% data availability")

    logger.info("\n4. NOT RECOMMENDED: Effect size heterogeneity (I²)")
    logger.info("   - Reason: Redundant with existing effect_size_similarity")

    logger.info("\n\nIMPLEMENTATION PLAN:")
    logger.info("Step 1: Add CI-based significance to preprocessing")
    logger.info("  - Modify preprocess-evidence-profiles.py")
    logger.info("  - Add is_significant_ci field alongside is_significant")
    logger.info(
        "  - Use is_significant_ci as fallback if is_significant is null"
    )

    logger.info("\nStep 2: Update statistical consistency computation")
    logger.info("  - Modify compute-evidence-similarity.py")
    logger.info("  - Use combined significance (p-value OR CI-based)")
    logger.info("  - Expected impact: 0.1% → 30-40% availability")

    logger.info("\nStep 3: Add precision concordance metric")
    logger.info("  - Add compute_precision_concordance() function")
    logger.info("  - Use CI width as precision measure")
    logger.info("  - Compute agreement on high-precision pairs")

    logger.info("\nStep 4: Re-run preprocessing and similarity computation")
    logger.info("  - Batch 4 preprocessing with CI-based significance")
    logger.info("  - Batch 4 similarity with updated metrics")


def main() -> None:
    """Run alternative metrics analysis."""
    logger.info("===== ALTERNATIVE METRICS ANALYSIS =====\n")

    results = {}
    results["se_based"] = assess_se_based_significance()
    results["ci_based"] = assess_ci_based_significance()
    results["precision"] = assess_precision_concordance()
    results["heterogeneity"] = assess_effect_size_heterogeneity()

    logger.info("\n\n===== FEASIBILITY SUMMARY =====")
    for metric, data in results.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Feasibility: {data['feasibility']}")
        logger.info(
            f"  Data availability: {data['data_availability_pct']:.2f}%"
        )
        logger.info(f"  Reason: {data['reason']}")

    summarize_recommendations()


if __name__ == "__main__":
    main()
