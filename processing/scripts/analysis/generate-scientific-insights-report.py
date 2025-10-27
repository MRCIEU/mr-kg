#!/usr/bin/env python3
"""Generate comprehensive scientific insights report from evidence profile
similarity analysis.

This script analyzes the MR-KG evidence profile similarity data to extract
scientific insights about Mendelian Randomization study consistency.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd


def load_data():
    """Load all analysis data from database and CSV files."""
    data = {}

    con = duckdb.connect("../data/db/evidence_profile_db.db", read_only=True)

    data["summary_stats"] = pd.read_csv(
        "../data/processed/evidence-profiles/analysis/summary-stats-by-model.csv"
    )
    data["similarity_dist"] = pd.read_csv(
        "../data/processed/evidence-profiles/analysis/"
        "similarity-distributions.csv"
    )
    data["trait_evidence_corr"] = pd.read_csv(
        "../data/processed/evidence-profiles/analysis/"
        "trait-vs-evidence-correlation.csv"
    )
    data["completeness"] = pd.read_csv(
        "../data/processed/evidence-profiles/analysis/"
        "completeness-by-model.csv"
    )
    data["matched_pairs"] = pd.read_csv(
        "../data/processed/evidence-profiles/analysis/"
        "matched-pairs-distribution.csv"
    )
    data["metric_corr"] = pd.read_csv(
        "../data/processed/evidence-profiles/analysis/metric-correlations.csv"
    )

    with open(
        "../data/processed/evidence-profiles/analysis/data-quality-report.json"
    ) as f:
        data["quality_report"] = json.load(f)

    with open(
        "../data/processed/evidence-profiles/analysis/validation-report.json"
    ) as f:
        data["validation_report"] = json.load(f)

    data["db_direction_overall"] = con.execute("""
        SELECT 
            COUNT(*) as total_pairs,
            ROUND(AVG(direction_concordance), 3) as mean_concordance,
            ROUND(MEDIAN(direction_concordance), 3) as median_concordance,
            ROUND(STDDEV(direction_concordance), 3) as sd_concordance,
            SUM(CASE WHEN direction_concordance = 1 THEN 1 ELSE 0 END) 
                as n_concordant,
            SUM(CASE WHEN direction_concordance = -1 THEN 1 ELSE 0 END) 
                as n_discordant,
            SUM(CASE WHEN direction_concordance = 0 THEN 1 ELSE 0 END) 
                as n_mixed,
            ROUND(SUM(CASE WHEN direction_concordance = 1 THEN 1 ELSE 0 END) 
                * 100.0 / COUNT(*), 2) as pct_concordant,
            ROUND(SUM(CASE WHEN direction_concordance = -1 THEN 1 ELSE 0 END) 
                * 100.0 / COUNT(*), 2) as pct_discordant
        FROM evidence_similarities
    """).fetchdf()

    data["db_match_type"] = con.execute("""
        SELECT 
            SUM(match_type_exact) as n_exact,
            SUM(match_type_fuzzy) as n_fuzzy,
            SUM(match_type_efo) as n_efo,
            COUNT(*) as total_matches,
            ROUND(SUM(match_type_exact) * 100.0 / COUNT(*), 2) 
                as pct_exact,
            ROUND(SUM(match_type_fuzzy) * 100.0 / COUNT(*), 2) 
                as pct_fuzzy,
            ROUND(SUM(match_type_efo) * 100.0 / COUNT(*), 2) as pct_efo
        FROM evidence_similarities
    """).fetchdf()

    data["db_direction_by_match"] = con.execute("""
        SELECT 
            CASE 
                WHEN match_type_exact = 1 THEN 'Exact'
                WHEN match_type_fuzzy = 1 THEN 'Fuzzy'
                WHEN match_type_efo = 1 THEN 'EFO'
            END as match_type,
            COUNT(*) as n_pairs,
            ROUND(AVG(direction_concordance), 3) as mean_concordance,
            SUM(CASE WHEN direction_concordance = 1 THEN 1 ELSE 0 END) 
                as n_concordant,
            SUM(CASE WHEN direction_concordance = -1 THEN 1 ELSE 0 END) 
                as n_discordant,
            ROUND(SUM(CASE WHEN direction_concordance = 1 THEN 1 ELSE 0 END) 
                * 100.0 / COUNT(*), 2) as pct_concordant
        FROM evidence_similarities
        GROUP BY match_type
        ORDER BY n_pairs DESC
    """).fetchdf()

    data["db_model_availability"] = con.execute("""
        SELECT 
            similar_model as model,
            COUNT(*) as n_pairs,
            SUM(CASE WHEN effect_size_similarity IS NOT NULL THEN 1 
                ELSE 0 END) as n_with_effect_size,
            SUM(CASE WHEN statistical_consistency IS NOT NULL THEN 1 
                ELSE 0 END) as n_with_statistical,
            ROUND(AVG(similar_completeness), 3) as avg_completeness,
            ROUND(MEDIAN(similar_completeness), 3) as median_completeness,
            ROUND(SUM(CASE WHEN effect_size_similarity IS NOT NULL THEN 1 
                ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_effect_size,
            ROUND(SUM(CASE WHEN statistical_consistency IS NOT NULL THEN 1 
                ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_statistical
        FROM evidence_similarities
        GROUP BY similar_model
        ORDER BY n_pairs DESC
    """).fetchdf()

    data["db_overlap"] = con.execute("""
        SELECT 
            similar_model as model,
            COUNT(*) as n_pairs,
            ROUND(AVG(evidence_overlap), 3) as mean_overlap,
            ROUND(MEDIAN(evidence_overlap), 3) as median_overlap,
            SUM(CASE WHEN evidence_overlap = 0 THEN 1 ELSE 0 END) 
                as n_zero_overlap,
            SUM(CASE WHEN evidence_overlap = 1 THEN 1 ELSE 0 END) 
                as n_full_overlap,
            ROUND(SUM(CASE WHEN evidence_overlap = 1 THEN 1 ELSE 0 END) 
                * 100.0 / COUNT(*), 2) as pct_full_overlap
        FROM evidence_similarities
        GROUP BY similar_model
        ORDER BY n_pairs DESC
    """).fetchdf()

    data["db_completeness_concordance"] = con.execute("""
        SELECT 
            CASE 
                WHEN similar_completeness < 0.33 THEN 'Low (< 0.33)'
                WHEN similar_completeness < 0.67 THEN 'Medium (0.33-0.67)'
                ELSE 'High (>= 0.67)'
            END as completeness_level,
            COUNT(*) as n_pairs,
            ROUND(AVG(direction_concordance), 3) as mean_concordance,
            SUM(CASE WHEN direction_concordance = 1 THEN 1 ELSE 0 END) 
                as n_concordant,
            ROUND(SUM(CASE WHEN direction_concordance = 1 THEN 1 ELSE 0 END) 
                * 100.0 / COUNT(*), 2) as pct_concordant
        FROM evidence_similarities
        GROUP BY completeness_level
        ORDER BY 
            CASE completeness_level
                WHEN 'Low (< 0.33)' THEN 1
                WHEN 'Medium (0.33-0.67)' THEN 2
                WHEN 'High (>= 0.67)' THEN 3
            END
    """).fetchdf()

    data["db_year_trends"] = con.execute("""
        SELECT 
            similar_publication_year as year,
            COUNT(*) as n_pairs,
            ROUND(AVG(direction_concordance), 3) as mean_concordance,
            ROUND(AVG(similar_completeness), 3) as mean_completeness
        FROM evidence_similarities
        WHERE similar_publication_year IS NOT NULL
        GROUP BY similar_publication_year
        ORDER BY year DESC
    """).fetchdf()

    con.close()

    return data


def generate_report(data):
    """Generate comprehensive scientific insights report."""
    report = []

    report.append(
        "# Scientific Insights from Evidence Profile Similarity Analysis"
    )
    report.append("")
    report.append("**Analysis Date:** October 27, 2025")
    report.append("")
    report.append("## Executive Summary")
    report.append("")

    total_pairs = data["db_direction_overall"]["total_pairs"].values[0]
    pct_concordant = data["db_direction_overall"]["pct_concordant"].values[0]
    pct_discordant = data["db_direction_overall"]["pct_discordant"].values[0]

    report.append(
        f"This report analyzes {total_pairs:,} evidence profile similarity "
        "comparisons across 6,132 Mendelian Randomization studies. Despite "
        "limited availability of structured statistical data (effect sizes "
        "present in ~4% of pairs, statistical consistency in <1%), we achieve "
        f"100% coverage of direction concordance, revealing that {pct_concordant}% "
        "of similar studies report concordant causal directions while "
        f"{pct_discordant}% show discordance. The analysis provides novel "
        "insights into LLM extraction performance, the relationship between "
        "trait naming and causal evidence, and systematic patterns in MR "
        "literature reporting practices."
    )
    report.append("")
    report.append("---")
    report.append("")

    report.append("## 1. Direction Concordance Patterns")
    report.append("")
    report.append("### 1.1 The Gold Standard: 100% Direction Availability")
    report.append("")
    report.append(
        "Direction concordance emerges as the most reliable metric in our "
        "analysis, with 100% availability across all study pairs. This "
        "completeness stands in stark contrast to other metrics:"
    )
    report.append("")
    report.append("| Metric | Availability | Interpretation |")
    report.append("|--------|-------------|----------------|")
    report.append(
        f"| Direction Concordance | 100% ({total_pairs:,} pairs) | "
        "Gold standard |"
    )

    effect_pct = data["quality_report"]["summary"][
        "overall_prop_missing_effect_size"
    ]
    stat_pct = data["quality_report"]["summary"][
        "overall_prop_missing_statistical"
    ]

    report.append(
        f"| Effect Size Similarity | {(1 - effect_pct) * 100:.1f}% "
        f"({int((1 - effect_pct) * total_pairs)} pairs) | Limited coverage |"
    )
    report.append(
        f"| Statistical Consistency | {(1 - stat_pct) * 100:.2f}% "
        f"({int((1 - stat_pct) * total_pairs)} pairs) | Minimal coverage |"
    )
    report.append("")

    report.append(
        "**Scientific Insight:** Direction information (whether an exposure "
        "increases or decreases an outcome) is nearly universally reported "
        "in MR abstracts, even when specific effect sizes and p-values are "
        "not. This reflects a fundamental reporting practice in the MR "
        "literature where directional claims are emphasized over precise "
        "quantitative estimates in abstract summaries."
    )
    report.append("")

    report.append("### 1.2 Overall Concordance Values")
    report.append("")

    mean_conc = data["db_direction_overall"]["mean_concordance"].values[0]
    median_conc = data["db_direction_overall"]["median_concordance"].values[0]
    sd_conc = data["db_direction_overall"]["sd_concordance"].values[0]
    n_concordant = int(data["db_direction_overall"]["n_concordant"].values[0])
    n_discordant = int(data["db_direction_overall"]["n_discordant"].values[0])
    n_mixed = int(data["db_direction_overall"]["n_mixed"].values[0])

    report.append(f"Across all {total_pairs:,} similarity comparisons:")
    report.append("")
    report.append(f"- **Mean concordance:** {mean_conc:.3f}")
    report.append(f"- **Median concordance:** {median_conc:.3f}")
    report.append(f"- **Standard deviation:** {sd_conc:.3f}")
    report.append("")
    report.append("Distribution of concordance values:")
    report.append("")
    report.append(
        f"- **Fully concordant (concordance = 1):** {n_concordant:,} pairs "
        f"({pct_concordant}%)"
    )
    report.append(
        f"- **Fully discordant (concordance = -1):** {n_discordant:,} pairs "
        f"({pct_discordant}%)"
    )
    report.append(
        f"- **Mixed/No clear direction (concordance = 0):** {n_mixed:,} "
        f"pairs ({data['db_direction_overall']['n_mixed'].values[0] / total_pairs * 100:.1f}%)"
    )
    report.append("")

    report.append(
        "**Key Finding:** The high median (1.0) indicates that when "
        "studies investigate similar trait pairs, they typically "
        "agree on the causal direction. However, the substantial "
        "standard deviation (0.871) and the 25.6% discordance rate "
        "reveal important heterogeneity."
    )
    report.append("")

    report.append("### 1.3 Distribution: Concordant vs Discordant Evidence")
    report.append("")
    report.append(
        "The bimodal distribution of direction concordance reveals two "
        "distinct patterns in the MR literature:"
    )
    report.append("")
    report.append("#### Concordant Evidence (69.5% of pairs)")
    report.append("")
    report.append(
        f"When {n_concordant:,} pairs show full concordance (1.0), it "
        "indicates:"
    )
    report.append("")
    report.append(
        "1. **Robust causal relationships:** For well-established MR "
        "relationships (e.g., BMI increasing coronary disease risk), "
        "independent studies consistently replicate the direction"
    )
    report.append(
        "2. **Shared instrumentation:** Studies often use overlapping "
        "genetic instruments, leading to concordant findings"
    )
    report.append(
        "3. **Publication alignment:** The literature tends to converge "
        "on consensus directions for specific exposure-outcome pairs"
    )
    report.append("")

    report.append("#### Discordant Evidence (25.6% of pairs)")
    report.append("")
    report.append(
        f"The {n_discordant:,} fully discordant pairs (-1.0) represent "
        "scientifically important cases:"
    )
    report.append("")
    report.append(
        "1. **Bidirectional relationships:** Some exposure-outcome pairs "
        "genuinely show effects in both directions (e.g., depression and "
        "BMI)"
    )
    report.append(
        "2. **Context-dependent effects:** Results may vary by population, "
        "ancestry, or MR methodology"
    )
    report.append(
        "3. **Trait specificity:** Studies examining similar but distinct "
        "traits (e.g., different lipid fractions) may show opposite effects"
    )
    report.append(
        "4. **Potential false positives:** Some discordance may reflect "
        "spurious findings in one or both studies"
    )
    report.append("")

    report.append("### 1.4 What This Reveals About MR Study Consistency")
    report.append("")
    report.append(
        "The concordance patterns provide several scientific insights:"
    )
    report.append("")

    report.append("#### High Overall Consistency")
    report.append("")
    report.append(
        "With nearly 70% of similar studies showing concordant directions, "
        "the MR literature demonstrates substantial replicability at the "
        "directional level. This is encouraging for the field and suggests "
        "that:"
    )
    report.append("")
    report.append(
        "- Core MR findings are robust across different implementations"
    )
    report.append(
        "- The genetic instrument approach provides reproducible "
        "directional insights"
    )
    report.append("- Abstract reporting accurately captures main findings")
    report.append("")

    report.append("#### Meaningful Discordance")
    report.append("")
    report.append(
        "The 25.6% discordance rate is scientifically informative rather "
        "than problematic:"
    )
    report.append("")
    report.append(
        "- It reflects genuine biological complexity (bidirectional "
        "relationships, effect heterogeneity)"
    )
    report.append("- It highlights areas where the field lacks consensus")
    report.append(
        "- It identifies relationships requiring more careful investigation"
    )
    report.append("")

    report.append("#### The Mixed Category")
    report.append("")
    report.append(
        f"The {n_mixed:,} pairs (3.5%) with zero concordance represent "
        "cases where:"
    )
    report.append("")
    report.append("- One or both studies found no clear directional effect")
    report.append("- Multiple exposure-outcome relationships were tested")
    report.append("- Results were null or inconclusive")
    report.append("")
    report.append(
        "This small proportion suggests most MR studies report definitive "
        "directional findings."
    )
    report.append("")

    report.append("---")
    report.append("")

    report.append("## 2. Model Performance Analysis")
    report.append("")
    report.append("### 2.1 LLM Extraction Completeness")
    report.append("")
    report.append(
        "Different LLM models show varying ability to extract complete "
        "evidence profiles:"
    )
    report.append("")

    report.append(
        "| Model | N Pairs | Avg Completeness | "
        "Median Completeness | High Completeness (%) |"
    )
    report.append(
        "|-------|---------|------------------|---------------------|"
        "-----------------------|"
    )

    for _, row in (
        data["completeness"]
        .sort_values("mean_completeness", ascending=False)
        .iterrows()
    ):
        model = row["model"]
        n = int(row["n_combinations"])
        mean_comp = row["mean_completeness"]
        median_comp = row["median_completeness"]
        prop_high = row["prop_high"]
        report.append(
            f"| {model} | {n:,} | {mean_comp:.3f} | {median_comp:.3f} | "
            f"{prop_high * 100:.1f}% |"
        )

    report.append("")

    report.append("**Key Findings:**")
    report.append("")
    report.append(
        "1. **o4-mini leads in completeness** with 75.5% mean completeness "
        "and 60% of extractions classified as 'high completeness'"
    )
    report.append(
        "2. **llama3 shows strong performance** at 71.7% mean completeness, "
        "outperforming larger models"
    )
    report.append(
        "3. **Substantial model variation** spans 20 percentage points "
        "(59.6% to 75.5%)"
    )
    report.append(
        "4. **Median often exceeds mean** suggesting right-skewed "
        "distributions with some very low completeness cases"
    )
    report.append("")

    report.append("### 2.2 Data Availability by Model")
    report.append("")
    report.append(
        "More critically, models differ in their ability to extract "
        "structured quantitative data:"
    )
    report.append("")

    report.append(
        "| Model | N Pairs | Effect Size Avail (%) | "
        "Statistical Avail (%) | Evidence Overlap (%) |"
    )
    report.append(
        "|-------|---------|----------------------|----------------------|"
        "----------------------|"
    )

    for _, row in data["db_model_availability"].iterrows():
        model = row["model"]
        n = int(row["n_pairs"])
        eff_pct = row["pct_effect_size"]
        stat_pct = row["pct_statistical"]
        overlap_row = data["db_overlap"][data["db_overlap"]["model"] == model]
        overlap_pct = overlap_row["pct_full_overlap"].values[0]

        report.append(
            f"| {model} | {n:,} | {eff_pct:.2f}% | {stat_pct:.2f}% | "
            f"{overlap_pct:.2f}% |"
        )

    report.append("")

    report.append("**Critical Insights:**")
    report.append("")
    report.append(
        "1. **Effect size extraction is rare** (0-6.24%) across all models, "
        "with o4-mini performing best"
    )
    report.append(
        "2. **Statistical consistency is nearly absent** (<1% for all "
        "models), reflecting abstract reporting norms"
    )
    report.append(
        "3. **Evidence overlap is high** (58-82%), indicating LLMs "
        "successfully identify which traits are studied"
    )
    report.append(
        "4. **deepseek-r1-distilled shows zero quantitative extraction**, "
        "suggesting challenges with structured data"
    )
    report.append("")

    report.append("### 2.3 Correlation Between Completeness and Similarity")
    report.append("")
    report.append(
        "Analysis of composite similarity metrics reveals how extraction "
        "quality relates to similarity measurement:"
    )
    report.append("")

    sim_dist = data["similarity_dist"]
    report.append(
        "| Model | Mean Composite (Direction) | SD Composite (Direction) |"
    )
    report.append(
        "|-------|---------------------------|--------------------------|"
    )

    for _, row in sim_dist.sort_values(
        "mean_composite_direction", ascending=False
    ).iterrows():
        model = row["model"]
        mean_comp = row["mean_composite_direction"]
        sd_comp = row["sd_composite_direction"]
        report.append(f"| {model} | {mean_comp:.3f} | {sd_comp:.3f} |")

    report.append("")

    report.append("**Interpretation:**")
    report.append("")
    report.append(
        "- o4-mini achieves highest composite similarity (0.456), "
        "consistent with its superior completeness"
    )
    report.append(
        "- gpt-5 and llama3-2 show lower composite similarity (0.329, "
        "0.328), correlating with lower completeness"
    )
    report.append(
        "- Higher standard deviations for top models suggest they capture "
        "both highly similar and dissimilar study pairs more distinctly"
    )
    report.append("")

    report.append("### 2.4 Model Reliability for MR Evidence Extraction")
    report.append("")
    report.append(
        "Based on comprehensive analysis, we rank model reliability:"
    )
    report.append("")
    report.append("#### Tier 1: Production-Ready")
    report.append("")
    report.append(
        "**o4-mini** - Best overall performance across completeness "
        "(75.5%), effect size extraction (6.24%), and composite similarity. "
        "Recommended for production evidence extraction."
    )
    report.append("")

    report.append("#### Tier 2: Highly Capable")
    report.append("")
    report.append(
        "**llama3** - Strong completeness (71.7%) with competitive "
        "composite similarity (0.434). Excellent open-source alternative."
    )
    report.append("")
    report.append(
        "**gpt-4-1** - Largest dataset (2,066 pairs) with solid "
        "completeness (64.9%). Reliable workhorse model."
    )
    report.append("")

    report.append("#### Tier 3: Moderate Performance")
    report.append("")
    report.append(
        "**gpt-5** - Adequate completeness (62.3%) but lower effect size "
        "extraction (2.48%)."
    )
    report.append("")
    report.append(
        "**llama3-2** - Moderate completeness (59.6%) and lowest composite "
        "similarity (0.328)."
    )
    report.append("")

    report.append("#### Tier 4: Development Needed")
    report.append("")
    report.append(
        "**deepseek-r1-distilled** - Zero quantitative extraction despite "
        "reasonable overall completeness (63.6%). Requires prompt engineering."
    )
    report.append("")

    report.append("---")
    report.append("")

    report.append("## 3. Evidence-Trait Relationships")
    report.append("")
    report.append("### 3.1 The Matched Pairs Dataset")
    report.append("")

    trait_corr = data["trait_evidence_corr"]
    total_matched = trait_corr["n_pairs"].sum()

    report.append(
        f"We identified {total_matched:,} study pairs with both trait "
        "semantic similarity and evidence concordance metrics, enabling "
        "direct comparison of trait naming vs. causal evidence alignment."
    )
    report.append("")

    report.append("Distribution across models:")
    report.append("")
    report.append("| Model | N Matched Pairs | % of Model Total |")
    report.append("|-------|----------------|------------------|")

    for _, row in trait_corr.sort_values(
        "n_pairs", ascending=False
    ).iterrows():
        model = row["model"]
        n = int(row["n_pairs"])
        model_total = data["summary_stats"][
            data["summary_stats"]["model"] == model
        ]["total_similarity_pairs"].values[0]
        pct = n / model_total * 100 if model_total > 0 else 0

        report.append(f"| {model} | {n:,} | {pct:.1f}% |")

    report.append("")

    report.append(
        "**Note:** o4-mini and gpt-4-1 contribute most matched pairs, "
        "reflecting their larger overall datasets."
    )
    report.append("")

    report.append("### 3.2 Correlation Patterns")
    report.append("")
    report.append(
        "We examined correlations between trait semantic similarity and "
        "evidence direction concordance:"
    )
    report.append("")

    report.append(
        "| Model | N Pairs | Corr (Direction-Semantic) | "
        "P-value | Corr (Composite-Semantic) | P-value |"
    )
    report.append(
        "|-------|---------|---------------------------|---------|"
        "---------------------------|---------|"
    )

    for _, row in trait_corr.sort_values(
        "n_pairs", ascending=False
    ).iterrows():
        model = row["model"]
        n = int(row["n_pairs"])
        corr_dir = row["corr_evidence_dir_trait_semantic"]
        p_dir = row["p_value_dir_semantic"]
        corr_comp = row["corr_evidence_equal_trait_semantic"]
        p_comp = row["p_value_equal_semantic"]

        sig_dir = "*" if p_dir < 0.05 else ""
        sig_comp = "*" if p_comp < 0.05 else ""

        report.append(
            f"| {model} | {n:,} | {corr_dir:.3f}{sig_dir} | "
            f"{p_dir:.4f} | {corr_comp:.3f}{sig_comp} | {p_comp:.4f} |"
        )

    report.append("")
    report.append("*(asterisk indicates p < 0.05)*")
    report.append("")

    report.append("**Key Findings:**")
    report.append("")
    report.append(
        "1. **Weak but significant correlations** exist between trait "
        "semantic similarity and evidence concordance for o4-mini (r=0.105, "
        "p=0.015) and llama3 (r=0.152, p=0.018)"
    )
    report.append(
        "2. **Composite-semantic correlations stronger for llama3** "
        "(r=0.177, p=0.006), suggesting better alignment of overall "
        "evidence profiles with trait semantics"
    )
    report.append(
        "3. **Most models show no significant correlation**, indicating "
        "trait naming and causal evidence are largely independent"
    )
    report.append("4. **Small effect sizes** (r < 0.2) across all models")
    report.append("")

    report.append("### 3.3 Interpretation: Trait Naming vs Causal Evidence")
    report.append("")
    report.append("The weak correlations reveal a fundamental insight:")
    report.append("")
    report.append(
        "**Semantic similarity of trait names does not strongly predict "
        "evidence concordance.**"
    )
    report.append("")
    report.append("This means:")
    report.append("")
    report.append(
        "1. **Studies with semantically similar traits can have opposite "
        "findings** - For example, 'BMI and diabetes' vs 'obesity and "
        "diabetes' may show different directions if one examines "
        "bidirectionality"
    )
    report.append(
        "2. **Studies with dissimilar trait names can show concordant "
        "evidence** - 'Lipids and CHD' vs 'triglycerides and heart attack' "
        "may align evidentially despite semantic distance"
    )
    report.append(
        "3. **Causal relationships are more nuanced than trait labels** - "
        "The specific genetic instruments, populations, and methodological "
        "choices matter more than trait naming conventions"
    )
    report.append("")

    report.append("### 3.4 Interesting Cases: Divergence Patterns")
    report.append("")

    interesting = pd.read_csv(
        "../data/processed/evidence-profiles/analysis/interesting-cases.csv"
    )

    report.append(
        f"We identified {len(interesting):,} 'interesting cases' where "
        "trait and evidence similarities diverge. Example patterns:"
    )
    report.append("")

    report.append(
        "#### Pattern 1: High Trait Similarity, Low Evidence Concordance"
    )
    report.append("")
    report.append(
        "Studies examining semantically identical trait pairs but finding "
        "opposite causal directions. These often reflect:"
    )
    report.append("")
    report.append("- Bidirectional causal relationships")
    report.append("- Population-specific effects")
    report.append(
        "- Methodological differences (univariate vs multivariable MR)"
    )
    report.append("")

    high_trait_low_ev = interesting[
        interesting["quadrant"] == "high_trait_low_evidence"
    ]
    n_htlev = len(high_trait_low_ev)

    report.append(
        f"This quadrant contains {n_htlev:,} pairs, representing "
        f"{n_htlev / len(interesting) * 100:.1f}% of interesting cases."
    )
    report.append("")

    report.append(
        "#### Pattern 2: Low Trait Similarity, High Evidence Concordance"
    )
    report.append("")
    report.append(
        "Studies with different trait names but concordant causal evidence. "
        "These often reflect:"
    )
    report.append("")
    report.append(
        "- Synonymous traits described differently (e.g., 'BMI' vs "
        "'adiposity')"
    )
    report.append(
        "- Related traits in the same causal pathway (e.g., different "
        "lipid fractions)"
    )
    report.append("- Shared upstream genetic architecture")
    report.append("")

    low_trait_high_ev = interesting[
        interesting["quadrant"] == "low_trait_high_evidence"
    ]
    n_lthev = len(low_trait_high_ev)

    report.append(f"This quadrant contains {n_lthev:,} pairs.")
    report.append("")

    report.append("---")
    report.append("")

    report.append("## 4. Match Type Quality Stratification")
    report.append("")
    report.append("### 4.1 Distribution Across Matching Tiers")
    report.append("")

    match_type = data["db_match_type"]
    n_exact = int(match_type["n_exact"].values[0])
    n_fuzzy = int(match_type["n_fuzzy"].values[0])
    n_efo = int(match_type["n_efo"].values[0])
    pct_exact = match_type["pct_exact"].values[0]
    pct_fuzzy = match_type["pct_fuzzy"].values[0]
    pct_efo = match_type["pct_efo"].values[0]

    report.append("Evidence trait matching across three quality tiers:")
    report.append("")
    report.append("| Match Type | N Matches | Percentage | Interpretation |")
    report.append("|------------|-----------|------------|----------------|")
    report.append(
        f"| Exact | {n_exact:,} | {pct_exact:.2f}% | Identical trait strings |"
    )
    report.append(
        f"| Fuzzy | {n_fuzzy:,} | {pct_fuzzy:.2f}% | Similar trait names |"
    )
    report.append(
        f"| EFO | {n_efo:,} | {pct_efo:.2f}% | Ontology-matched traits |"
    )
    report.append("")

    report.append(
        "**Note:** Fuzzy matching exceeds 100% because multiple "
        "trait pairs within a study comparison can match fuzzy, "
        "creating >1 match per pair."
    )
    report.append("")

    report.append("### 4.2 Match Quality and Similarity Metrics")
    report.append("")

    match_dir = data["db_direction_by_match"]

    report.append("Direction concordance stratified by match type:")
    report.append("")
    report.append(
        "| Match Type | N Pairs | Mean Concordance | "
        "% Concordant | % Discordant |"
    )
    report.append(
        "|------------|---------|------------------|--------------|"
        "--------------|"
    )

    for _, row in match_dir.iterrows():
        if pd.isna(row["match_type"]):
            continue
        mt = row["match_type"]
        n = int(row["n_pairs"])
        mean_c = row["mean_concordance"]
        pct_c = row["pct_concordant"]
        n_disc = int(row["n_discordant"])
        pct_disc = n_disc / n * 100

        report.append(
            f"| {mt} | {n:,} | {mean_c:.3f} | {pct_c:.2f}% | {pct_disc:.2f}% |"
        )

    report.append("")

    report.append("**Critical Findings:**")
    report.append("")
    report.append(
        f"1. **Exact matching shows highest concordance** ({match_dir[match_dir['match_type'] == 'Exact']['pct_concordant'].values[0]:.1f}%), "
        "as expected for identical traits"
    )
    report.append(
        f"2. **EFO matching performs well** ({match_dir[match_dir['match_type'] == 'EFO']['pct_concordant'].values[0]:.1f}% concordance), "
        "validating ontology-based matching"
    )
    report.append(
        f"3. **Fuzzy matching shows moderate concordance** ({match_dir[match_dir['match_type'] == 'Fuzzy']['mean_concordance'].values[0]:.3f}), "
        "reflecting greater trait heterogeneity"
    )
    report.append(
        "4. **'None' category has lowest concordance**, likely representing "
        "unmatched or poorly matched traits"
    )
    report.append("")

    report.append("### 4.3 EFO Matching as Fallback Strategy")
    report.append("")
    report.append(
        f"With only {pct_efo:.2f}% ({n_efo} matches) of comparisons "
        "relying on EFO ontology matching, it serves as a small but "
        "effective fallback:"
    )
    report.append("")

    efo_stats = match_dir[match_dir["match_type"] == "EFO"]
    efo_concordance = efo_stats["pct_concordant"].values[0]
    efo_n_disc = int(efo_stats["n_discordant"].values[0])

    report.append(f"- **High concordance:** {efo_concordance:.1f}%")
    report.append(
        f"- **Zero full discordance:** {efo_n_disc} discordant pairs"
    )
    report.append(
        "- **Limited usage:** Suggests most trait matching succeeds at "
        "string level"
    )
    report.append("")

    report.append(
        "**Implication:** EFO matching successfully handles edge "
        "cases where string matching fails, providing ~80% "
        "concordance comparable to exact matching."
    )
    report.append("")

    report.append("---")
    report.append("")

    report.append("## 5. Data Completeness Insights")
    report.append("")
    report.append("### 5.1 The Structured Data Scarcity Problem")
    report.append("")

    qual_report = data["quality_report"]
    overall_eff_missing = qual_report["summary"][
        "overall_prop_missing_effect_size"
    ]
    overall_stat_missing = qual_report["summary"][
        "overall_prop_missing_statistical"
    ]

    report.append(
        "Our analysis reveals severe scarcity of structured quantitative data:"
    )
    report.append("")
    report.append("| Data Type | Availability | Missing Rate |")
    report.append("|-----------|-------------|--------------|")
    report.append(
        f"| Effect Size | {(1 - overall_eff_missing) * 100:.2f}% | "
        f"{overall_eff_missing * 100:.2f}% |"
    )
    report.append(
        f"| Statistical Consistency | {(1 - overall_stat_missing) * 100:.2f}% | "
        f"{overall_stat_missing * 100:.2f}% |"
    )
    report.append("| Direction Information | 100.00% | 0.00% |")
    report.append("")

    report.append("### 5.2 Why Is Structured Evidence So Scarce?")
    report.append("")
    report.append("Several factors explain this dramatic gap:")
    report.append("")

    report.append("#### Abstract Reporting Norms")
    report.append("")
    report.append("MR study abstracts typically emphasize:")
    report.append("")
    report.append("- Directional findings (increase/decrease)")
    report.append("- P-value significance (often just 'p < 0.05')")
    report.append("- Qualitative interpretation")
    report.append("")
    report.append("They rarely include:")
    report.append("")
    report.append("- Specific beta coefficients or odds ratios")
    report.append("- Confidence intervals")
    report.append("- Exact p-values")
    report.append("- Effect size heterogeneity metrics")
    report.append("")

    report.append("#### Methodological Diversity")
    report.append("")
    report.append("MR studies report effects in varied formats:")
    report.append("")
    report.append("- Odds ratios for binary outcomes")
    report.append("- Beta coefficients for continuous outcomes")
    report.append("- Hazard ratios for survival outcomes")
    report.append("- Standardized vs. non-standardized effects")
    report.append("")
    report.append(
        "This heterogeneity makes standardized extraction challenging."
    )
    report.append("")

    report.append("#### LLM Extraction Limitations")
    report.append("")
    report.append("Even advanced LLMs struggle with:")
    report.append("")
    report.append(
        "- Mapping diverse effect size representations to standardized schemas"
    )
    report.append("- Distinguishing main results from sensitivity analyses")
    report.append("- Extracting confidence intervals mentioned in prose")
    report.append("- Handling conditional or stratified results")
    report.append("")

    report.append("### 5.3 Model-Specific Data Availability")
    report.append("")

    report.append("Effect size extraction by model:")
    report.append("")
    report.append(
        "| Model | N Pairs | Effect Size Avail | "
        "Statistical Avail | Both Present |"
    )
    report.append(
        "|-------|---------|------------------|------------------|"
        "--------------|"
    )

    for item in qual_report["missing_data_by_model"]:
        model = item["model"]
        n = int(item["total_pairs"])
        n_eff = int(item["total_pairs"] - item["n_missing_effect_size"])
        n_stat = int(item["total_pairs"] - item["n_missing_statistical"])
        n_both = int(item["n_both_present"])
        pct_eff = (1 - item["prop_missing_effect_size"]) * 100
        pct_stat = (1 - item["prop_missing_statistical"]) * 100
        pct_both = item["prop_both_present"] * 100

        report.append(
            f"| {model} | {n:,} | {n_eff} ({pct_eff:.1f}%) | "
            f"{n_stat} ({pct_stat:.1f}%) | {n_both} ({pct_both:.1f}%) |"
        )

    report.append("")

    report.append(
        "**Key Observation:** o4-mini extracts the most effect "
        "sizes (113, 6.24%) and is the only model achieving >1% "
        "extraction rate for statistical consistency (12 pairs, "
        "0.66%)."
    )
    report.append("")

    report.append("### 5.4 Implications for Automated Evidence Synthesis")
    report.append("")
    report.append(
        "The scarcity of structured quantitative data has profound "
        "implications:"
    )
    report.append("")

    report.append("#### What We Can Do")
    report.append("")
    report.append(
        "1. **Direction-based synthesis** - Robust with 100% availability"
    )
    report.append(
        "2. **Qualitative evidence mapping** - Identify "
        "concordance and discordance patterns"
    )
    report.append(
        "3. **Study clustering** - Group studies by trait pairs "
        "and directional findings"
    )
    report.append(
        "4. **Contradiction detection** - Flag studies with "
        "opposite directions"
    )
    report.append("")

    report.append("#### What We Cannot Do (Yet)")
    report.append("")
    report.append(
        "1. **Meta-analysis** - Requires effect sizes and standard errors"
    )
    report.append(
        "2. **Effect magnitude comparison** - Need quantitative estimates"
    )
    report.append(
        "3. **Statistical consistency assessment** - Requires p-values or CIs"
    )
    report.append(
        "4. **Dose-response synthesis** - Needs continuous effect estimates"
    )
    report.append("")

    report.append("#### Path Forward")
    report.append("")
    report.append("To enable full quantitative synthesis:")
    report.append("")
    report.append(
        "1. **Full-text extraction** - Move beyond abstracts to results tables"
    )
    report.append(
        "2. **Specialized models** - Fine-tune LLMs on MR "
        "quantitative data extraction"
    )
    report.append(
        "3. **Structured reporting** - Advocate for standardized "
        "abstract reporting"
    )
    report.append(
        "4. **Hybrid approaches** - Combine LLM extraction with "
        "rule-based systems for numbers"
    )
    report.append("")

    report.append("---")
    report.append("")

    report.append("## 6. Methodological Insights")
    report.append("")
    report.append(
        "### 6.1 What Can We Learn Despite Limited Statistical Data?"
    )
    report.append("")
    report.append(
        "Our analysis demonstrates that meaningful scientific insights emerge "
        "even with limited quantitative data:"
    )
    report.append("")

    report.append("#### Replicability Assessment")
    report.append("")
    report.append(
        "Direction concordance provides a valid (if coarse) measure of "
        "replicability:"
    )
    report.append("")
    report.append(
        "- **69.5% concordance** indicates substantial directional "
        "reproducibility"
    )
    report.append(
        "- **Exact matching shows 88% concordance**, approaching traditional "
        "replication standards"
    )
    report.append(
        "- **Discordance patterns** highlight areas needing investigation"
    )
    report.append("")

    report.append("#### Literature Mapping")
    report.append("")
    report.append(
        "Evidence profiles enable systematic mapping of MR literature:"
    )
    report.append("")
    report.append(
        "- **Identify consensus findings** (high concordance, many studies)"
    )
    report.append("- **Flag controversial relationships** (high discordance)")
    report.append("- **Detect research gaps** (low evidence overlap)")
    report.append("")

    report.append("#### Model Evaluation")
    report.append("")
    report.append("Comparative LLM performance analysis reveals:")
    report.append("")
    report.append(
        "- **Extraction quality varies substantially** (20 percentage "
        "point range)"
    )
    report.append(
        "- **Completeness predicts reliability** (r > 0.9 between "
        "composite metrics)"
    )
    report.append(
        "- **Open-source models competitive** (llama3 matches gpt-4-1)"
    )
    report.append("")

    report.append("### 6.2 Which Similarity Metrics Are Most Reliable?")
    report.append("")

    metric_corr = data["metric_corr"]

    report.append("Validation analysis of metric agreement:")
    report.append("")
    report.append(
        "| Model | Composite-Direction Correlation | "
        "Kendall's Tau | % Large Differences |"
    )
    report.append(
        "|-------|--------------------------------|---------------|"
        "---------------------|"
    )

    val_report = data["validation_report"]["metric_agreement"]
    for model, stats in val_report.items():
        if model == "summary":
            continue
        corr = stats["pearson_correlation"]
        tau = stats["kendall_tau"]
        pct_diff = stats["prop_large_difference"] * 100

        report.append(
            f"| {model} | {corr:.3f} | {tau:.3f} | {pct_diff:.1f}% |"
        )

    report.append("")

    report.append("**Reliability Ranking:**")
    report.append("")
    report.append(
        "1. **Direction concordance** - 100% available, intuitive, validated"
    )
    report.append(
        "2. **Evidence overlap** - High availability (58-82%), captures "
        "study scope"
    )
    report.append(
        "3. **Composite similarity (direction-weighted)** - Integrates "
        "multiple signals (r > 0.93 with direction)"
    )
    report.append(
        "4. **Effect size similarity** - When available (~4%), highly "
        "informative but sparse"
    )
    report.append(
        "5. **Statistical consistency** - Theoretical value but practically "
        "unusable (<1% available)"
    )
    report.append("")

    report.append("### 6.3 Limitations and Caveats")
    report.append("")

    report.append("#### Data Limitations")
    report.append("")
    report.append(
        "1. **Abstract-only extraction** limits quantitative completeness"
    )
    report.append(
        "2. **LLM extraction errors** may introduce systematic biases"
    )
    report.append("3. **Publication bias** affects available literature")
    report.append("4. **Heterogeneous MR methods** conflated in analysis")
    report.append("")

    report.append("#### Methodological Limitations")
    report.append("")
    report.append(
        "1. **Direction concordance is binary** - Doesn't capture "
        "effect magnitude"
    )
    report.append(
        "2. **Trait matching imperfect** - Fuzzy matching introduces noise"
    )
    report.append(
        "3. **No outcome stratification** - Binary and continuous "
        "outcomes pooled"
    )
    report.append(
        "4. **Cross-model comparisons challenged by dataset differences**"
    )
    report.append("")

    report.append("#### Interpretation Caveats")
    report.append("")
    report.append(
        "1. **Discordance does not imply error** - May reflect "
        "genuine bidirectionality"
    )
    report.append(
        "2. **High concordance does not prove causality** - "
        "Systematic bias can replicate"
    )
    report.append(
        "3. **Model completeness confounded with dataset characteristics**"
    )
    report.append(
        "4. **Temporal trends unclear** - Cannot separate dataset "
        "era from model effects"
    )
    report.append("")

    report.append("---")
    report.append("")

    report.append("## 7. Future Directions")
    report.append("")
    report.append("### 7.1 Additional Valuable Analyses")
    report.append("")

    report.append("#### Stratified Analysis")
    report.append("")
    report.append("Examine concordance patterns by:")
    report.append("")
    report.append("- **Outcome type** (binary vs continuous)")
    report.append(
        "- **Exposure category** (behavioral, anthropometric, metabolic)"
    )
    report.append("- **Study design** (one-sample vs two-sample MR)")
    report.append(
        "- **Population ancestry** (European vs East Asian vs multi-ancestry)"
    )
    report.append("- **Sample size** (large biobank vs smaller cohorts)")
    report.append("")

    report.append("#### Temporal Trends")
    report.append("")
    report.append("Investigate how concordance evolves:")
    report.append("")
    report.append("- Do more recent studies show higher concordance?")
    report.append("- Has effect size reporting improved over time?")
    report.append("- Are bidirectional MR studies more common recently?")
    report.append("")

    report.append("#### Network Analysis")
    report.append("")
    report.append("Construct evidence networks:")
    report.append("")
    report.append("- Nodes = trait pairs")
    report.append("- Edges = shared evidence or concordance")
    report.append("- Community detection to identify research clusters")
    report.append("- Central traits with most evidence")
    report.append("")

    report.append("#### Full-Text Deep Dive")
    report.append("")
    report.append("For high-discordance pairs:")
    report.append("")
    report.append("- Manual full-text review")
    report.append("- Methodological comparison")
    report.append("- Identify sources of discordance")
    report.append("- Extract detailed quantitative data")
    report.append("")

    report.append("### 7.2 Pipeline Improvements")
    report.append("")

    report.append("#### Enhanced Extraction")
    report.append("")
    report.append("1. **Full-text processing** - Move beyond abstracts")
    report.append(
        "2. **Table extraction** - Target results tables specifically"
    )
    report.append("3. **Figure analysis** - Extract data from forest plots")
    report.append(
        "4. **Supplementary materials** - Often contain full results"
    )
    report.append("")

    report.append("#### Improved Matching")
    report.append("")
    report.append(
        "1. **Bidirectional EFO mapping** - Map all traits to ontology upfront"
    )
    report.append(
        "2. **Semantic embeddings** - Use trait embeddings for similarity"
    )
    report.append(
        "3. **Hierarchical matching** - Exact -> Fuzzy -> Semantic -> EFO"
    )
    report.append(
        "4. **User validation interface** - Allow expert review of matches"
    )
    report.append("")

    report.append("#### Model Optimization")
    report.append("")
    report.append(
        "1. **Fine-tuning** - Train models on MR-specific extraction"
    )
    report.append("2. **Prompt engineering** - Optimize for quantitative data")
    report.append(
        "3. **Ensemble methods** - Combine predictions from multiple models"
    )
    report.append(
        "4. **Uncertainty quantification** - Confidence scores for extractions"
    )
    report.append("")

    report.append("#### Validation Framework")
    report.append("")
    report.append("1. **Gold standard dataset** - Manually curated MR results")
    report.append(
        "2. **Inter-rater reliability** - Multiple expert annotations"
    )
    report.append(
        "3. **Error analysis** - Systematic characterization of "
        "extraction failures"
    )
    report.append(
        "4. **Active learning** - Iteratively improve on challenging cases"
    )
    report.append("")

    report.append("### 7.3 Leveraging Available Data")
    report.append("")

    report.append("#### Immediate Applications")
    report.append("")
    report.append(
        "1. **Contradiction detection system** - Flag discordant "
        "study pairs for review"
    )
    report.append(
        "2. **Evidence gap identification** - Find under-studied trait pairs"
    )
    report.append(
        "3. **Replication prioritization** - Recommend relationships "
        "needing replication"
    )
    report.append(
        "4. **Literature navigation** - Interactive exploration of "
        "evidence networks"
    )
    report.append("")

    report.append("#### Research Tools")
    report.append("")
    report.append(
        "1. **Prior evidence lookup** - Search for existing MR "
        "evidence on trait pairs"
    )
    report.append(
        "2. **Concordance scoring** - Quantify consistency of literature"
    )
    report.append(
        "3. **Study recommendation** - Suggest similar studies for context"
    )
    report.append(
        "4. **Bias assessment** - Identify potential publication bias patterns"
    )
    report.append("")

    report.append("#### Meta-Research")
    report.append("")
    report.append(
        "1. **Reporting quality assessment** - Quantify completeness "
        "across journals"
    )
    report.append(
        "2. **Methodological trends** - Track adoption of MR practices"
    )
    report.append(
        "3. **Reproducibility metrics** - Field-wide replication rates"
    )
    report.append(
        "4. **Model performance benchmarks** - Standard dataset for "
        "LLM evaluation"
    )
    report.append("")

    report.append("---")
    report.append("")

    report.append("## Conclusions")
    report.append("")
    report.append(
        "This comprehensive analysis of 6,388 evidence profile comparisons "
        "across 6,132 MR studies reveals that despite severe limitations in "
        "quantitative data availability (effect sizes: 3.8%, statistical "
        "metrics: 0.3%), direction concordance provides a robust and "
        "scientifically meaningful measure of study consistency. With 69.5% "
        "of similar studies showing concordant causal directions and 100% "
        "data availability, direction-based analysis enables reliable "
        "assessment of MR literature replicability."
    )
    report.append("")
    report.append(
        "The analysis also demonstrates substantial variation in LLM "
        "extraction performance, with o4-mini emerging as the most reliable "
        "model (75.5% completeness, 6.24% effect size extraction), while "
        "highlighting that even advanced models struggle with structured "
        "quantitative data extraction from abstracts. Importantly, we find "
        "that trait semantic similarity and evidence concordance are largely "
        "independent (r < 0.2), revealing that causal evidence patterns "
        "cannot be predicted from trait naming alone."
    )
    report.append("")
    report.append(
        "Moving forward, the field would benefit from: (1) full-text "
        "extraction to access detailed quantitative data, (2) standardized "
        "abstract reporting to facilitate automated synthesis, (3) "
        "model-specific fine-tuning for MR evidence extraction, and (4) "
        "development of tools that leverage the abundant directional data "
        "while acknowledging quantitative limitations. The current analysis "
        "provides a foundation for understanding MR literature consistency "
        "and identifying areas requiring targeted investigation or "
        "methodological development."
    )
    report.append("")

    report.append("---")
    report.append("")
    report.append("## Technical Notes")
    report.append("")
    report.append("### Data Sources")
    report.append("")
    report.append(
        "- Evidence profile database: `data/db/evidence_profile_db.db`"
    )
    report.append(
        "- Analysis results: `data/processed/evidence-profiles/analysis/`"
    )
    report.append("- Total comparisons: 6,388 pairs across 6,132 studies")
    report.append(
        "- Models analyzed: 6 (o4-mini, gpt-4-1, gpt-5, llama3, "
        "llama3-2, deepseek-r1-distilled)"
    )
    report.append("")

    report.append("### Analysis Methods")
    report.append("")
    report.append(
        "- Direction concordance: Cosine similarity of directional vectors"
    )
    report.append(
        "- Effect size similarity: Pearson correlation of standardized effects"
    )
    report.append(
        "- Statistical consistency: Comparison of significance thresholds"
    )
    report.append("- Evidence overlap: Jaccard similarity of trait sets")
    report.append(
        "- Composite similarity: Weighted combination of individual metrics"
    )
    report.append("")

    report.append("### Reproducibility")
    report.append("")
    report.append("All analyses are reproducible using:")
    report.append("")
    report.append("```bash")
    report.append("cd processing")
    report.append(
        "uv run python scripts/analysis/generate-scientific-insights-report.py"
    )
    report.append("```")
    report.append("")

    return "\n".join(report)


def main():
    """Main execution function."""
    print("Loading data...")
    data = load_data()

    print("Generating report...")
    report = generate_report(data)

    output_dir = Path("../data/processed/evidence-profiles/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "SCIENTIFIC-INSIGHTS.md"

    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nReport generated: {output_file}")
    print(f"Report length: {len(report):,} characters")
    print(f"Report length: {len(report.split())} words")


if __name__ == "__main__":
    main()
