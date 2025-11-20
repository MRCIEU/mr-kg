# Glossary of key terms

This document provides detailed explanations of key concepts and terms used
throughout the MR-KG project documentation and codebase.

## Core concepts

### Match type

Classification of how exposure-outcome trait pairs are matched between studies
when computing evidence profile similarities and reproducibility metrics.

There are three match types:

1. **Exact match**: Trait labels are identical strings after normalization
   (case-insensitive, whitespace-normalized). This represents the highest
   confidence that studies are investigating the same exposure-outcome
   relationship.

   Example: Study A reports "body mass index" -> "type 2 diabetes" and Study B
   reports "Body Mass Index" -> "Type 2 Diabetes". After normalization, these
   are exact matches.

2. **Fuzzy match**: Trait labels are similar but not identical, matched using
   string similarity algorithms (e.g., Levenshtein distance, trigram
   similarity). This captures trait pairs that likely refer to the same
   concept despite spelling variations or minor wording differences.

   Example: "body mass index" vs "body-mass index" or "type 2 diabetes" vs
   "type II diabetes". These would be fuzzy matches.

3. **EFO match**: Traits are mapped to the same Experimental Factor Ontology
   (EFO) term, establishing semantic equivalence through standardized
   biomedical ontology. This provides the most robust matching for traits
   expressed with different terminology but referring to the same biomedical
   concept.

   Example: "BMI" and "body mass index" both map to EFO:0004340, or
   "myocardial infarction" and "heart attack" both map to the same EFO term.

Match type is tracked for each trait pair comparison using boolean indicators:
`has_exact_match`, `has_fuzzy_match`, `has_efo_match`. A single study pair may
have different match types for different trait pairs.

Match type sensitivity varies by disease category. Case Study 1 analysis
revealed a significant Category x Match type interaction (p = 0.025), showing:

- Autoimmune outcomes: 67.2 percentage point concordance gap between exact and
  fuzzy matches
- Metabolic outcomes: 47.9 percentage point gap
- Cardiovascular outcomes: 25.5 percentage point gap
- Psychiatric outcomes: 17.8 percentage point gap (most robust to match
  quality)
- Cancer outcomes: 15.3 percentage point gap (universally low concordance
  regardless of match type)

This domain-specific pattern suggests that match quality thresholds should be
adapted by disease category rather than applied uniformly across all MR
research.

See @docs/processing/evidence-profile-similarity.md for how match types are
used in similarity computation.

### Direction concordance

Agreement in classified effect directions (positive, negative, null) for
matched exposure-outcome pairs across independent studies. This is the primary
metric for evidence profile similarity.

**Data availability:** 100 percent (always computable for any matched trait
pairs)

**Range:** -1 to +1

- +1: Perfect concordance (all effects agree in direction)
- 0: Equal concordant and discordant pairs
- -1: Perfect discordance (all effects opposite)

**Formula:**

$$
C_{\text{direction}}(A, B) = \frac{n_{\text{concordant}} - n_{\text{discordant}}}{n_{\text{concordant}} + n_{\text{discordant}}}
$$

where:
- Concordant pairs: Both studies report same non-zero direction (both positive
  or both negative)
- Discordant pairs: Studies report opposite directions (one positive, one
  negative)
- Null pairs: Pairs where either study reports null effect are excluded from
  calculation

See @docs/processing/evidence-profile-similarity.md for detailed computation
methodology.

### Reproducibility tier

Four-tier classification system for categorizing direction concordance quality
across independent studies investigating the same exposure-outcome trait pair.

1. **High tier** (concordance >= 0.7): Strong directional agreement; studies
   consistently report the same effect direction

2. **Moderate tier** (0.3 <= concordance < 0.7): Mixed agreement; some studies
   agree but substantial variation exists

3. **Low tier** (0.0 <= concordance < 0.3): Weak agreement; minimal directional
   consistency across studies

4. **Discordant tier** (concordance < 0.0): Systematic disagreement; studies
   report opposite effect directions more often than same direction

Reproducibility tiers are based on mean pairwise direction concordance computed
across all independent studies investigating the same trait pair. Threshold
values (0.7, 0.3, 0.0) are configurable in processing/config/case_studies.yml.

High tier pairs represent reproducible findings suitable for downstream
synthesis. Discordant tier pairs require investigation for methodological
issues, phenotype misalignment, or true heterogeneity.

See @docs/processing/case-studies.md (Case Study 1) for reproducibility
analysis methodology.

### Trait profile

The combined set of exposure and outcome traits extracted from a study,
representing all traits investigated in that research. Each trait profile is
identified uniquely by PMID and extraction model.

Trait profiles enable discovery of studies with similar research focuses by
comparing trait sets using semantic similarity (embedding-based) and Jaccard
similarity (set-based) metrics.

See @docs/processing/trait-profile-similarity.md for trait profile similarity
methodology.

### Evidence profile

The collection of statistical results for all exposure-outcome pairs
investigated in a study (identified by PMID and extraction model).

Evidence profiles capture how trait relationships behave statistically:
effect directions, effect sizes, p-values, and confidence intervals. By
comparing evidence profiles across studies, we can assess reproducibility and
consistency of MR findings.

Evidence profile similarity differs from trait profile similarity:
- Trait profile similarity: What traits are investigated (research focus)
- Evidence profile similarity: How trait relationships behave statistically
  (finding consistency)

See @docs/processing/evidence-profile-similarity.md for evidence profile
similarity methodology.

### Evidence profile similarity

A multi-metric quantitative assessment of statistical evidence alignment
between two studies, computed across matched exposure-outcome pairs using
effect size correlation, direction concordance, significance consistency, and
evidence overlap measures.

Primary metrics:
- Direction concordance (100 percent availability)
- Effect size similarity (~3.82 percent availability)
- Statistical consistency (~0.27 percent availability)
- Precision concordance (~3.33 percent availability)

See @docs/processing/evidence-profile-similarity.md for detailed methodology.

### Trait profile similarity

A quantitative measure of research focus overlap between two studies, computed
using both semantic similarity (embedding-based) and Jaccard similarity
(set-based) of their trait profiles.

See @docs/processing/trait-profile-similarity.md for detailed methodology.

### Temporal era

Year ranges defining distinct methodological periods in Mendelian Randomization
research. Used for stratification in temporal trend analyses (Case Studies 1
and 5).

Standard era definitions (configurable in processing/config/case_studies.yml):

1. **early_mr** (2003-2014 or 2010-2014): Foundation era covering early MR
   applications before widespread adoption of sensitivity analyses

2. **mr_egger** (2015-2017): Introduction of MR-Egger regression for pleiotropy
   testing (Bowden et al. 2015)

3. **mr_presso** (2018-2019): Introduction of MR-PRESSO for outlier detection
   and horizontal pleiotropy testing (Verbanck et al. 2018)

4. **within_family** (2020): Introduction of within-family designs to address
   population stratification and confounding (Brumpton et al. 2020)

5. **strobe_mr** (2021-2025): Publication of STROBE-MR reporting guidelines
   (Skrivankova et al. 2021)

Era boundaries align with major methodological publications. Studies outside
defined ranges are assigned to "other" or "unknown" eras.

Case Study 1 found no significant temporal improvements in reproducibility
across eras (p > 0.5 for all era effects), suggesting methodological
innovations have enhanced analytical rigor but not addressed root causes of
discordance (weak instruments, phenotype misalignment).

See @docs/processing/case-studies.md for temporal era usage in case studies.

### Harmonized effect size

Effect size converted to beta (log) scale to enable comparison across different
effect measures (beta, OR, HR).

Conversion rules:
- Odds ratio (OR): beta = log(OR)
- Hazard ratio (HR): beta = log(HR)
- Beta coefficients: kept unchanged

This harmonization enables correlation and comparison of effect sizes across
studies that report results in different formats.

### Matched pairs

Exposure-outcome trait combinations present in both studies being compared,
identified by matching trait indices. Matched pairs are the foundation for
computing evidence profile similarity metrics.

Pairs can be matched through exact, fuzzy, or EFO matching methods (see Match
type).

### Statistical consistency

Cohen's kappa measuring agreement in statistical significance classifications
(p < 0.05 threshold) across matched exposure-outcome pairs between studies.

**Data availability:** ~0.27 percent (limited by p-value reporting in
abstracts)

Range: -1 to +1, where +1 indicates perfect agreement in significance calls.

### Comparison count

Number of pairwise comparisons between studies. Mathematically: study_count
choose 2.

Example: If 5 studies investigate the same trait pair, comparison_count =
(5 choose 2) = 10 pairwise comparisons.

### Study count bands

Stratification bins for grouping trait pairs by replication breadth. Used in
Case Study 1 reproducibility analysis.

Bands: 2-3, 4-6, 7-10, 11+ studies

Rationale:
- 2-3 studies: Initial replications, may show selection bias toward concordant
  findings
- 4-6 studies: Moderate replication, heterogeneity begins to emerge
- 7-10 studies: Well-replicated pairs, heterogeneity often maximized
- 11+ studies: Extensively studied relationships, methodological consensus

## Database and schema terms

### Study count

Total number of independent studies investigating the same exposure-outcome
trait pair. Computed as comparison_count + 1 (adds the focal study to pairwise
comparison count).

"Independent" means unique PMID and model combinations. Used for stratification
into bands: 2-3, 4-6, 7-10, 11+ studies.

Primary metric for assessing replication breadth across the MR literature.
Case Study 1 found negative association between study count and concordance
(beta = -0.024, p < 0.001), suggesting accumulating studies surface latent
heterogeneity.

### Trait index

Integer identifier assigned to each unique trait label during preprocessing.
Trait indices enable efficient database joins and similarity computations.

Stored in data/processed/unique_traits.csv with mapping:
trait_index -> trait_label

All analyses reference traits by index rather than string labels for
consistency and performance.

### EFO term

Standardized biomedical concept from the Experimental Factor Ontology (EFO).
EFO provides hierarchical relationships and semantic mappings for experimental
variables, diseases, and phenotypes.

MR-KG uses EFO v3.80 (~67K terms) for trait harmonization and semantic
matching. Traits are mapped to EFO terms using fuzzy string matching and
embedding similarity.

EFO mappings enable robust cross-study trait matching when researchers use
different terminology for the same biomedical concept.

Source: https://github.com/EBISPOT/efo/releases

### Vector store

DuckDB database (data/db/vector_store.db) containing trait and EFO embeddings
with precomputed cosine similarity pairs for semantic search.

Contains ~1.7 billion trait-EFO similarity pairs enabling fast nearest-neighbor
queries for trait matching and evidence profile construction.

See @docs/processing/databases.md for database schema and @docs/DATA.md for
data structure details.

## Analysis-specific terms

### Hype cycle

Pattern where a trait experiences rapid increase in study frequency, peak
research activity, then subsequent decline or plateau. Identified in Case
Study 5 (fashionable traits analysis) to distinguish transient research trends
from sustained focus.

Traits with hype cycles may indicate:
- Fashionable research topics driven by funding or publication trends
- Initial excitement followed by disappointing results
- Methodological challenges leading to decreased interest

Contrasts with sustained growth patterns for traits with cumulative evidence.

### Winner's curse

Phenomenon where initial discovery studies overestimate effect sizes due to
selection for statistical significance. Replication studies typically show
smaller effects.

Case Study 5 (Phase 7) investigates winner's curse in MR literature by
comparing discovery versus replication effect sizes for multi-study trait
pairs. Expected decline: 10-30 percent in epidemiological studies.

Systematic positive decline confirms winner's curse presence. Temporal trends
reveal whether awareness (preregistration, reporting standards) has mitigated
the issue in recent publications.

### Pleiotropic pair

Exposure-outcome trait combination where the exposure is known to affect
multiple outcomes through different biological pathways (horizontal
pleiotropy).

Canonical examples:
- Body mass index: Type 2 diabetes, cardiovascular disease, osteoarthritis
- Education: Income, cognitive function, mental health
- C-reactive protein: Cardiovascular disease, diabetes, depression

Case Study 5 (Phase 6) tracks pleiotropic pair frequencies over time as an
indicator of pleiotropy awareness in MR research. Increasing study of
pleiotropic pairs alongside MR-PRESSO adoption suggests growing methodological
sophistication.

## See also

- Data structure: @docs/DATA.md
- Processing pipeline: @docs/processing/pipeline.md
- Case study analyses: @docs/processing/case-studies.md
- Database schema: @docs/processing/db-schema.md
- Trait profile similarity: @docs/processing/trait-profile-similarity.md
- Evidence profile similarity: @docs/processing/evidence-profile-similarity.md
