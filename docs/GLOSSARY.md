# Glossary of key terms

This document provides detailed explanations of key concepts and terms used
throughout the MR-KG project documentation and codebase.

## Core concepts

### Match type

Classification of how exposure-outcome trait pairs are matched between studies
when computing evidence profile similarities and reproducibility metrics.

#### Hierarchical fallback design

The three match types form a **hierarchical fallback scheme**, not independent
parallel methods.
The algorithm processes tiers in order of precision, and each trait pair is
matched at most once:

1. **Tier 1 (Exact)**: Attempt exact trait index matching first
2. **Tier 2 (Fuzzy)**: For pairs unmatched by Tier 1, attempt embedding-based
   fuzzy matching
3. **Tier 3 (EFO)**: For pairs still unmatched, attempt EFO ontology matching

This means fuzzy matching only considers pairs that failed exact matching, and
EFO matching only considers pairs that failed both exact and fuzzy matching.
The tiers do not overlap - a pair matched at a higher-precision tier will not
be re-evaluated at lower-precision tiers.

#### Match type definitions

1. **Exact match**: Trait pairs have identical trait indices, meaning they
   were assigned the same unique identifier during preprocessing.
   This represents the highest confidence that studies are investigating the
   same exposure-outcome relationship.

   Example: Study A reports "body mass index" -> "type 2 diabetes" and Study B
   reports "Body Mass Index" -> "Type 2 Diabetes".
   After normalization during preprocessing, both receive the same trait
   indices.

2. **Fuzzy match**: Trait pairs have different trait indices but high semantic
   similarity based on embedding vectors.
   Matching uses cosine similarity between 200-dimensional ScispaCy embeddings.
   Both exposure AND outcome traits must exceed the similarity threshold
   (default: 0.70) for a pair to match.

   Example: "body mass index" (trait_index=42) vs "BMI" (trait_index=107) may
   have cosine similarity of 0.85, exceeding the threshold.

3. **EFO match**: Trait pairs map to the same Experimental Factor Ontology
   (EFO) term, establishing semantic equivalence through standardized
   biomedical ontology.
   This is the lowest-precision tier, matching at the category level rather
   than the specific trait level.

   Example: "myocardial infarction" and "heart attack" both map to the same
   EFO term.

Match type is tracked for each trait pair comparison using boolean indicators:
`has_exact_match`, `has_fuzzy_match`, `has_efo_match`.
A single study pair may have different match types for different trait pairs
(e.g., some pairs matched exactly, others via fuzzy matching).

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

#### Technical implementation

The three-tier matching algorithm is implemented in
`processing/scripts/evidence-profile/compute-evidence-similarity.py`
(function `match_exposure_outcome_pairs_tiered`).

**Algorithm flow:**

```
For each query-similar study pair:
  1. Run exact matching on all query results
     -> Add matches to results with type="exact"
     -> Track matched query indices

  2. Filter to unmatched query results (not matched in step 1)
     Run fuzzy matching on unmatched results
     -> Add matches to results with type="fuzzy"
     -> Track matched query indices

  3. Filter to still-unmatched query results (not matched in steps 1-2)
     Run EFO matching on unmatched results
     -> Add matches to results with type="efo"
```

**Tier details:**

1. **Exact matching** (`match_exposure_outcome_pairs`): Direct comparison of
   trait indices.
   O(n) complexity using dictionary lookup.

2. **Fuzzy matching** (`match_exposure_outcome_pairs_fuzzy`): Cosine similarity
   between 200-dimensional ScispaCy embeddings.
   Both exposure AND outcome traits must have similarity >= threshold
   (default: 0.70, configurable via `--fuzzy-threshold`).
   When multiple candidates exist, selects the best match using combined score
   `(exp_sim + out_sim) / 2`.
   Greedy algorithm: each similar result can only be matched once.

3. **EFO matching** (`match_exposure_outcome_pairs_efo`): Ontology-based
   matching through shared EFO term mappings.
   Category-level matching (lowest precision).

See docs/processing/evidence-profile-similarity.md for how match types are
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

#### Technical implementation

Direction concordance is computed in
`processing/scripts/evidence-profile/compute-evidence-similarity.py`
(function `compute_direction_concordance`).

Direction classification occurs during evidence profile preprocessing in
`processing/scripts/evidence-profile/preprocess-evidence-profiles.py`
(function `classify_direction`). Effect directions are converted from string
labels (e.g., "positive", "negative", "null") to numeric indicators: +1
(positive), -1 (negative), 0 (null/unclear). The classifier uses pattern
matching against predefined term lists for each direction category.

The concordance calculation:
1. Iterates through matched trait pairs
2. Excludes pairs where either study has direction = 0 (null effect)
3. Counts concordant pairs (same non-zero direction) and discordant pairs
   (opposite directions)
4. Returns `(concordant - discordant) / (concordant + discordant)`

See docs/processing/evidence-profile-similarity.md for detailed computation
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

#### Technical implementation

Reproducibility tier classification is performed in Case Study 1 analysis
scripts. The tier assignment logic:
1. Computes mean pairwise direction concordance for each trait pair across all
   independent studies
2. Applies threshold-based classification (default thresholds: 0.7, 0.3, 0.0)
3. Assigns tier labels: "High", "Moderate", "Low", "Discordant"

Threshold values are configurable in `processing/config/case_studies.yml`.

See docs/processing/case-studies.md (Case Study 1) for reproducibility
analysis methodology.

### Trait profile

The combined set of exposure and outcome traits extracted from a study,
representing all traits investigated in that research. Each trait profile is
identified uniquely by PMID and extraction model.

Trait profiles enable discovery of studies with similar research focuses by
comparing trait sets using semantic similarity (embedding-based) and Jaccard
similarity (set-based) metrics.

#### Technical implementation

Trait profile similarity computation is implemented in
`processing/scripts/trait-profile/compute-trait-similarity.py`. The script
compares trait profiles using:
1. Semantic similarity: Cosine distance between aggregated trait embeddings
2. Jaccard similarity: Set-based overlap of unique trait indices

Results are stored in `data/db/trait_profile_db.db`.

See docs/processing/trait-profile-similarity.md for trait profile similarity
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

See docs/processing/evidence-profile-similarity.md for evidence profile
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

#### Technical implementation

Evidence profile similarity computation is implemented in
`processing/scripts/evidence-profile/compute-evidence-similarity.py`. Key
functions:
- `compute_direction_concordance`: Direction agreement metric
- `compute_effect_size_correlation`: Pearson correlation of
  harmonized effect sizes
- `compute_statistical_consistency`: Cohen's kappa for significance
  agreement
- `compute_precision_concordance`: Correlation of standard errors

Results are stored in `data/db/evidence_profile_db.db`.

See docs/processing/evidence-profile-similarity.md for detailed methodology.

### Trait profile similarity

A quantitative measure of research focus overlap between two studies, computed
using both semantic similarity (embedding-based) and Jaccard similarity
(set-based) of their trait profiles.

See docs/processing/trait-profile-similarity.md for detailed methodology.

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

See docs/processing/case-studies.md for temporal era usage in case studies.

### Harmonized effect size

Effect size converted to beta (log) scale to enable comparison across different
effect measures (beta, OR, HR).

Conversion rules:
- Odds ratio (OR): beta = log(OR)
- Hazard ratio (HR): beta = log(HR)
- Beta coefficients: kept unchanged

This harmonization enables correlation and comparison of effect sizes across
studies that report results in different formats.

#### Technical implementation

Effect size harmonization is performed during evidence profile preprocessing in
`processing/scripts/evidence-profile/preprocess-evidence-profiles.py`
(function `harmonize_effect_size`).

The transformation logic:
1. For OR and HR values: applies natural logarithm transformation
   (`math.log(value)`)
2. For beta coefficients: returns the value unchanged
3. Validates that OR/HR values are positive before transformation
4. Returns None for invalid values or unknown effect types

The harmonized effect sizes are stored in the `harmonized_effect_size` field
of evidence profile records and used for computing effect size correlation
metrics (`compute-evidence-similarity.py`).

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

#### Technical implementation

Statistical consistency computation is implemented in
`processing/scripts/evidence-profile/compute-evidence-similarity.py`
(function `compute_statistical_consistency`). The function uses Cohen's kappa
to measure agreement in binary significance classifications (significant vs
non-significant at p < 0.05 threshold) across matched pairs.

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
matching.
Traits are mapped to EFO terms using cosine similarity between their
200-dimensional ScispaCy embeddings.

EFO mappings enable robust cross-study trait matching when researchers use
different terminology for the same biomedical concept.

#### Technical implementation

EFO preprocessing and embedding generation:
- `processing/scripts/main-processing/preprocess-efo.py`: Extracts and
  normalizes EFO terms from ontology files
- `processing/scripts/main-processing/embed-efo.py`: Generates 200-dimensional
  embeddings for EFO terms

Trait-to-EFO mapping is performed during evidence profile computation using
precomputed similarity scores from the vector store database.

Source: https://github.com/EBISPOT/efo/releases

### Vector store

DuckDB database (data/db/vector_store.db) containing trait and EFO embeddings
with precomputed cosine similarity pairs for semantic search.

Contains ~1.7 billion trait-EFO similarity pairs enabling fast nearest-neighbor
queries for trait matching and evidence profile construction.

#### Technical implementation

The vector store is constructed by:
1. `processing/scripts/main-processing/embed-traits.py`: Generates
   200-dimensional embeddings for trait labels
2. `processing/scripts/main-processing/aggregate-embeddings.py`: Aggregates
   embeddings from distributed computation
3. Similarity computation scripts: Precompute cosine similarities between all
   trait-EFO pairs above a threshold

The database schema is documented in docs/processing/db-schema.md.

See docs/processing/databases.md for database schema and docs/DATA.md for
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

- Data structure: docs/DATA.md
- Processing pipeline: docs/processing/pipeline.md
- Case study analyses: docs/processing/case-studies.md
- Database schema: docs/processing/db-schema.md
- Trait profile similarity: docs/processing/trait-profile-similarity.md
- Evidence profile similarity: docs/processing/evidence-profile-similarity.md
