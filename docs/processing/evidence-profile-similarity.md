# Evidence profile similarity

See @processing/README.md for complete processing pipeline workflow.

## Overview

Evidence profile similarity is a method for comparing studies based on the
statistical evidence patterns observed in their Mendelian Randomization (MR)
analyses.
Unlike trait profile similarity which focuses on which traits are investigated,
evidence profile similarity examines how the exposure-outcome relationships
behave statistically across studies.

Each study (identified by PMID and extraction model) is represented by an
evidence profile: the collection of statistical results for all
exposure-outcome pairs investigated in that study.
By comparing these evidence patterns, we can discover studies with concordant
or discordant findings, identify potential systematic biases, and understand
the consistency of MR evidence across different study contexts.

Studies with similar evidence profiles may:

- Report consistent effect directions across shared trait pairs
- Show similar patterns of statistical significance
- Have comparable effect size magnitudes
- Share methodological characteristics affecting results

By comparing evidence profiles, we can assess the robustness and consistency
of MR findings beyond simple meta-analysis, considering the full pattern of
evidence across multiple trait relationships.

### Conceptual difference from trait similarity

Evidence profile similarity differs fundamentally from trait profile similarity:

- **Trait profile similarity**: Compares what traits are investigated,
  regardless of results (research focus overlap)
- **Evidence profile similarity**: Compares how trait relationships behave
  statistically, requiring matched trait pairs (finding consistency)

Example scenario:

Two studies (A and B) investigate the same set of two exposure-outcome pairs:

- BMI -> Type 2 Diabetes
- BMI -> Coronary Heart Disease

Trait similarity: 100 percent (identical trait profiles)

Evidence similarity: Depends on statistical patterns:

- If both find positive, significant effects: High evidence similarity
- If Study A finds positive effects and Study B finds null effects: Low
  evidence similarity

This distinction enables separate analysis of research coverage (trait
similarity) versus finding reproducibility (evidence similarity).

### Model-specific comparisons

Evidence profiles are computed per extraction model (e.g., gpt-4-1, llama3).
Comparisons are performed only within the same model to ensure consistency in
extraction methodology.
This design choice prevents artifacts from model differences influencing
similarity measures.

## Similarity metrics

We compute several complementary metrics to measure evidence profile similarity,
focusing on those with reliable data availability.

**Primary metrics** (stored in evidence_similarities table):

1. **Direction concordance** (100% availability) - Gold standard metric
2. **Effect size similarity** (~3.82% availability) - Supplementary when available
3. **Statistical consistency** (~0.27% availability) - Exploratory metric
4. **Precision concordance** (~3.33% availability) - Exploratory metric
5. **Composite similarity scores** - Weighted combinations (subject to same
   availability constraints as component metrics)

**Removed metrics** (as of Phase 1-2 cleanup):

- **Evidence overlap** - Removed due to low utility
- **Null concordance** - Removed due to experimental nature

See the "Metric data availability" section under "Limitations and assumptions"
for detailed guidance on metric reliability.

### 1. Effect size similarity

Definition: Pearson correlation of harmonized effect sizes for matched
exposure-outcome pairs.

**Data availability: ~3.82%** due to abstract-only extraction limitations.

Rationale: Quantifies agreement in effect magnitude and direction.
High correlation indicates studies observe similar effect size patterns across
shared trait relationships.

Formula:

Given two studies $A$ and $B$ with $n$ matched exposure-outcome pairs, where $\beta_A^{(i)}$ and $\beta_B^{(i)}$ are the harmonized effect sizes for pair $i$, the effect size similarity is the Pearson correlation coefficient:

$$
r_{\text{effect}}(A, B) = \frac{\sum_{i=1}^{n} (\beta_A^{(i)} - \bar{\beta}_A)(\beta_B^{(i)} - \bar{\beta}_B)}{\sqrt{\sum_{i=1}^{n} (\beta_A^{(i)} - \bar{\beta}_A)^2} \sqrt{\sum_{i=1}^{n} (\beta_B^{(i)} - \bar{\beta}_B)^2}}
$$

where:
- $\bar{\beta}_A$ and $\bar{\beta}_B$ are the mean harmonized effect sizes across matched pairs
- Harmonization transforms OR and HR to log scale: $\beta = \log(\text{OR})$ or $\beta = \log(\text{HR})$
- Beta coefficients are kept unchanged

Computation:

1. Identify exposure-outcome pairs present in both studies
2. Harmonize all effect sizes to beta (log scale) equivalents:
   - OR and HR: Apply log transformation (log(OR), log(HR))
   - Beta coefficients: Keep as-is
3. Compute Pearson correlation of effect sizes across matched pairs

Range: -1 to 1

- 1: Perfect positive correlation (effect sizes align perfectly)
- 0: No correlation (effect sizes are independent)
- -1: Perfect negative correlation (effect sizes systematically opposite)

Edge cases:

- Requires minimum 3 matched pairs for computation
- Requires at least 2 unique values in each study's effect sizes
- Returns None if insufficient data or variance

Stratified analysis:

- Within-type similarity: Correlation for pairs with same effect type
  (beta-beta, OR-OR, HR-HR) - more reliable
- Cross-type similarity: Correlation for pairs with different effect types
  - less reliable due to harmonization assumptions

### 2. Direction concordance

Definition: Agreement in classified effect directions (positive, negative,
null) for matched pairs.

**Data availability: 100%** - always computable for any matched trait pairs.
This is the gold standard metric for evidence profile similarity.

Rationale: Provides categorical assessment of effect direction consistency
without sensitivity to exact effect size magnitudes.

Formula:

Given two studies $A$ and $B$ with $n$ matched exposure-outcome pairs, where $d_A^{(i)}, d_B^{(i)} \in \{-1, 0, +1\}$ represent the classified directions (negative, null, positive), the direction concordance is:

$$
C_{\text{direction}}(A, B) = \frac{n_{\text{concordant}} - n_{\text{discordant}}}{n_{\text{concordant}} + n_{\text{discordant}}}
$$

where:
- $n_{\text{concordant}}$ = number of pairs where both have same non-zero direction
- $n_{\text{discordant}}$ = number of pairs where directions are opposite (one positive, one negative)
- Pairs involving null directions (where either $d_A^{(i)} = 0$ or $d_B^{(i)} = 0$) are excluded from the calculation

Direction classification:
- Positive: harmonized beta > 0 (direction = +1)
- Negative: harmonized beta < 0 (direction = -1)
- Null: beta = 0 or missing (direction = 0)

Computation:

1. Classify each effect as positive, negative, or null based on harmonized
   beta values
2. For each matched pair, score agreement:
   - Both same direction (positive-positive or negative-negative): concordant
   - Opposite directions (positive-negative or negative-positive): discordant
   - One or both null: excluded from calculation
3. Compute concordance score as (concordant - discordant) / (concordant + discordant)

Range: -1 to 1

- 1: Perfect concordance (all effects agree in direction)
- 0: Equal concordant and discordant pairs
- -1: Perfect discordance (all effects opposite)

Edge cases:

- If no matched pairs: returns 0.0
- If all pairs involve null directions: returns 0.0 (no directional comparisons possible)

### 3. Statistical consistency

Definition: Cohen's kappa coefficient for agreement in statistical
significance classifications.

**Data availability: ~0.27%** - extremely limited due to matching sparsity.
Requires minimum 3 matched pairs, but 82% of comparisons have only 1 matched
pair. Consider this an exploratory metric that rarely succeeds.

Rationale: Measures consistency in which trait relationships reach statistical
significance, accounting for chance agreement.

Formula:

Given two studies $A$ and $B$ with $n$ matched exposure-outcome pairs, where each pair is classified as significant ($s=1$) or non-significant ($s=0$) based on $p < 0.05$ threshold, Cohen's kappa is:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where:
- $p_o$ = observed agreement proportion = $\frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[s_A^{(i)} = s_B^{(i)}]$
- $p_e$ = expected agreement by chance = $p_{\text{sig},A} \cdot p_{\text{sig},B} + p_{\text{nonsig},A} \cdot p_{\text{nonsig},B}$
- $p_{\text{sig},A}$ = proportion of significant pairs in study A
- $p_{\text{nonsig},A}$ = proportion of non-significant pairs in study A (and similarly for study B)
- $\mathbb{1}[\cdot]$ = indicator function (1 if true, 0 if false)

Computation:

1. Classify each effect as significant or non-significant (p < 0.05 threshold)
2. Compute observed agreement proportion across matched pairs
3. Compute expected agreement proportion by chance
4. Calculate kappa = (observed - expected) / (1 - expected)

Range: -1 to 1

- 1: Perfect consistency (significance patterns identical)
- 0: Agreement equivalent to chance
- -1: Systematic disagreement (worse than chance)

Interpretation (Landis & Koch scale):

- < 0: Poor (less than chance)
- 0.0 to 0.20: Slight
- 0.21 to 0.40: Fair
- 0.41 to 0.60: Moderate
- 0.61 to 0.80: Substantial
- 0.81 to 1.00: Almost perfect

Edge cases:

- Requires minimum 3 matched pairs for computation
- Returns None if insufficient data or no variance in classifications
- Returns 0.0 if expected agreement = 1.0 (all pairs have same classification)

### 4. Precision concordance

Definition: Spearman correlation of confidence interval widths for matched
exposure-outcome pairs.

**Data availability: ~3.33%** - very limited due to the combination of CI
extraction limitations and matching sparsity. Requires both confidence intervals
from abstract extraction AND at least 3 matched pairs for correlation.

Rationale: Measures how similar two studies are in terms of the precision of
their effect estimates.

Formula:

Given two studies $A$ and $B$ with $n$ matched exposure-outcome pairs, where $w_A^{(i)} = \text{CI}_{\text{upper}}^{(i)} - \text{CI}_{\text{lower}}^{(i)}$ represents the confidence interval width for pair $i$, the precision concordance is the Spearman rank correlation coefficient of log-transformed widths:

$$
\rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
$$

where:
- $d_i$ = difference in ranks between $\log(w_A^{(i)})$ and $\log(w_B^{(i)})$
- Log transformation applied to handle skewed distribution of CI widths
- Spearman correlation used instead of Pearson to capture monotonic (not just linear) relationships

Computation:

1. For each matched pair, compute CI width as |upper - lower|
2. Log-transform widths to handle skewed distributions: $w' = \log(w)$
3. Rank the log-transformed widths separately for each study
4. Compute Spearman correlation of ranks across matched pairs

Range: -1 to 1

- 1: Perfect positive correlation (precision patterns align)
- 0: No correlation (precision patterns independent)
- -1: Perfect negative correlation (inverse precision patterns)

Edge cases:

- Requires minimum 3 matched pairs with valid CIs for computation
- Returns None if insufficient data
- Requires at least 2 unique CI width values in each study

### 5. Evidence overlap (REMOVED)

**Status: Removed in Phase 1-2 cleanup**

This metric (Jaccard similarity of significant finding sets) was removed due
to low utility and correlation with other metrics.

Previous definition: Jaccard similarity of sets of statistically significant
exposure-outcome pairs.

Removal rationale:

- Redundant with direction concordance and statistical consistency
- Added complexity without additional insight
- Set-based approach less informative than pairwise concordance

### 6. Null concordance (REMOVED)

**Status: Removed in Phase 1-2 cleanup**

This metric (proportion of matched pairs with mutual non-significance) was
removed due to its experimental nature and unclear interpretation.

Previous definition: Proportion of matched pairs where both results are
non-significant (p >= 0.05).

Removal rationale:

- Ambiguous interpretation (power limitations vs. true nulls)
- Better addressed through sample size and CI width analysis
- Limited actionable insights for similarity assessment

### 7. Composite similarity scores

Definition: Weighted combinations of individual metrics into summary scores.

**Data availability: Limited** - composite scores inherit the availability
constraints of their component metrics. Due to the low availability of
effect_size_similarity, statistical_consistency, and precision_concordance,
composite scores will frequently be incomplete or NULL.

**Recommendation**: Given the data availability issues, prioritize
direction_concordance (100% availability) as the primary similarity metric
rather than relying on composite scores.

Rationale: Provide single interpretable values balancing multiple evidence
dimensions for ranking and filtering.

Implementation: Two composite scores with different weighting schemes:

**Equal-weighted composite:**

Formula:

$$
S_{\text{equal}} = Q \cdot \frac{1}{k} \sum_{m \in M} N(m)
$$

where:
- $M$ = set of available metrics (non-NULL values)
- $k$ = number of available metrics = $|M|$
- $N(m)$ = normalized metric value for metric $m$ (transformed to [0, 1] scale)
- $Q$ = quality weight = $\min(C_A, C_B)$ where $C_A$, $C_B$ are data completeness scores for studies A and B

Properties:
- Average of all available normalized metrics
- Each metric contributes equally to final score
- Balances all aspects of evidence similarity
- Frequently NULL due to missing component metrics

**Direction-prioritized composite:**

Formula:

$$
S_{\text{direction}} = Q \cdot \frac{\sum_{m \in M} w_m \cdot N(m)}{\sum_{m \in M} w_m}
$$

where:
- $w_m$ = predefined weight for metric $m$:
  - Direction concordance: $w = 0.50$
  - Effect size similarity: $w = 0.20$
  - Statistical consistency: $w = 0.15$
  - Precision concordance: $w = 0.15$
- Weights renormalized across available metrics: $\sum_{m \in M} w_m$ in denominator
- $Q$ = quality weight (same as equal-weighted composite)

Properties:
- Weights: 0.50 × direction + 0.20 × effect_size + 0.15 × consistency + 0.15
  × precision
- Emphasizes directional concordance as primary indicator
- More likely to be computable due to 50% weight on always-available
  direction_concordance
- Recommended over equal-weighted when both are available

Normalization procedure:

All metrics are normalized to [0, 1] scale before combination:

- Effect similarity: $(r + 1) / 2$ transforms $[-1, 1] \to [0, 1]$
- Direction concordance: $(c + 1) / 2$ transforms $[-1, 1] \to [0, 1]$
- Statistical consistency: $(\kappa + 1) / 2$ transforms $[-1, 1] \to [0, 1]$
- Precision concordance: $(\rho + 1) / 2$ transforms $[-1, 1] \to [0, 1]$

Missing data handling:

- If component metrics are None, they are excluded from the composite calculation
- Weights are renormalized across available metrics (denominator in formula)
- Requires minimum 2 non-null metrics (direction_concordance is always available)
- Returns None if insufficient metrics

**Note**: The removal of evidence_overlap and null_concordance metrics in Phase
1-2 cleanup means composite scores now rely on fewer component metrics than
originally designed.

Quality weighting:

- Composite scores are multiplied by $Q = \min(C_A, C_B)$ where $C_A$ and $C_B$ are data completeness scores
- Down-weights comparisons involving studies with incomplete data
- Accounts for data quality differences across studies

Range: 0 to 1 (or None if insufficient data)

- 1: Maximum similarity across all available dimensions with perfect data quality
- 0: No similarity or very low data quality
- Scores reduced proportionally to worst data completeness between study pair

## Workflow

The evidence similarity computation follows these steps:

```mermaid
flowchart TD
    A[Load evidence profiles] --> B[Preprocess and harmonize]
    B --> C[Group by extraction model]
    C --> D[For each query study]
    D --> E{Filter to same model}
    E --> F[Match exposure-outcome pairs]
    F --> G{Sufficient matched pairs?}
    G -->|Yes| H[Compute effect size correlation]
    G -->|No| I[Skip pair]
    H --> J[Compute direction concordance]
    J --> K[Compute statistical consistency]
    K --> L[Compute evidence overlap]
    L --> M[Store results]
    M --> N{More queries?}
    N -->|Yes| D
    N -->|No| O[Aggregate results]
    O --> P[Build evidence profile database]
    I --> N
```

### Data preparation

1. Load evidence profiles from model results
2. Each profile contains:
   - Study identifier (PMID)
   - Extraction model used
   - List of exposure-outcome pairs with:
     - Trait indices (exposure_trait_idx, outcome_trait_idx)
     - Effect size (beta or OR or HR)
     - Standard error
     - P-value
     - Sample size (if available)

### Harmonization

Before comparison, statistical measures are harmonized:

1. **Effect size harmonization:**

   - Convert OR (odds ratio) to log scale: beta = log(OR)
   - Convert HR (hazard ratio) to log scale: beta = log(HR)
   - Keep beta values unchanged
   - Handle invalid values (OR/HR ≤ 0) as missing

2. **Direction classification:**

   - Positive: harmonized beta > 0
   - Negative: harmonized beta < 0
   - Null: beta = 0, missing, or invalid

3. **Significance classification:**

   - Significant: p < 0.05
   - Non-significant: p ≥ 0.05 or missing

### Pairwise comparison

For each query PMID-model combination:

1. Filter comparison candidates to same model only
2. For each candidate (excluding self):
   - Match exposure-outcome pairs by trait indices
   - Skip if fewer than 2 matched pairs (insufficient for correlation)
   - Compute all four similarity metrics on matched pairs
   - Calculate summary statistics:
     - Total exposure-outcome pairs in each study
     - Number of matched pairs
     - Matched pair fraction (matched / min(query_pairs, candidate_pairs))
3. Store all results (no filtering by similarity score)

### Parallel processing

The workload is distributed using SLURM job arrays:

- Total combinations divided into N chunks
- Each job processes one chunk independently
- Within each job, multiprocessing parallelizes query processing
- Results saved as separate JSON files per chunk

### Output structure

Each query-candidate comparison produces a record containing:

**Query metadata:**

- PMID and extraction model
- Number of exposure-outcome pairs in profile

**Candidate metadata:**

- PMID and extraction model
- Number of exposure-outcome pairs in profile

**Matching information:**

- Number of matched exposure-outcome pairs
- Matched pair fraction

**Similarity metrics:**

- Effect size similarity (Pearson correlation, null if insufficient pairs)
- Direction concordance (agreement score)
- Statistical consistency (Cohen's kappa, null if no variance)
- Evidence overlap (Jaccard similarity of significant pairs)

## Analysis workflows

The evidence profile database supports four complementary analysis workflows:

### 1. Summary statistics analysis

**Purpose:** Understand overall distribution and characteristics of evidence
similarities.

**Key outputs:**

- Model-level statistics (mean, median, std for each metric)
- Percentile distributions showing metric ranges
- Metric correlations revealing relationships between measures

**Use cases:**

- Assess typical similarity levels in the dataset
- Identify whether metrics capture different aspects (low correlation) or
  redundant information (high correlation)
- Establish baseline expectations for similarity values

### 2. Trait comparison analysis

**Purpose:** Compare evidence-based similarities with trait-based similarities
to understand relationship between research focus and finding consistency.

**Key outputs:**

- Correlations between trait and evidence metrics by model
- Quadrant classification:
  - High trait, high evidence: Studies investigating similar traits with
    consistent findings
  - High trait, low evidence: Studies investigating similar traits with
    inconsistent findings (interesting for heterogeneity)
  - Low trait, high evidence: Studies investigating different traits with
    similar evidence patterns
  - Low trait, low evidence: Unrelated studies
- Top discordant cases for detailed examination

**Use cases:**

- Identify studies with high trait overlap but divergent findings (potential
  heterogeneity sources)
- Discover methodological factors affecting finding reproducibility
- Assess whether similar trait profiles predict similar evidence patterns

### 3. Data quality analysis

**Purpose:** Assess completeness and distribution of evidence data.

**Key outputs:**

- Overall data quality metrics (total pairs, matched pairs distribution)
- Field completeness by model (percentage of records with each field)
- Distribution of matched exposure-outcome pairs per study pair

**Use cases:**

- Identify models with sparse evidence profiles
- Assess whether matching requirements filter too many comparisons
- Validate data extraction quality by field completeness

### 4. Validation analysis

**Purpose:** Validate similarity computation through examination of extreme
cases.

**Key outputs:**

- Summary statistics for top similar pairs
- Detailed records of highest similarity study pairs
- Discordant pairs (conflicting direction and significance patterns)

**Use cases:**

- Verify that high similarity pairs are truly similar through manual
  inspection
- Identify potential computation errors or edge cases
- Find interesting biological examples of consistent or contradictory evidence

### 5. Match type quality stratification

**Purpose:** Analyze whether match quality varies by trait matching strategy
(exact, fuzzy, EFO).

**Script:** `scripts/analysis/analyze-match-type-quality.py`

**Key outputs:**

- Quality metrics stratified by predominant match type
- Direction concordance, effect size similarity, statistical consistency by
  match type
- Match type distribution across similarity quartiles
- Data completeness analysis by match type

**Use cases:**

- Validate that exact matches produce higher quality similarities than fuzzy
  or EFO matches
- Identify whether EFO category-level matching introduces noise
- Assess trade-off between match rate and match quality for fuzzy matching
- Inform decisions about fuzzy matching thresholds

**Note:** This analysis operates on intermediate chunk files from HPC jobs
(before aggregation into the database).

**Command reference:**

Just recipe:

```bash
just analyze-match-type-quality
```

Python script:

```bash
cd processing
uv run scripts/analysis/analyze-match-type-quality.py \
  --input-dir ../data/output/<experiment-id>/results
```

The script expects chunk files at the specified input directory with pattern
`evidence_similarities_chunk_*.json`.

### 6. Alternative metrics analysis

**Purpose:** Assess feasibility of alternative similarity metrics beyond the
core six metrics.

**Script:** `scripts/evidence-profile/analyze-alternative-metrics.py`

**Key outputs:**

- Evaluation of alternative metric formulations
- Comparison with existing metrics
- Feasibility assessment for implementation

**Use cases:**

- Explore whether alternative metrics capture additional aspects of evidence
  similarity
- Validate that current metrics provide comprehensive coverage
- Inform future metric development decisions

**Command reference:**

Just recipe:

```bash
just analyze-alternative-metrics
```

Python script:

```bash
cd processing
uv run scripts/evidence-profile/analyze-alternative-metrics.py \
  --evidence-db ../data/db/evidence_profile_db.db \
  --output-dir ../data/processed/evidence-profiles/analysis
```

### 7. EFO matching failure analysis

**Purpose:** Investigate EFO matching performance and threshold sensitivity to
understand why trait matching may fail.

**Script:** `scripts/evidence-profile/analyze-efo-matching-failure.py`

**Key outputs:**

- Analysis of EFO matching success rates
- Threshold sensitivity evaluation
- Common failure patterns identification

**Use cases:**

- Identify systematic issues in EFO-based trait matching
- Optimize matching thresholds for better coverage
- Understand trade-offs between precision and recall in trait matching

**Note:** This analysis operates on intermediate chunk files from HPC jobs.

**Command reference:**

Just recipe:

```bash
just analyze-efo-matching-failure
```

Python script:

```bash
cd processing
uv run scripts/evidence-profile/analyze-efo-matching-failure.py \
  --batch-output-dir ../data/output/<experiment-id>/results
```

The script expects chunk files at the specified batch output directory.

## Database schema

The evidence profile database contains a single table:

### evidence_similarities

**Description:** Pairwise evidence profile similarity scores between studies.

**Columns:**

- `query_pmid` (INTEGER): PMID of query study
- `query_model` (VARCHAR): Extraction model for query study
- `query_pairs` (INTEGER): Number of exposure-outcome pairs in query study
- `similar_pmid` (INTEGER): PMID of candidate study
- `similar_model` (VARCHAR): Extraction model for candidate study
- `similar_pairs` (INTEGER): Number of exposure-outcome pairs in candidate
  study
- `matched_pairs` (INTEGER): Number of exposure-outcome pairs matched between
  studies
- `matched_fraction` (DOUBLE): Fraction of pairs matched relative to smaller
  study
- `effect_size_similarity` (DOUBLE): Pearson correlation of harmonized effect
  sizes (null if insufficient data)
- `direction_concordance` (DOUBLE): Agreement in effect directions
  (-1 to 1 scale)
- `statistical_consistency` (DOUBLE): Cohen's kappa for significance agreement
  (null if no variance)
- `evidence_overlap` (DOUBLE): Jaccard similarity of significant pairs

**Indexes:**

- Primary key: (query_pmid, query_model, similar_pmid, similar_model)
- Index on query_pmid for efficient query lookup
- Index on similar_pmid for reverse lookup
- Index on query_model for model-specific queries

**Constraints:**

- Query and similar records must have same extraction model
- matched_pairs ≤ min(query_pairs, similar_pairs)
- matched_fraction between 0 and 1
- All similarity metrics between -1 and 1 (except where null allowed)

## Usage examples

### Query similar evidence profiles

Find studies with similar evidence patterns to a specific study:

```sql
SELECT
    similar_pmid,
    similar_pairs,
    matched_pairs,
    effect_size_similarity,
    direction_concordance,
    statistical_consistency,
    evidence_overlap
FROM evidence_similarities
WHERE query_pmid = 12345678
  AND query_model = 'gpt-4-1'
ORDER BY effect_size_similarity DESC NULLS LAST
LIMIT 10;
```

### Find highly concordant studies

Identify study pairs with strong agreement across all metrics:

```sql
SELECT
    query_pmid,
    similar_pmid,
    matched_pairs,
    effect_size_similarity,
    direction_concordance,
    statistical_consistency,
    evidence_overlap
FROM evidence_similarities
WHERE query_model = 'gpt-4-1'
  AND matched_pairs >= 5
  AND effect_size_similarity > 0.7
  AND direction_concordance > 0.7
  AND statistical_consistency > 0.6
  AND evidence_overlap > 0.5
ORDER BY effect_size_similarity DESC;
```

### Identify discordant evidence patterns

Find study pairs investigating similar traits but with opposite findings:

```sql
-- Requires joining with trait_profile_db.db
-- Conceptual query (requires ATTACH DATABASE)
SELECT
    e.query_pmid,
    e.similar_pmid,
    e.matched_pairs,
    e.effect_size_similarity,
    e.direction_concordance,
    t.semantic_similarity as trait_semantic_similarity,
    t.jaccard_similarity as trait_jaccard_similarity
FROM evidence_similarities e
JOIN trait_similarities t
    ON e.query_pmid = t.query_pmid
    AND e.query_model = t.query_model
    AND e.similar_pmid = t.similar_pmid
    AND e.similar_model = t.similar_model
WHERE e.query_model = 'gpt-4-1'
  AND t.semantic_similarity > 0.7  -- High trait similarity
  AND e.direction_concordance < -0.3  -- Opposite effect directions
  AND e.matched_pairs >= 3
ORDER BY t.semantic_similarity DESC;
```

### Model comparison

Compare evidence similarity distributions across extraction models:

```sql
SELECT
    query_model,
    COUNT(*) as n_comparisons,
    AVG(matched_pairs) as avg_matched_pairs,
    AVG(effect_size_similarity) as avg_effect_corr,
    AVG(direction_concordance) as avg_direction,
    AVG(statistical_consistency) as avg_kappa,
    AVG(evidence_overlap) as avg_overlap
FROM evidence_similarities
GROUP BY query_model
ORDER BY query_model;
```

## Limitations and assumptions

### Metric data availability

**Critical constraint:** Due to matching sparsity and abstract-only
extraction limitations, several metrics have very low data availability and
should be interpreted accordingly.

**Data availability by metric:**

| Metric | Availability | Reliability |
|--------|-------------|-------------|
| Direction concordance | 100% | Gold standard - always computable |
| Effect size similarity | ~3.82% | Limited - requires abstract effect extraction |
| Precision concordance | ~3.33% | Very limited - requires CIs + 3+ pairs |
| Statistical consistency | ~0.27% | Extremely limited - requires 3+ pairs |

**Root causes:**

1. **Matching sparsity:** 82% of study pairs share only 1 matched trait pair.
   Correlation-based metrics (statistical consistency, precision concordance)
   require minimum 3 pairs, making them inapplicable in most cases.

2. **Abstract extraction limits:** Effect sizes and confidence intervals are
   extracted from abstracts only, not full text. Abstract reporting is
   incomplete, especially for negative/null results.

**Recommendations:**

- **Prioritize direction concordance** (100% availability) as the primary
  metric for evidence profile similarity
- Use effect size similarity (3.82%) as supplementary when available
- Treat precision concordance (3.33%) and statistical consistency (0.27%) as
  exploratory metrics that rarely succeed
- Consider matched_pairs count when interpreting similarity scores - pairs
  with only 1-2 matched traits provide limited evidence overlap

**Database implications:**

The `evidence_profile_similarities` table includes a `metric_availability`
view showing actual data availability percentages. Most records will have
NULL values for low-availability metrics, which is expected behavior, not a
data quality issue.

### Within-study correlation

**Limitation:** Results within the same study are non-independent due to
shared methodological features, instrumental variables, and populations.

**Impact:** Confidence intervals and significance tests for similarity scores
may be overstated since the independence assumption is violated.

**Current approach:** Phase 1 implementation treats results as independent and
computes mean similarity across matched pairs without adjustment for
correlation structure.

**Future enhancement (Phase 2):** Implement bootstrap confidence intervals
that resample entire studies (not individual results) to properly account for
within-study correlation when estimating uncertainty in similarity scores.

**Practical note:** This primarily affects uncertainty quantification, not
point estimates. The similarity metrics themselves remain valid descriptive
measures of evidence patterns.

### Cross-effect-type comparisons

**Assumption:** Harmonizing OR, HR, and beta to log scale enables meaningful
cross-type comparisons.

**Limitation:** Log(OR) and log(HR) are not strictly equivalent to beta
coefficients due to different underlying outcome scales and measurement units.

**Current approach:**

- Primary metric: effect_size_similarity combines all effect types
- Match type tracking: match_type_exact, match_type_fuzzy, match_type_efo
  boolean flags indicate matching method for each comparison

**Note on Phase 1-2 changes:** Previously tracked stratified metrics
(effect_size_within_type, effect_size_cross_type, n_within_type_pairs,
n_cross_type_pairs) were removed to simplify the schema. The overall
effect_size_similarity metric remains but no longer separates within-type vs
cross-type correlations.

**Recommendation:** Interpret effect size correlations cautiously when studies
use different effect types. The match_type flags can help identify comparison
quality but do not separate by effect type harmonization.

### Temporal and population heterogeneity

**Limitation:** Studies from different time periods or populations may have
systematically different effect sizes even for the same causal relationships.

**Current approach:** All studies compared regardless of publication year or
population. Publication year extracted and stored for post-hoc stratification.

**Future enhancement:** Add publication year difference penalty or stratify
analyses by time period if temporal effects are detected.

### Missing data and power

**Limitation:** Studies with high null concordance may reflect shared power
limitations rather than truly null relationships.

**Current approach:** null_concordance metric reported separately to flag
potential power issues. Quality weighting down-weights studies with incomplete
data.

**Recommendation:** Interpret high null concordance in context of study sample
sizes and precision. High null_concordance + wide confidence intervals suggests
power issues rather than evidence consistency.

## Glossary

- **Evidence profile:** The collection of statistical results (effect sizes,
  p-values, directions) for all exposure-outcome pairs investigated in a study,
  representing the complete pattern of observed statistical evidence
  (identified uniquely by PMID and extraction model)
- **Evidence profile similarity:** A multi-metric quantitative assessment of
  statistical evidence alignment between two studies, computed across matched
  exposure-outcome pairs using effect size correlation, direction concordance,
  significance consistency, and evidence overlap measures
- **Harmonized effect size:** Effect size converted to beta (log) scale to
  enable comparison across different effect measures (beta, OR, HR)
- **Matched pairs:** Exposure-outcome trait combinations present in both
  studies being compared, identified by matching trait indices
- **Direction concordance:** Agreement in classified effect directions
  (positive, negative, null) across matched pairs
- **Statistical consistency:** Cohen's kappa measuring agreement in
  statistical significance classifications (p < 0.05 threshold)
- **Evidence overlap:** Jaccard similarity of sets of statistically
  significant exposure-outcome pairs between studies
- **Null concordance:** Proportion of matched pairs where both results are
  non-significant, indicating shared null findings
- **Within-type similarity:** Effect size correlation computed only for
  matched pairs with identical effect types (beta-beta, OR-OR, HR-HR)
- **Cross-type similarity:** Effect size correlation computed for matched
  pairs with different effect types after harmonization
- **Quality weighting:** Adjustment of composite scores by multiplying by
  min(query_completeness, similar_completeness) to account for data quality
  differences
