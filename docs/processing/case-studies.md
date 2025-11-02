# Case study analysis pipelines

This document describes the case study analysis pipelines implemented in MR-KG.
Case studies apply the processed databases to answer specific research
questions about reproducibility, network structure, and statistical patterns in
Mendelian Randomization literature.

## Overview

Case study analyses are located in `processing/scripts/analysis/` with
configuration in `processing/config/case_studies.yml`.

Each case study follows a modular pipeline structure:

1. Extract relevant data from databases
2. Compute study-specific metrics
3. Fit statistical models
4. Generate validation examples
5. Produce output artifacts for interpretation

## Case study 1: Reproducibility analysis

Examines consistency of effect directions across independent studies
investigating the same exposure-outcome pairs.

### Research questions

1. What proportion of trait pairs show consistent effect directions across
   studies?
2. How does reproducibility vary by study count, temporal era, and matching
   quality?
3. Are there temporal trends in reproducibility over publication years?
4. Which well-established causal relationships are captured in the data?

### Pipeline stages

#### Stage 1: Extract multi-study pairs

Identifies trait pairs investigated by multiple independent studies.

Command:

```bash
just case-study-cs1-extract-pairs
```

Script:

```bash
uv run scripts/analysis/case_study_1_extract_pairs.py
```

Input:

- `data/db/evidence_profile_db.db`
  - `query_combinations` table: trait pairs with match metadata
  - `mr_results` table: effect sizes, directions, p-values

Configuration:

- `min_study_count`: Minimum studies required per pair (default: 2)

Output:

- `data/processed/case-study-cs1/raw_pairs/multi_study_pairs.csv`
  - Study identifiers: study1_pmid, study1_model, title, publication_year
  - Core metrics: study_count, comparison_count, mean_direction_concordance,
    median/min/max/std direction concordance
  - Match indicators: has_exact_match, has_fuzzy_match, has_efo_match,
    total_matched_pairs
  - JSON column: trait_pairs_json (array of matched trait pairs with match
    types)
- `data/processed/case-study-cs1/raw_pairs/multi_study_pairs_metadata.json`
  - Extraction timestamp, database info, summary statistics

Key metrics computed per pair:

- `study_count`: Number of independent studies (primary metric)
- `comparison_count`: Number of pairwise comparisons (study_count choose 2)
- `mean_direction_concordance`: Average pairwise direction agreement
- `trait_pairs_json`: JSON array of matched trait pairs with match types
- Match indicators: Boolean flags for exact/fuzzy/EFO matches

#### Stage 2: Compute reproducibility metrics

Assigns reproducibility tiers and stratifies by study characteristics.

Command:

```bash
just case-study-cs1-reproducibility-metrics
```

Script:

```bash
uv run scripts/analysis/case_study_1_reproducibility_metrics.py
```

Input:

- `data/processed/case-study-cs1/raw_pairs/multi_study_pairs.csv`

Configuration:

- `reproducibility_tiers`: Concordance thresholds for high/moderate/low
  (default: 0.7, 0.3, 0.0)
- `study_count_bands`: Stratification bins (default: 2-3, 4-6, 7-10, 11+)
- `temporal_eras`: Year ranges for early/recent studies
  (default: 2000-2015, 2016-2025)

Output files (in `data/processed/case-study-cs1/metrics/`):

1. `pair_reproducibility_metrics.csv`
   - Per-pair metrics with tier assignments
   - Inherits all columns from multi_study_pairs.csv
   - Additional columns: reproducibility_tier, study_count_band,
     temporal_era, concordance_variance

2. `tier_distribution.csv`
   - Overall tier counts and percentages

3. `stratified_by_study_count.csv`
   - Tier distribution across study count bands

4. `stratified_by_temporal_era.csv`
   - Tier distribution for early vs recent publications

5. `stratified_by_match_type.csv`
   - Tier distribution by EFO match quality (exact/fuzzy/unmatched)

#### Stage 3: Temporal trend model

Fits linear regression to assess temporal patterns in reproducibility.

Command:

```bash
just case-study-cs1-temporal-model
```

Script:

```bash
uv run scripts/analysis/case_study_1_temporal_model.py
```

Input:

- `data/processed/case-study-cs1/metrics/pair_reproducibility_metrics.csv`

Model specification:

```text
direction_concordance ~ publication_year + study_count + match_type_exact
```

Where:

- `direction_concordance`: Mean pairwise direction agreement (0-1)
- `publication_year`: Latest publication year for the pair
- `study_count`: Number of independent studies
- `match_type_exact`: Boolean indicator for exact EFO matches

Output files (in `data/processed/case-study-cs1/models/`):

1. `temporal_model_coefficients.csv`
   - Regression coefficients with standard errors, t-statistics, p-values
   - 95% confidence intervals

2. `temporal_model_diagnostics.json`
   - R-squared, adjusted R-squared
   - F-statistic and p-value
   - Residual diagnostics (Durbin-Watson, Jarque-Bera, Omnibus)
   - Condition number for multicollinearity

3. `temporal_model_summary.txt`
   - Full regression summary from statsmodels
   - Human-readable format

4. `temporal_predictions.csv`
   - Predicted vs observed values
   - Residuals for diagnostic plots

Interpretation:

The model quantifies how publication year, study count, and matching quality
influence reproducibility. Significant coefficients indicate systematic
patterns:

- Positive year coefficient: reproducibility improving over time
- Negative study count coefficient: dilution effect from more studies
- Positive match type coefficient: exact matches more reproducible

#### Stage 4: Generate validation examples

Creates human-readable briefs for canonical trait pairs and extreme cases.

Command:

```bash
just case-study-cs1-validation-examples
```

Script:

```bash
uv run scripts/analysis/case_study_1_validation_examples.py
```

Input:

- `data/processed/case-study-cs1/metrics/pair_reproducibility_metrics.csv`

Configuration:

- `canonical_pairs`: Known causal relationships to validate
  - Body mass index and type 2 diabetes
  - Smoking and lung cancer
  - LDL cholesterol and coronary artery disease
  - Alcohol consumption and cardiovascular disease
  - Systolic blood pressure and stroke
- `n_top_concordant`: Number of highly concordant examples (default: 20)
- `n_top_discordant`: Number of highly discordant examples (default: 20)

Output directory: `.notes/analysis-notes/case-study-analysis/cs1-validation/`

File types:

1. Canonical pair briefs (e.g., `canonical_bmi_diabetes.md`)
   - Matched if found in data
   - Includes reproducibility metrics and study details

2. Top concordant examples (e.g., `concordant_rank_01.md`)
   - Highest reproducibility pairs
   - Shows consistent replication pattern

3. Top discordant examples (e.g., `discordant_rank_01.md`)
   - Lowest reproducibility pairs
   - Highlights inconsistent findings requiring investigation

Brief structure:

```markdown
# Validation brief: [Trait A] -> [Trait B]

Reproducibility tier: [HIGH/MODERATE/LOW/DISCORDANT]
Direction concordance: [0.XX]
Study count: [N]
Publication year range: [YYYY-YYYY]

## Study details

1. PMID [XXXXXX] (model, YYYY)
   - Effect direction: [positive/negative/null]
   - Match type: [exact/fuzzy/unmatched]

...

## Interpretation

[Concordant]: Strong reproducibility across independent studies
[Discordant]: Inconsistent findings warrant further investigation
```

#### Running complete pipeline

Execute all stages sequentially:

```bash
just case-study-cs1-all
```

This runs:

1. `case-study-cs1-extract-pairs`
2. `case-study-cs1-reproducibility-metrics`
3. `case-study-cs1-temporal-model`
4. `case-study-cs1-validation-examples`

### Key findings

Based on analysis of 2,075 multi-study trait pairs:

#### Reproducibility distribution

- High reproducibility: 60.9% (concordance >= 0.7, n=1,264)
- Moderate reproducibility: 8.4% (concordance 0.3-0.7, n=174)
- Low reproducibility: 10.5% (concordance 0.0-0.3, n=218)
- Discordant: 20.2% (concordance < 0.0, n=419)

#### Temporal trends

Linear regression results:

- R-squared: 0.0325 (3.25% variance explained)
- Study count effect: beta = -0.024 (p < 0.001)
  - More studies associated with lower concordance (dilution effect)
- Match type effect: beta = 0.327 (p < 0.001)
  - Exact EFO matches show 33% higher concordance
- Year effect: Not significant
  - No systematic temporal trend in reproducibility

#### Stratification patterns

Study count:

- 2-3 studies: 1,492 pairs, mean concordance = 0.50 (68.8% high tier)
- 4-6 studies: 437 pairs, mean concordance = 0.44 (43.2% high tier)
- 7-10 studies: 105 pairs, mean concordance = 0.33 (29.5% high tier)
- 11+ studies: 41 pairs, mean concordance = 0.44 (41.5% high tier)
- Pattern: Higher variance in small samples, dilution effect with more studies

Temporal era:

- Early studies (2000-2015): 60 pairs, mean concordance = 0.38 (55.0% high tier)
- Recent studies (2016-2025): 2,014 pairs, mean concordance = 0.48 (61.1% high tier)
- Note: One pair classified as "other" temporal band (100% high tier)

Match type:

- Exact matches: 438 pairs, mean concordance = 0.71 (70.5% high tier)
- Fuzzy matches: 1,920 pairs, mean concordance = 0.44 (58.2% high tier)
- EFO matches: 72 pairs, mean concordance = 0.52 (51.4% high tier)

#### Canonical pair validation

Matched 3 of 5 canonical pairs:

- Body mass index and type 2 diabetes: FOUND
- Smoking and lung cancer: FOUND
- Alcohol consumption and cardiovascular disease: FOUND
- LDL cholesterol and coronary artery disease: NOT FOUND
- Systolic blood pressure and stroke: NOT FOUND

Unmatched pairs may be absent due to:

- Trait name variations not captured by fuzzy matching
- Studies not yet indexed in database
- Filtering criteria (minimum study count)

### Interpretation guidelines

#### High reproducibility pairs

Characteristics:

- Consistent effect directions across studies
- Often exact EFO matches
- May represent well-established causal relationships

Use cases:

- Candidate gold standard pairs for validation
- Training data for causal inference models
- Prioritization for follow-up studies

#### Discordant pairs

Characteristics:

- Inconsistent or opposing effect directions
- May indicate heterogeneity, confounding, or weak effects
- Require careful interpretation

Potential explanations:

1. True heterogeneity across populations or contexts
2. Methodological differences (instruments, adjustments)
3. Trait definition ambiguity despite matching
4. Statistical fluctuation in small studies
5. Publication bias favoring significant results

Next steps:

- Manual review of study methodologies
- Meta-analysis with sensitivity analyses
- Investigation of effect modifiers

### Configuration reference

File: `processing/config/case_studies.yml`

Key parameters:

```yaml
case_study_1:
  min_study_count: 2
  
  reproducibility_tiers:
    high: 0.7
    moderate: 0.3
    low: 0.0
  
  study_count_bands:
    - [2, 3]
    - [4, 6]
    - [7, 10]
    - [11, 999]
  
  temporal_eras:
    early: [2000, 2015]
    recent: [2016, 2025]
  
  validation:
    canonical_pairs:
      - ["body mass index", "type 2 diabetes"]
      - ["smoking", "lung cancer"]
      - ["LDL cholesterol", "coronary artery disease"]
      - ["alcohol consumption", "cardiovascular disease"]
      - ["systolic blood pressure", "stroke"]
    n_top_concordant: 20
    n_top_discordant: 20

databases:
  evidence_profile: "data/db/evidence_profile_db.db"
  trait_profile: "data/db/trait_profile_db.db"
  vector_store: "data/db/vector_store.db"

output:
  case_study_1:
    base: "data/processed/case-study-cs1"
    raw_pairs: "data/processed/case-study-cs1/raw_pairs"
    metrics: "data/processed/case-study-cs1/metrics"
    models: "data/processed/case-study-cs1/models"
    figures: "data/processed/case-study-cs1/figures"
```

### Output directory structure

```text
data/processed/case-study-cs1/
├── raw_pairs/
│   ├── multi_study_pairs.csv
│   └── metadata.json
├── metrics/
│   ├── pair_reproducibility_metrics.csv
│   ├── tier_distribution.csv
│   ├── stratified_by_study_count.csv
│   ├── stratified_by_temporal_era.csv
│   └── stratified_by_match_type.csv
├── models/
│   ├── temporal_model_coefficients.csv
│   ├── temporal_model_diagnostics.json
│   ├── temporal_model_summary.txt
│   └── temporal_predictions.csv
└── figures/
    └── (placeholder for visualizations)

.notes/analysis-notes/case-study-analysis/cs1-validation/
├── canonical_*.md
├── concordant_rank_*.md
└── discordant_rank_*.md
```

### Dependencies

Python packages (managed via `uv`):

- pandas: Data manipulation
- duckdb: Database queries
- pyyaml: Configuration loading
- statsmodels: Linear regression
- scikit-learn: Potential future use
- matplotlib: Potential visualization
- seaborn: Potential visualization

These are added as development dependencies:

```bash
cd processing
uv add --dev statsmodels scikit-learn matplotlib seaborn
```

## Case study 2: Pleiotropic trait networks

Case study 2 builds trait co-occurrence artefacts, community annotated networks, hotspot rankings, and concordance overlays using the configuration in @processing/config/case_studies.yml.
All scripts accept `--dry-run` to verify paths and thresholds, mirror the output directories under `data/processed/case-study-cs2/`, and register recipes in the `processing/justfile`.

### Trait profile preparation

Command: `just case-study-cs2-prepare`.
Script: `processing/scripts/analysis/case_study_2_prepare_profiles.py`.
Inputs: `data/db/trait_profile_db.db`, `data/db/vector_store.db` (read-only).
Outputs (stored in `data/processed/case-study-cs2/cooccurrence/`):
- `trait_profiles.csv` and `trait_profiles.json` summarising trait lists per study
- `trait_pair_metrics.csv` and JSON companion with association statistics
- `trait_frequency.csv` with study counts per trait
- Optional `trait_cooccurrence_matrix.npz` sparse adjacency snapshot
- `trait_cooccurrence_summary.json` capturing thresholds and sample fraction

### Network construction and community detection

Command: `just case-study-cs2-network`.
Script: `processing/scripts/analysis/case_study_2_network_analysis.py`.
Consumes co-occurrence metrics, filters edges using `edge_weight_threshold`, `semantic_similarity_threshold`, and `jaccard_threshold`, and attaches the vector store for EFO annotations.
Outputs (stored in `data/processed/case-study-cs2/network/`):
- `trait_network_nodes.csv` with node metrics and community assignments
- `trait_network_clusters.csv` summarising Louvain clusters (size, density, domain diversity)
- `trait_network.json` for lightweight sharing and `trait_network.graphml` for Gephi or Cytoscape
- `trait_network_metadata.json` containing parameter snapshot, modularity, and file references

### Evidence concordance overlay

Command: `just case-study-cs2-overlay`.
Script: `processing/scripts/analysis/case_study_2_concordance_overlay.py`.
Joins `trait_profile_similarity` with `direction_concordance`, computes correlation statistics, classifies quadrant flags, and renders scatter plots. The script now records both study-level summaries and normalized trait pair rows so downstream components can analyse concordance per trait.
Outputs (stored in `data/processed/case-study-cs2/overlays/` and `figures/`):
- `trait_similarity_concordance.csv` with study pair metrics, quadrant labels, and aggregated trait lists
- `trait_similarity_concordance_pairs.csv` containing one row per matched trait pair (`trait_a`, `trait_b`, match type, direction agreement)
- `trait_similarity_concordance_summary.json` (correlation scores, match type counts, trait pair totals)
- `figures/trait_similarity_vs_concordance.(png|svg)` annotated with threshold lines

### Hotspot profiling and briefs

Command: `just case-study-cs2-hotspots`.
Script: `processing/scripts/analysis/case_study_2_hotspot_profiles.py`.
Ranks hub traits by z-scored degree, strength, betweenness, and eigenvector centralities, evaluates ego network diversity, and writes Markdown briefs. Concordance statistics (`mean`, `std`, counts, positive/negative direction balance) are computed per trait from the normalized overlay file and embedded in the ranking table and briefs.
Outputs (stored in `data/processed/case-study-cs2/hotspots/` and `.notes/analysis-notes/case-study-analysis/cs2-hotspots/`):
- `hotspot_rankings.csv` with hub scores, diversity metrics, concordance aggregates, and cluster context
- `hotspot_summary.json` referencing ranking table and generated notes
- Trait briefs `hotspot-*.md` summarising network metrics, concordance stability, and top neighbours for manual review

### End-to-end workflow

Command: `just case-study-cs2-all` to execute the four scripts sequentially and refresh the network, overlay, and hotspot artefacts.
Document deviations (e.g. sampling fractions, override thresholds) in `.notes/analysis-notes/case-study-analysis/` for reproducibility.

## Future case studies

### Case study 3: Statistical patterns

Planned investigation of effect size distributions and p-value patterns.

## See also

- Evidence profile methodology: @docs/processing/evidence-profile-similarity.md
- Database schema: @docs/processing/db-schema.md
- Processing pipeline: @docs/processing/pipeline.md
