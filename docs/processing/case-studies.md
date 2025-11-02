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

#### Stage 5: Generate figures

Creates publication-ready figures from Case Study 1 analysis outputs.

All figure scripts support the following CLI flags:

- `--config`: Path to case_studies.yml (default: auto-detected)
- `--input-csv`: Override input CSV path
- `--output-dir`: Override output directory
- `--dry-run`: Validate paths without generating figures

Commands:

1. Tier distribution (100% stacked bar chart):
   ```bash
   just case-study-cs1-fig-tier
   ```
   
   Script: `case_study_1_fig_tier_distribution.py`
   
   Input: `tier_distribution.csv`
   
   Output:
   - `tier_distribution.png/svg`: Stacked bar chart showing proportion
     of high/moderate/low/discordant tiers with annotated counts

2. Study count scatter (LOWESS curve):
   ```bash
   just case-study-cs1-fig-scatter
   ```
   
   Script: `case_study_1_fig_study_count_scatter.py`
   
   Input: `pair_reproducibility_metrics.csv`
   
   Output:
   - `study_count_vs_concordance.png/svg`: Scatter plot with LOWESS
     smoothing curve
   - `study_count_correlation.csv`: Spearman correlation statistics
   - Highlights negative association between study count and concordance

3. Match type comparison (stacked bar chart):
   ```bash
   just case-study-cs1-fig-match-type
   ```
   
   Script: `case_study_1_fig_match_type_stacked.py`
   
   Input: `stratified_by_match_type.csv`
   
   Output:
   - `match_type_distribution.png/svg`: Stacked bar chart by match type
   - `match_type_distribution.csv`: Summary statistics per match type
   - Illustrates reproducibility penalty for fuzzy vs exact matches

4. Temporal era comparison (grouped bar chart):
   ```bash
   just case-study-cs1-fig-temporal
   ```
   
   Script: `case_study_1_fig_temporal_era.py`
   
   Input: `stratified_by_temporal_era.csv`
   
   Output:
   - `temporal_era_comparison.png/svg`: Grouped bar chart
   - `temporal_era_comparison.csv`: Era-level statistics
   - Compares tier percentages between early and recent eras

5. Regression diagnostics (multi-panel):
   ```bash
   just case-study-cs1-fig-diagnostics
   ```
   
   Script: `case_study_1_fig_regression_diagnostics.py`
   
   Input:
   - `temporal_predictions.csv`: Fitted values and residuals
   - `temporal_model_diagnostics.json`: Model diagnostics
   
   Output:
   - `regression_diagnostics.png/svg`: Three-panel figure
     - Panel 1: Residuals vs fitted values
     - Panel 2: QQ plot for normality
     - Panel 3: Autocorrelation function (lag 1-20)

6. Concordance variance heatmap:
   ```bash
   just case-study-cs1-fig-heatmap
   ```
   
   Script: `case_study_1_fig_concordance_heatmap.py`
   
   Input: `pair_reproducibility_metrics.csv`
   
   Output:
   - `concordance_variance_heatmap.png/svg`: Heatmap showing variance
     across study count bands and reproducibility tiers
   - `concordance_variance_pivot.csv`: Pivoted data for heatmap

Generate all figures:
```bash
just case-study-cs1-fig-all
```

All figures:
- Export in both PNG (300 DPI) and SVG formats
- Include companion CSV/JSON metadata files
- Use consistent color schemes (high=green, moderate=yellow,
  low=orange, discordant=red)
- Saved to `data/processed/case-study-cs1/figures/`

#### Running complete pipeline

Execute all stages and figures sequentially:

```bash
just case-study-cs1-all
```

This runs:

1. `case-study-cs1-extract-pairs`
2. `case-study-cs1-reproducibility-metrics`
3. `case-study-cs1-temporal-model`
4. `case-study-cs1-validation-examples`
5. `case-study-cs1-fig-all` (all 6 figures)

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
    ├── tier_distribution.png/svg
    ├── study_count_vs_concordance.png/svg
    ├── study_count_correlation.csv
    ├── match_type_distribution.png/svg
    ├── match_type_distribution.csv
    ├── temporal_era_comparison.png/svg
    ├── temporal_era_comparison.csv
    ├── regression_diagnostics.png/svg
    ├── concordance_variance_heatmap.png/svg
    └── concordance_variance_pivot.csv

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

## Case study 5: Temporal trends in MR research

Examines how Mendelian Randomization research has evolved over time across
multiple dimensions: trait diversity, evidence consistency, and reporting
quality.

### Research questions

1. Has trait diversity increased over time as MR expands to new phenotypes?
2. Are methodological eras (MR-Egger, MR-PRESSO, within-family MR) associated
   with trait selection patterns?
3. Has evidence consistency improved in later methodological eras?
4. Has reporting completeness improved over time, particularly after STROBE-MR
   guidelines?

### Pipeline structure

Case Study 5 follows a modular, four-phase approach:

1. **Phase 0 (Temporal preparation)**: Assigns methodological eras to all studies
2. **Phase 1 (Trait diversity)**: Analyzes temporal trends in trait selection
3. **Phase 3 (Evidence consistency)**: Examines concordance patterns across eras
4. **Phase 4 (Reporting completeness)**: Evaluates reporting quality improvements

All scripts use consistent configuration from `processing/config/case_studies.yml`
and share the temporal metadata created in Phase 0.

### Phase 0: Temporal preparation

Creates shared temporal metadata by assigning methodological eras and computing
era-level statistics.

Command:

```bash
just case-study-cs5-temporal-prep
```

Script:

```bash
uv run scripts/analysis/case_study_5_temporal_preparation.py
```

Input:

- `data/db/vector_store.db` (model_results table)
- Configuration: `temporal_eras` from case_studies.yml

Era definitions:

- **early_mr** (2010-2014): Foundation era before sensitivity analyses
- **mr_egger** (2015-2017): Introduction of MR-Egger for pleiotropy
- **mr_presso** (2018-2019): MR-PRESSO for outlier detection
- **within_family** (2020): Within-family designs to address confounding
- **strobe_mr** (2021-2024): STROBE-MR reporting guidelines published

Output (in `data/processed/case-study-cs5/temporal/`):

- `temporal_metadata.csv`: PMID, pub_year, era assignments for all studies
- `era_statistics.csv`: Study counts and year ranges per era
- `temporal_metadata.json`: Summary metadata with era definitions

Key features:

- Studies without valid publication years assigned to "unknown" era
- Era boundaries aligned with major methodological developments
- Metadata shared across all subsequent case study 5 phases

### Phase 1: Trait diversity analysis

Examines temporal trends in trait selection patterns.

Research Question 1: Has trait diversity increased over time as MR expands to
new phenotypes?

Command:

```bash
just case-study-cs5-trait-diversity
```

Script:

```bash
uv run scripts/analysis/case_study_5_trait_diversity.py
```

Input:

- `data/db/vector_store.db` (model_results table)
- `data/processed/case-study-cs5/temporal/temporal_metadata.csv`
- Configuration: temporal_eras, trait_type_categories

Output (in `data/processed/case-study-cs5/diversity/`):

- `trait_counts_by_year.csv`: Unique exposures and outcomes per year
- `trait_counts_by_era.csv`: Aggregated trait statistics per era
- `temporal_trend_model.csv`: Linear regression coefficients testing temporal
  trends
- `era_comparison_tests.csv`: ANOVA results comparing diversity across eras
- `diversity_metadata.json`: Summary statistics and model diagnostics

Figures (in `data/processed/case-study-cs5/figures/`):

- `trait_diversity_over_time.png/svg`: Time series of exposure and outcome counts
- `trait_diversity_by_era.png/svg`: Box plots comparing diversity across eras

Analysis approach:

- Counts unique exposure and outcome traits per study
- Aggregates at yearly and era levels
- Tests temporal trends using linear regression
- Compares eras using ANOVA

Key findings:

- 14,634 studies analyzed (gpt-5 model)
- Mean exposures per study range: 1.86 to 3.36 across eras
- Mean outcomes per study range: 3.13 to 8.18 across eras
- Significant differences between eras (ANOVA p < 0.001)

### Phase 3: Evidence consistency analysis

Examines whether methodological advances have improved the internal consistency
of MR evidence.

Research Question 3: Has evidence consistency improved in later methodological
eras?

Command:

```bash
just case-study-cs5-evidence-consistency
```

Script:

```bash
uv run scripts/analysis/case_study_5_evidence_consistency.py
```

Input:

- `data/db/evidence_profile_db.db` (query_combinations table)
- `data/db/vector_store.db` (model_results table for PMIDs)
- `data/processed/case-study-cs5/temporal/temporal_metadata.csv`
- Configuration: temporal_eras, consistency thresholds

Concordance metric:

Direction concordance is computed from the `query_combinations` table, which
stores matched trait pairs between studies. For each pair:

- Extract `positive_count`, `negative_count`, `null_count` from JSON
- Compute agreement: (max_count - other_counts) / total_count
- Range: -1 (complete disagreement) to +1 (perfect agreement)

Output (in `data/processed/case-study-cs5/consistency/`):

- `concordance_by_year.csv`: Mean concordance and pair counts per year
- `concordance_by_era.csv`: Era-level concordance statistics
- `concordance_by_match_type_era.csv`: Stratified by exact/fuzzy/EFO matches
- `era_comparison_tests.csv`: Statistical tests comparing eras
- `strobe_impact_analysis.csv`: Pre/post 2021 comparisons
- `consistency_metadata.json`: Summary metadata

Figures (in `data/processed/case-study-cs5/figures/`):

- `concordance_over_time.png/svg`: Temporal trends in direction agreement
- `concordance_by_era.png/svg`: Era comparison box plots
- `strobe_impact.png/svg`: Before/after 2021 comparison

Analysis approach:

- Aggregates direction concordance at yearly and era levels
- Tests for significant differences between eras (Kruskal-Wallis test)
- Evaluates STROBE-MR impact using 2021 breakpoint
- Stratifies by match quality (exact/fuzzy/EFO)

Key findings:

- 21,133 trait pairs analyzed across 14,634 studies
- Pre-STROBE-MR concordance: 0.538
- Post-STROBE-MR concordance: 0.531 (no significant change, p=0.069)
- Match type matters: exact matches show higher concordance (0.62) than fuzzy
  (0.51)
- Era differences significant (p < 0.001), but no consistent improvement trend

### Phase 4: Reporting completeness analysis

Evaluates how reporting quality has evolved over time, testing whether
STROBE-MR guidelines improved completeness.

Research Question 4: Has reporting completeness improved over time,
particularly after STROBE-MR?

Command:

```bash
just case-study-cs5-reporting-completeness
```

Script:

```bash
uv run scripts/analysis/case_study_5_reporting_completeness.py
```

Input:

- `data/db/vector_store.db` (model_results table)
- `data/processed/case-study-cs5/temporal/temporal_metadata.csv`
- Configuration: temporal_eras, STROBE-MR breakpoint (2021)

Completeness metrics:

Tracks presence of 7 key fields in extracted results JSON:

- Effect size measures: beta, odds ratio, hazard ratio
- Uncertainty measures: 95% CI, standard error
- Statistical measures: P-value
- Interpretation: direction

For each study, checks if at least one result reports each field (non-null,
non-empty, non-"N/A" values).

Output (in `data/processed/case-study-cs5/completeness/`):

- `field_completeness_by_year.csv`: Field-specific completeness rates per year
- `field_completeness_by_era.csv`: Era-level completeness statistics
- `field_type_by_era.csv`: Completeness by field categories (effect_size,
  statistical, confidence_interval, direction)
- `strobe_impact_on_reporting.csv`: Field-specific chi-square tests for 2021
  breakpoint
- `completeness_metadata.json`: Summary with all statistical tests

Figures (in `data/processed/case-study-cs5/figures/`):

- `completeness_over_time.png/svg`: Temporal trends with STROBE-MR marker
- `strobe_reporting_impact.png/svg`: Before/after comparison bars
- `completeness_by_field_type.png/svg`: Category trends by era

Analysis approach:

- Parses results JSON to check field presence
- Aggregates at yearly and era levels
- Tests STROBE-MR impact using 2021 breakpoint (t-tests and chi-square tests)
- Groups fields into categories for interpretability

Key findings:

- 14,634 studies analyzed (gpt-5 model)
- Overall completeness improved from 40.3% (2015-2020) to 44.2% (2021-2024)
- Change significant: +3.9 percentage points (t=-9.707, p<0.0001)
- Field-specific improvements:
  - Confidence intervals: 89.1% to 96.1% (+6.9 pp, p<0.0001)
  - Odds ratios: 38.4% to 53.7% (+15.3 pp, p<0.0001)
  - P-values: 37.6% to 50.5% (+12.9 pp, p<0.0001)
  - Direction: 81.6% to 89.2% (+7.6 pp, p<0.0001)
- Beta coefficients decreased: 24.1% to 14.4% (-9.7 pp, p<0.0001)
  - Reflects shift toward epidemiological measures (OR/HR) over continuous beta
- All field changes statistically significant (all p < 0.001)

Interpretation:

STROBE-MR guidelines published in 2021 appear to have had a measurable positive
impact on reporting completeness. The most substantial improvements occurred
in confidence intervals, odds ratios, and P-values, suggesting better adoption
of uncertainty reporting and standardized effect measures. The decrease in
beta coefficients reflects a shift in study design toward binary/time-to-event
outcomes rather than incomplete reporting.

### Phase 5: Fashionable traits analysis

Examines temporal trends in trait popularity to identify hype cycles and
sustained research focus.

Research Question 5: Do certain traits experience periods of intense research
activity followed by decline, suggesting fashionability effects in topic
selection?

Command:

```bash
just case-study-cs5-fashionable-traits
```

Script:

```bash
uv run scripts/analysis/case_study_5_fashionable_traits.py
```

Input:

- `data/db/vector_store.db` (model_results table)
- `data/processed/case-study-cs5/temporal/temporal_metadata.csv`
- Configuration: temporal_eras, popularity thresholds

Methodology:

Identifies traits that show:
- Rapid increase in study frequency
- Peak research activity
- Subsequent decline or plateau
- Deviation from overall MR research growth

For each trait (exposure and outcome), computes:
- Study counts per year
- Growth rates and acceleration
- Peak year and magnitude
- Hype cycle indicators (rapid rise and fall)

Output (in `data/processed/case-study-cs5/fashionable/`):

- `trait_popularity_by_year.csv`: Study counts per trait per year
- `top_traits_by_era.csv`: Most frequently studied traits in each era
- `hype_cycle_candidates.csv`: Traits showing fashionability patterns
- `popularity_trends_model.csv`: Statistical tests for trend patterns
- `fashionable_metadata.json`: Summary with trend classifications

Figures (in `data/processed/case-study-cs5/figures/`):

- `trait_popularity_trends.png/svg`: Time series for top traits
- `hype_cycle_examples.png/svg`: Example traits with rise-and-fall patterns
- `era_trait_heatmap.png/svg`: Trait frequency heatmap across eras

Analysis approach:

- Tracks yearly study counts for each unique trait
- Identifies top N traits per era (default: top 20)
- Classifies trend patterns: sustained growth, hype cycle, emerging, declining
- Uses change point detection to identify peak years
- Normalizes for overall research volume growth

Key metrics:

- Peak magnitude: Maximum studies per year for a trait
- Growth rate: Year-over-year percentage change
- Hype score: Composite metric of rise velocity and subsequent decline
- Persistence: Number of consecutive years above threshold

### Phase 6: Pleiotropy awareness analysis

Examines how awareness of pleiotropy and horizontal pleiotropy has evolved,
using canonical pleiotropic pairs and MR-PRESSO adoption as indicators.

Research Question 6: Has awareness of pleiotropy increased over time, as
evidenced by study of canonical pleiotropic pairs and adoption of methods like
MR-PRESSO?

Command:

```bash
just case-study-cs5-pleiotropy-awareness
```

Script:

```bash
uv run scripts/analysis/case_study_5_pleiotropy_awareness.py
```

Input:

- `data/db/vector_store.db` (model_results table)
- `data/db/trait_profile_db.db` (trait_profile_similarity table)
- `data/processed/case-study-cs5/temporal/temporal_metadata.csv`
- Configuration: canonical pleiotropic pairs, MR-PRESSO adoption

Canonical pleiotropic pairs (example):

- Body mass index: Type 2 diabetes, cardiovascular disease, osteoarthritis
- Education: Income, cognitive function, mental health
- C-reactive protein: Cardiovascular disease, diabetes, depression

Methodology:

Tracks three indicators of pleiotropy awareness:

1. Study of canonical pleiotropic exposure-outcome pairs
2. Number of distinct outcomes per study (outcome diversity)
3. MR-PRESSO adoption trends (extracted from methods sections or result types)

For canonical pairs:
- Match exposure-outcome combinations against known pleiotropic relationships
- Track study frequency over time
- Compare awareness across eras

For outcome diversity:
- Count unique outcomes per exposure trait
- Aggregate at yearly and era levels
- Test for increasing diversity trends

Output (in `data/processed/case-study-cs5/pleiotropy/`):

- `canonical_pair_trends.csv`: Study counts for known pleiotropic pairs over time
- `outcomes_per_study_by_era.csv`: Mean outcome count distributions
- `mr_presso_adoption.csv`: Studies using MR-PRESSO methods by year
- `pleiotropy_awareness_tests.csv`: Statistical tests for temporal trends
- `pleiotropy_metadata.json`: Summary with adoption breakpoints

Figures (in `data/processed/case-study-cs5/figures/`):

- `canonical_pairs_over_time.png/svg`: Frequency of pleiotropic pair studies
- `outcomes_per_exposure.png/svg`: Distribution of outcome counts by era
- `mr_presso_adoption_curve.png/svg`: Cumulative adoption with 2018 marker

Analysis approach:

- Identifies canonical pairs using trait matching (exact/fuzzy/EFO)
- Computes outcomes per study as pleiotropy breadth indicator
- Extracts MR-PRESSO usage from methods or result metadata
- Tests for significant changes after 2018 (MR-PRESSO publication)
- Stratifies by methodological era

Key findings indicators:

- Increase in canonical pair studies suggests greater awareness
- Rising outcome diversity indicates broader phenome scanning
- MR-PRESSO adoption marks explicit pleiotropy testing
- Post-2018 breakpoint tests direct methodological impact

### Phase 7: Winner's curse analysis

Investigates whether effect sizes tend to decline in replication studies,
a signature pattern of winner's curse bias.

Research Question 7: Do effect sizes show systematic decline from discovery
to replication, consistent with winner's curse?

Command:

```bash
just case-study-cs5-winners-curse
```

Script:

```bash
uv run scripts/analysis/case_study_5_winners_curse.py
```

Input:

- `data/db/evidence_profile_db.db` (query_combinations, mr_results tables)
- `data/processed/case-study-cs5/temporal/temporal_metadata.csv`
- Configuration: temporal_eras, effect size thresholds

Winner's curse background:

Winner's curse occurs when initial discovery studies overestimate effect sizes
due to selection for statistical significance. Replication studies typically
show smaller effects. Expected decline: 15-25% in meta-analyses.

Methodology:

Two-stage analysis comparing discovery and replication:

Stage 1: Identify multi-study trait pairs
- Extract pairs studied by 2+ independent studies
- Classify earliest study as "discovery", later as "replication"
- Restrict to pairs with consistent effect directions

Stage 2: Compare effect magnitudes
- Extract effect sizes (beta, OR, HR) from results JSON
- Standardize to common scale where possible
- Compute percentage decline from discovery to mean replication
- Test for systematic decline using paired tests

For each pair:
- Discovery effect: Effect from earliest publication year
- Replication effect: Mean effect from subsequent studies
- Decline percentage: (Discovery - Replication) / Discovery * 100
- Significance: Test if decline > 0

Output (in `data/processed/case-study-cs5/winners_curse/`):

- `stage1_multi_study_pairs.csv`: Pairs eligible for analysis with study counts
- `stage2_effect_size_comparison.csv`: Discovery vs replication effects
- `decline_distribution.csv`: Summary statistics of effect size declines
- `temporal_decline_model.csv`: Regression testing era effects on decline
- `winners_curse_metadata.json`: Analysis parameters and findings

Figures (in `data/processed/case-study-cs5/figures/`):

- `discovery_vs_replication.png/svg`: Scatter plot with identity line
- `decline_distribution.png/svg`: Histogram of percentage declines
- `decline_by_era.png/svg`: Box plots showing era differences
- `winners_curse_examples.png/svg`: Top pairs showing strongest decline

Analysis approach:

- Matches trait pairs across studies using evidence profile database
- Orders studies chronologically by publication year
- Standardizes effect sizes to absolute scale for comparisons
- Uses Wilcoxon signed-rank test (non-parametric paired test)
- Stratifies by study count, temporal era, and match quality

Statistical tests:

- Paired t-test: Test if mean decline differs from zero
- Wilcoxon signed-rank: Non-parametric alternative for skewed distributions
- Linear regression: Test if decline varies by publication year, study count
- Subgroup analyses: Stratify by effect size type (beta vs OR vs HR)

Expected patterns:

- Positive decline: Discovery effects larger than replication (winner's curse)
- Decline magnitude: 10-30% typical in epidemiological studies
- Era differences: Earlier eras may show stronger decline
- Study count effect: More replications show stronger evidence

Interpretation:

Systematic positive decline confirms winner's curse presence. The magnitude
indicates severity of initial overestimation. Temporal trends can reveal
whether awareness (e.g., preregistration, reporting standards) has mitigated
the issue in recent publications.

### Running complete pipeline

Execute all phases sequentially:

```bash
just case-study-cs5-all
```

This runs:

1. `case-study-cs5-temporal-prep` (Phase 0)
2. `case-study-cs5-trait-diversity` (Phase 1)
3. `case-study-cs5-evidence-consistency` (Phase 3)
4. `case-study-cs5-reporting-completeness` (Phase 4)
5. `case-study-cs5-fashionable-traits` (Phase 5)
6. `case-study-cs5-pleiotropy-awareness` (Phase 6)
7. `case-study-cs5-winners-curse` (Phase 7)

Estimated runtime: ~10-15 minutes for full pipeline

### Configuration reference

File: `processing/config/case_studies.yml`

Key parameters:

```yaml
case_study_5:
  models_included:
    - "gpt-5"  # Analysis restricted to single model for consistency
  
  temporal_eras:
    early_mr: [2010, 2014]      # Foundation era
    mr_egger: [2015, 2017]      # Pleiotropy methods
    mr_presso: [2018, 2019]     # Outlier detection
    within_family: [2020, 2020] # Family-based designs
    strobe_mr: [2021, 2024]     # Reporting guidelines
  
  strobe_breakpoint: 2021       # STROBE-MR publication year
  
  trait_type_categories:        # For diversity analysis
    anthropometric: ["height", "weight", "BMI", "waist"]
    cardiovascular: ["blood pressure", "heart rate", "cholesterol"]
    metabolic: ["glucose", "insulin", "diabetes"]
    psychiatric: ["depression", "anxiety", "schizophrenia"]
    lifestyle: ["smoking", "alcohol", "physical activity"]

output:
  case_study_5:
    base: "data/processed/case-study-cs5"
    temporal: "data/processed/case-study-cs5/temporal"
    diversity: "data/processed/case-study-cs5/diversity"
    consistency: "data/processed/case-study-cs5/consistency"
    completeness: "data/processed/case-study-cs5/completeness"
    fashionable: "data/processed/case-study-cs5/fashionable"
    pleiotropy: "data/processed/case-study-cs5/pleiotropy"
    winners_curse: "data/processed/case-study-cs5/winners_curse"
    figures: "data/processed/case-study-cs5/figures"
```

### Output directory structure

```text
data/processed/case-study-cs5/
├── temporal/
│   ├── temporal_metadata.csv          # Shared era assignments
│   ├── era_statistics.csv
│   └── temporal_metadata.json
├── diversity/
│   ├── trait_counts_by_year.csv
│   ├── trait_counts_by_era.csv
│   ├── temporal_trend_model.csv
│   ├── era_comparison_tests.csv
│   └── diversity_metadata.json
├── consistency/
│   ├── concordance_by_year.csv
│   ├── concordance_by_era.csv
│   ├── concordance_by_match_type_era.csv
│   ├── era_comparison_tests.csv
│   ├── strobe_impact_analysis.csv
│   └── consistency_metadata.json
├── completeness/
│   ├── field_completeness_by_year.csv
│   ├── field_completeness_by_era.csv
│   ├── field_type_by_era.csv
│   ├── strobe_impact_on_reporting.csv
│   └── completeness_metadata.json
└── figures/
    ├── trait_diversity_over_time.png/svg
    ├── trait_diversity_by_era.png/svg
    ├── concordance_over_time.png/svg
    ├── concordance_by_era.png/svg
    ├── strobe_impact.png/svg
    ├── completeness_over_time.png/svg
    ├── strobe_reporting_impact.png/svg
    └── completeness_by_field_type.png/svg
```

### Dependencies

Python packages (managed via `uv`):

- pandas: Data manipulation and aggregation
- duckdb: Database queries
- pyyaml: Configuration loading
- scipy: Statistical tests (ANOVA, Kruskal-Wallis, t-tests, chi-square)
- statsmodels: Linear regression modeling
- matplotlib: Visualization
- seaborn: Statistical plotting
- numpy: Numerical operations
- loguru: Logging

All dependencies are managed in `processing/pyproject.toml`.

### Methodological notes

#### Era boundary selection

Era boundaries were chosen to align with publication of major methodological
papers:

- 2015: Bowden et al. MR-Egger regression
- 2018: Verbanck et al. MR-PRESSO
- 2020: Brumpton et al. within-family MR
- 2021: Skrivankova et al. STROBE-MR guidelines

#### Model selection

Analysis restricted to gpt-5 model only to ensure consistency across temporal
comparisons. Multi-model analyses would require additional considerations for
model performance differences.

#### Statistical power

Some eras have unbalanced sample sizes:

- within_family (2020): 821 studies (single year)
- strobe_mr (2021-2024): 11,905 studies (73% of dataset)

This imbalance provides high power for STROBE-MR comparisons but lower power
for within_family era analyses.

#### Completeness metric limitations

The completeness analysis tracks field presence, not quality. A reported
P-value may be present but improperly calculated. Future work could examine
statistical coherence (e.g., do reported P-values match computed values from
effect sizes and SEs?).

## Future case studies

### Case study 3: Statistical patterns

Planned investigation of effect size distributions and p-value patterns.

## See also

- Evidence profile methodology: @docs/processing/evidence-profile-similarity.md
- Database schema: @docs/processing/db-schema.md
- Processing pipeline: @docs/processing/pipeline.md
