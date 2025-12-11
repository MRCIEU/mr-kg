# Manuscript tables

This document describes the manuscript-ready tables generated for publication,
including their structure, data sources, and interpretation guidelines.

## Overview

Manuscript tables are stored in: `data/artifacts/manuscript-tables/`

These tables are optimized for publication with:
- CSV format for data portability and inspection
- LaTeX format for direct manuscript inclusion
- Consistent formatting and nomenclature
- Clear column headers and row labels
- Metadata JSON files documenting generation parameters

## Case Study 1: Reproducibility tables

**Generation script**:
`processing/scripts/analysis/consolidate_cs1_manuscript_tables.py`

**Just command**: `just case-study-cs1-manuscript-tables`

**Metadata file**: `cs1_tables_metadata.json`

### Table 1: Reproducibility tier distribution

**Files**:
- `cs1_tier_distribution.csv` - Data in CSV format
- `cs1_tier_distribution.tex` - LaTeX formatted table

**Purpose**: Summarize overall distribution of trait pairs across
reproducibility tiers.

**Data source**: `data/processed/case-study-cs1/metrics/tier_distribution.csv`

**Structure**:

| Column | Description |
|--------|-------------|
| Reproducibility Tier | Classification: High, Moderate, Low, Discordant |
| Pairs (n) | Number of trait pairs in each tier |
| Percentage (%) | Percentage of total pairs (to 1 decimal place) |

**Reproducibility tier definitions**:

- **High**: Direction concordance >= 0.7 (70% of studies agree)
- **Moderate**: Direction concordance 0.5-0.69 (50-69% agreement)
- **Low**: Direction concordance 0.0-0.49 (below 50% but not opposing)
- **Discordant**: Direction concordance < 0.0 (opposite directions)

**Key features**:
- Includes a "Total" row with sum of all pairs
- Percentages sum to exactly 100.0%
- Tiers ordered from highest to lowest reproducibility quality

**Interpretation**: Shows that the majority (60.9%) of trait pairs achieve
high reproducibility, while a notable fraction (20.2%) show discordant
results, highlighting both the reliability and challenges in MR literature.

### Table 2: Concordance statistics by match type and category

**Files**:
- `cs1_match_type_by_category.csv` - Data in CSV format
- `cs1_match_type_by_category.tex` - LaTeX formatted table

**Purpose**: Present concordance distribution statistics for both overall
(subplot C) and category-specific (subplot D) analyses, showing how match
quality affects concordance distributions.

**Data source**:
`data/processed/case-study-cs1/metrics/pair_reproducibility_metrics.csv`

**Structure**:

| Column | Description |
|--------|-------------|
| Outcome Category | "All" for overall, or specific category name |
| Match Type | Exact or Fuzzy |
| N | Number of trait pairs |
| Mean | Mean direction concordance |
| Median | Median direction concordance |
| SD | Standard deviation of concordance |

**Match type definitions**:

- **Exact**: Identical trait indices across studies (highest confidence)
- **Fuzzy**: Similar traits matched via embedding-based semantic similarity
  (moderate confidence)

**Table organization**:
- First two rows show overall statistics ("All" category) corresponding to
  subplot C
- Remaining rows show category-specific statistics corresponding to subplot D
- Categories sorted alphabetically within each match type group

**Key features**:
- Combines subplot C (overall) and subplot D (by category) statistics in a
  single table
- Shows complete distribution characteristics (N, mean, median, SD) for each
  combination
- Long format allows easy comparison across categories and match types
- Statistics rounded to 3 decimal places for precision

**Interpretation**: Overall, exact matches show substantially higher mean
concordance (0.739) than fuzzy matches (0.414). This pattern holds across most
outcome categories, with psychiatric traits showing the highest exact match
concordance (0.903) and cancer traits the lowest (0.462). The table reveals
that match quality is a strong predictor of reproducibility, with exact matches
consistently outperforming fuzzy matches by 0.3+ concordance points on average.

## Case Study 5: Temporal trends tables

**Generation script**:
`processing/scripts/analysis/consolidate_cs5_manuscript_tables.py`

**Just command**: `just case-study-cs5-manuscript-tables`

**Metadata file**: `cs5_tables_metadata.json`

### Table 1: Era summary statistics

**Files**:
- `cs5_era_summary.csv` - Data in CSV format
- `cs5_era_summary.tex` - LaTeX formatted table

**Purpose**: Summarize key characteristics of MR literature across
methodological eras.

**Data source**:
`data/processed/case-study-cs5/diversity/trait_counts_by_era.csv`

**Structure**:

| Column | Description |
|--------|-------------|
| Era | Methodological era name |
| Studies (n) | Number of studies in this era |
| Year Start | First year of the era |
| Year End | Last year of the era |
| Mean Traits/Study | Average number of traits per study |
| Median Traits/Study | Median number of traits per study |

**Methodological eras**:

| Era Name | Period | Key Methodological Development |
|----------|--------|-------------------------------|
| Early Mr | 2003-2014 | Foundation period |
| Mr Egger | 2015-2017 | MR-Egger regression introduced |
| Mr Presso | 2018-2019 | MR-PRESSO outlier detection |
| Within Family | 2020-2020 | Within-family MR designs |
| Strobe Mr | 2021-2025 | STROBE-MR reporting guidelines |

**Key features**:
- Shows both mean and median to capture distribution characteristics
- Era names use title case with spaces for readability
- Year ranges clearly demarcate temporal boundaries
- Study counts reveal the explosive growth in STROBE-MR era

**Interpretation**: Demonstrates dramatic growth in MR literature
(454 studies pre-2015 vs. 12,780 in STROBE-MR era) alongside increasing trait
diversity (mean traits per study: 4.11 early vs. 7.14 in STROBE-MR era).

### Table 2: STROBE-MR reporting impact

**Files**:
- `cs5_strobe_impact.csv` - Data in CSV format
- `cs5_strobe_impact.tex` - LaTeX formatted table

**Purpose**: Quantify the impact of STROBE-MR guidelines (2021+) on reporting
completeness.

**Data source**:
`data/processed/case-study-cs5/completeness/strobe_impact_analysis.csv`

**Structure**:

| Column | Description |
|--------|-------------|
| Field | Evidence field type (CI, P-value, Direction, OR, Beta) |
| Pre-STROBE (%) | Completeness percentage before 2021 |
| Post-STROBE (%) | Completeness percentage after 2021 |
| Change (pp) | Change in percentage points |
| Chi-square | Chi-square test statistic |
| P-value | Statistical significance (scientific notation) |

**Evidence field types**:

- **Confidence Interval**: 95% CI reporting
- **P Value**: P-value reporting
- **Direction**: Effect direction (positive/negative)
- **OR**: Odds ratio reporting
- **Beta**: Beta coefficient reporting

**Key features**:
- Direct pre/post comparison centered on STROBE-MR publication (2021)
- Change expressed in percentage points (pp) for clarity
- Statistical significance testing via chi-square tests
- P-values in scientific notation (e.g., 3.19e-61)
- Negative changes indicate decreasing reporting (e.g., beta coefficients)

**Interpretation**: STROBE-MR guidelines significantly improved reporting
across most fields, with largest increases for confidence intervals (+7.7 pp)
and odds ratios (+15.4 pp). All changes are highly significant (p < 1e-30).
Beta coefficient reporting paradoxically decreased (-9.9 pp), possibly
reflecting a field-wide shift toward odds ratios for binary outcomes.

## Summary tables: Database overview

**Generation script**:
`processing/scripts/analysis/generate-manuscript-summary-table.py`

**Just command**: `just generate-manuscript-tables`

**Data file**: `summary-table-data.json`

### Aggregated summary table

**File**: `summary-table-full.tex`

**Purpose**: Provide comprehensive overview of MR-KG database characteristics.

**Data sources**:
- Overall database: `data/processed/overall-stats/`
- Trait profiles: `data/processed/trait-profiles/analysis/`
- Evidence profiles: `data/processed/evidence-profiles/analysis/`

**Table sections**:

1. **Overall MR-KG Characteristics**
   - Total unique papers (PMIDs)
   - Total unique traits
   - Total trait mentions
   - Model extraction records and results
   - Temporal coverage
   - Average results per paper

2. **Trait Profile Similarity**
   - Total PMID-model combinations
   - Total pairwise comparisons
   - Semantic similarity statistics (mean, median)
   - Jaccard similarity statistics (mean, median)
   - Semantic-Jaccard correlation

3. **Evidence Profile Similarity**
   - Total evidence combinations
   - Total pairwise comparisons
   - Direction concordance statistics
   - Composite similarity statistics

**Key features**:
- LaTeX tabular format ready for manuscript inclusion
- Numbers formatted with thousand separators for readability
- Year ranges use en-dash (2003--2026)
- Organized into logical sections with bold headers
- Caption and label included for cross-referencing

**Interpretation**: Provides readers with a quantitative overview of MR-KG
scope (15,635 papers, 75,121 traits, 2003-2026 coverage) and data quality
(semantic similarity mean 0.695, direction concordance mean 0.426).

### Per-model summary tables

**Files** (6 models):
- `summary-table-full-gpt-4-1.tex`
- `summary-table-full-gpt-5.tex`
- `summary-table-full-o4-mini.tex`
- `summary-table-full-llama3.tex`
- `summary-table-full-llama3-2.tex`
- `summary-table-full-deepseek-r1-distilled.tex`

**Purpose**: Enable model-specific performance comparison.

**Structure**: Same section structure as aggregated table but with
model-specific statistics.

**Key differences by model**:

| Model | Total Combinations | Avg Traits/Study | Avg Completeness |
|-------|-------------------|------------------|------------------|
| gpt-4-1 | 15,626 | 5.17 | 67.8% |
| gpt-5 | 15,606 | 6.81 | 62.9% |
| o4-mini | 5,366 | 5.90 | 78.2% |
| llama3 | 6,416 | 4.37 | 72.5% |
| llama3-2 | 6,670 | 5.03 | 64.9% |
| deepseek-r1-distilled | 718 | 4.45 | 64.6% |

**Interpretation**: Shows substantial variation in model coverage and
performance. GPT-4-1 and GPT-5 provide broadest coverage (15,000+
combinations), while o4-mini achieves highest evidence completeness (78.2%).
These tables support transparency and enable readers to assess model-specific
reliability.

## Technical specifications

### CSV files

- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Headers**: First row contains column names
- **Decimal precision**: 1 decimal place for percentages, varies for other
  metrics
- **Missing values**: Not applicable (all tables complete)

### LaTeX files

- **Format**: Standard `tabular` environment
- **Alignment**: Left-aligned text, right-aligned numbers
- **Lines**: `\hline` for section separators
- **Numbers**: Formatted with thousand separators where appropriate
- **Comments**: Header comments document generation script and customization
  options

### JSON metadata files

Each metadata file contains:
- `script`: Name of generation script
- `input_dir`: Source directory for input data
- `output_dir`: Destination directory for tables
- `tables_generated`: List of table identifiers

The `summary-table-data.json` file contains the complete raw data structure
used to generate all summary tables, enabling custom table generation.

## Usage guidelines

### Generating tables

All manuscript tables can be regenerated using just commands:

```bash
# Generate all manuscript tables
just manuscript-artifacts

# Generate specific case study tables
just case-study-cs1-manuscript-tables
just case-study-cs5-manuscript-tables
just generate-manuscript-tables
```

### Updating tables

When updating tables:

1. Modify the corresponding script in `processing/scripts/analysis/`
2. Run the appropriate just command
3. Verify the output in `data/artifacts/manuscript-tables/`
4. Update this documentation if table structure or interpretation changes
5. Check both CSV (for inspection) and LaTeX (for formatting) outputs

### Including LaTeX tables in manuscripts

To include a LaTeX table in your manuscript:

```latex
\input{path/to/data/artifacts/manuscript-tables/cs1_tier_distribution.tex}
```

Alternatively, copy the table contents directly into your manuscript for
further customization.

### Customizing LaTeX output

LaTeX tables include header comments with customization suggestions:

- Adjust column widths with `p{width}` specifiers
- Modify fonts with standard LaTeX font commands
- Change spacing with `\arraystretch` or `\extrarowheight`
- Add or remove `\hline` separators
- Modify number formatting as needed

## Related documentation

- Pipeline overview: @pipeline.md
- Database schema: @db-schema.md
- Manuscript figures: @manuscript-figures.md
- Case studies configuration: @../processing/config/case_studies.yml
- Summary statistics: @summary-statistics.md
