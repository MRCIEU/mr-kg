# Manuscript figures

This document describes the manuscript-ready figures generated for publication, including their design decisions, data sources, and interpretation guidelines.

## Overview

Manuscript figures are stored in: `data/artifacts/manuscript-figures/`

These figures are optimized for publication with:
- 300 DPI resolution
- Altair-based vector graphics exported to PNG
- Publication-appropriate sizing and formatting
- Clear legends and axis labels

## Case Study 5: Temporal trends in MR evidence

**File**: `cs5_fig_temporal_trends.png`

**Figure title**: Temporal Trends in MR Evidence

**Generation script**: `processing/scripts/analysis/case_study_5_manuscript_figures.py`

**Just command**: `just case-study-cs5-manuscript-figures`

### Figure structure

This figure comprises three vertically stacked time series subplots covering the period 2003-2025:

1. **Trait Diversity Over Time** (top panel)
2. **Overall Reporting Completeness Over Time** (middle panel)
3. **Reporting Completeness Over Time by Field Type** (bottom panel)

All three subplots share a common temporal axis (2003-2025) with methodological era markers indicated by red vertical lines with numeric labels.

### Methodological era markers

Red vertical lines mark the beginning of each methodological era, labeled numerically:

| Label | Era Name | Start Year | End Year | Description |
|-------|----------|------------|----------|-------------|
| 1 | early_mr | 2003 | 2014 | Early MR period |
| 2 | mr_egger | 2015 | 2016 | Introduction of MR-Egger |
| 3 | mr_presso | 2017 | 2019 | Introduction of MR-PRESSO |
| 4 | within_family | 2020 | 2020 | Within-family MR methods |
| 5 | strobe_mr | 2021 | 2025 | STROBE-MR guidelines era |

**Note**: Era 5 (STROBE-MR, 2021) is particularly significant as it marks the introduction of reporting guidelines for Mendelian Randomization studies.

### Subplot 1: Trait diversity over time

**Purpose**: Visualize temporal trends in the diversity of traits examined in MR studies.

**Data source**: `data/processed/case-study-cs5/diversity/trait_counts_by_year.csv`

**Visual elements**:
- Blue dashed line connecting data points
- Blue circular points representing mean trait count per study for each year
- Gray shaded area showing standard error bands

**Data availability note**: 
- Data begins in **2005**, not 2003
- Years 2003-2004 have no data points because the trait diversity analysis requires sufficient trait extraction data, which was not available for these early years
- The x-axis spans 2003-2025 for consistency with other subplots, but data visualization only covers 2005-2025

**Interpretation**: Shows increasing trait diversity in MR studies over time, with notable acceleration around the MR-PRESSO era (2017+).

### Subplot 2: Overall reporting completeness over time

**Purpose**: Track the overall completeness of key statistical reporting elements across all MR studies.

**Data source**: `data/processed/case-study-cs5/completeness/field_completeness_by_year.csv`

**Visual elements**:
- Three colored lines with points representing different completeness metrics:
  - **95% CI** (blue): Confidence interval reporting
  - **P-value** (purple): P-value reporting  
  - **SE** (brown): Standard error reporting

**Key differences from subplot 3**:
- This subplot shows **aggregated completeness** across all field types for the three most commonly reported statistical measures
- Focuses on overall temporal trends in statistical reporting completeness
- Simpler view highlighting the most critical reporting elements

**Interpretation**: Confidence interval reporting shows consistently high and improving completeness (>80%). P-value reporting is moderate (40-55%). Standard error reporting remains low (<15%) throughout the period.

### Subplot 3: Reporting completeness over time by field type

**Purpose**: Provide detailed view of reporting completeness trends for different types of evidence fields.

**Data source**: Derived from `data/processed/case-study-cs5/completeness/field_completeness_by_year.csv`

**Visual elements**:
- Four colored lines with points representing different field types:
  - **Confidence Interval** (orange): CI reporting completeness
  - **Direction** (green): Effect direction reporting
  - **Effect Size** (red): Effect size reporting (any type: beta, OR, HR)
  - **Statistical** (pink): Statistical significance reporting (p-values)

**Data preparation**:
- Effect size completeness is computed as the maximum across three effect size types (beta coefficients, odds ratios, hazard ratios)
- All field types are tracked yearly to show granular temporal trends

**Key differences from subplot 2**:
- This subplot shows **field-type-specific completeness** for four categories of evidence
- Includes effect direction and combined effect size measures not shown in subplot 2
- More comprehensive view of all reporting completeness dimensions
- Different categorization scheme emphasizing evidence profile components

**Why two similar completeness subplots?**

The two completeness subplots serve complementary purposes:

1. **Subplot 2 (Overall)**: Provides a high-level summary focused on the three most critical statistical measures (CI, P-value, SE). This is the "headline" view showing general reporting quality trends.

2. **Subplot 3 (By Field Type)**: Offers a detailed breakdown by evidence field categories, including effect direction and combined effect sizes. This view is essential for understanding:
   - Which types of evidence are better reported
   - How different field types have evolved differently over time
   - The relationship between mandatory STROBE-MR fields and completeness

Together, these subplots tell a complete story: subplot 2 shows *whether* key statistics are reported, while subplot 3 shows *what types* of evidence are reported.

### Technical specifications

- **Width**: 900 pixels (1.5× standard width for time series data)
- **Height**: 200 pixels per subplot
- **Resolution**: 300 DPI
- **Year axis**: Integer formatting (2005, not 2005.0)
- **Completeness scale**: 0-100%

### Legend note

**Current limitation**: Due to Altair's vconcat constraints, all three subplots currently share a combined legend on the right side labeled "Field, Field Type". This shows all unique field/metric names across the three subplots.

**Interpretation guide for legend**:
- **Subplot 1** uses: No legend items (error bands shown in gray)
- **Subplot 2** uses: P-value, 95% CI, SE
- **Subplot 3** uses: Confidence Interval, Direction, Effect Size, Statistical

Users should reference the subplot titles and line colors to interpret each panel correctly.

## Case Study 1: Reproducibility figures

**Files**: 
- `cs1_fig1_category_reproducibility.png` 
- `cs1_fig2_study_count_reproducibility.png`

**Generation script**: `processing/scripts/analysis/case_study_1_manuscript_figures.py`

**Just command**: `just case-study-cs1-manuscript-figures`

### Figure 1: Distribution of reproducibility metrics

**Figure title**: Distribution of reproducibility metrics

**Purpose**: Show reproducibility metrics distribution through three complementary views: overall tier distribution, category-specific tier distribution, and concordance by match quality.

**Structure**: Two-column layout with:
- **Left panel**: 
  1. **Overall Tier Distribution** (top): Single stacked horizontal bar showing overall reproducibility tier percentages
  2. **Tier Distribution by Outcome Category** (bottom): Stacked horizontal bars showing tier distribution per outcome category
- **Right panel**:
  3. **Concordance by Match Quality and Outcome Category**: Error bars showing mean direction concordance

**Layout rationale**: Subplots A and B are grouped together in the left panel because they both show tier distribution metrics, while subplot C displays a different metric (concordance) in the right panel. This grouping emphasizes the relationship between overall and category-specific tier distributions.

**Visual elements (Panel A - top left)**:
- Single stacked horizontal bar showing percentage distribution across all pairs
- Four reproducibility tiers with distinct colors:
  - High (green)
  - Moderate (orange)
  - Low (red)
  - Discordant (gray)
- Text labels showing absolute pair counts (n) centered within each bar segment
- X-axis spans 0-100% with percentage scale
- Width: 400px
- Height: 80px
- No legend (legend shown in panel B instead)

**Visual elements (Panel B - bottom left)**:
- Stacked horizontal bars showing percentage distribution per outcome category
- Four reproducibility tiers with distinct colors (same as Panel A)
- Text labels centered within each bar segment showing absolute counts
- Categories sorted by high tier percentage (ascending)
- X-axis: Percentage (0-100%)
- Y-axis: Outcome Category
- Width: 400px
- Height: 300px
- Legend: Reproducibility Tier (bottom)

**Visual elements (Panel C - right)**:
- Horizontal bars showing mean direction concordance
- Error bars representing 95% confidence intervals
- Two match types with distinct colors:
  - Exact match (dark green)
  - Fuzzy match (orange)
- Y-offset positioning for grouped comparison within each category
- White bold text labels inside bars showing concordance values (2 decimal places)
- Width: 400px
- Height: 300px
- Legend: Match Type (bottom)

**Key features**:
- Left panel (A+B) groups tier distribution metrics for visual coherence
- Panel A provides high-level summary of overall reproducibility distribution
- Panel B shows how reproducibility varies by disease category
- Right panel (C) examines effect of match quality on reproducibility
- Uses "Outcome Category" (not "Disease Category") for consistency
- Categories: cancer, autoimmune, cardiovascular, metabolic, other, psychiatric
- Reproducibility tiers: high, moderate, low, discordant
- Match types: exact, fuzzy
- Text labels improve readability by showing exact pair counts
- Independent color scales for different legend types

**Data sources**:
- Panel A: `data/artifacts/manuscript-tables/cs1_tier_distribution.csv`
- Panel B: Computed from `pair_reproducibility_metrics.csv`
- Panel C: `data/processed/case-study-cs1/interactions/category_match_interaction.csv`

### Figure 2: Study count and reproducibility

**Purpose**: Examine relationship between number of studies per trait pair and reproducibility.

**Structure**: 2×2 faceted histograms showing concordance distribution across four study count bands:
- Small sample (2-3 studies): mean = 0.50
- Medium sample (4-6 studies): mean = 0.44
- Large sample (7-10 studies): mean = 0.33
- Very large sample (11+ studies): mean = 0.44

**Visual elements**:
- Steelblue histograms showing distribution of direction concordance values (20 bins)
- Red vertical line: mean concordance for each band
- Red text label: numeric value of mean (2 decimal places, positioned near top)
- Green dashed line: high reproducibility threshold (0.7)
- Reference lines legend at bottom

**Technical specifications**:
- Width: 300 pixels per panel (1.5× standard)
- Height: 200 pixels per panel
- Independent scales for both x and y axes across facets
- Each facet computes and displays its own mean value

## Summary figure: MR literature temporal distribution

**File**: `summary_temporal_distribution.png`

**Generation script**: `processing/scripts/analysis/summary_manuscript_figures.py`

**Just command**: `just summary-manuscript-figures`

**Purpose**: Show the growth of Mendelian Randomization literature over time.

**Data source**: Derived from `processing/notebooks/1-mr-literature.ipynb`

**Visual elements**:
- Log-scale y-axis (papers per year)
- Line plot with data point labels
- Covers full temporal range of MR literature

**Key features**:
- Illustrates exponential growth in MR publications
- Provides context for the temporal scope of the MR-KG database
- Shows the dramatic increase in MR research, especially post-2015

## Usage guidelines

### Generating figures

All manuscript figures can be regenerated using just commands:

```bash
# Generate all manuscript figures
just manuscript-artifacts

# Generate specific case study figures
just case-study-cs1-manuscript-figures
just case-study-cs5-manuscript-figures
just summary-manuscript-figures
```

### Updating figures

When updating figures:

1. Modify the corresponding script in `processing/scripts/analysis/`
2. Run the appropriate just command
3. Verify the output in `data/artifacts/manuscript-figures/`
4. Update this documentation if visual elements or interpretations change

## Related documentation

- Pipeline overview: @pipeline.md
- Database schema: @db-schema.md
- Case studies configuration: @../processing/config/case_studies.yml
- Summary statistics: @summary-statistics.md
