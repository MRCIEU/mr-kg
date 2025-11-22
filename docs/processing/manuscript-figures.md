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

**Purpose**: Show reproducibility metrics distribution through four complementary views: overall tier distribution, category-specific tier distribution, overall concordance distribution, and category-specific concordance distribution.

**Structure**: Two-column layout with:
- **Left panel**: 
  1. **Overall Tier Distribution** (subplot A, top): Single stacked horizontal bar showing overall reproducibility tier percentages
  2. **Tier Distribution by Outcome Category** (subplot B, bottom): Stacked horizontal bars showing tier distribution per outcome category
- **Right panel**:
  3. **Overall Concordance by Match Type** (subplot C, top): Ridge plots (overlapping density curves) showing distribution of concordance values for all categories combined, stratified by match type (exact vs fuzzy)
  4. **Concordance by Match Type and Outcome Category** (subplot D, bottom): Faceted ridge plots showing distribution of concordance values for each outcome category, stratified by match type

**Layout rationale**: 
- Subplots A and B are grouped together in the left panel because they both show tier distribution metrics
- Subplots C and D in the right panel both show concordance distributions
- Subplot C is vertically aligned with subplot A via a spacer, creating a balanced two-column layout
- This grouping emphasizes the relationship between aggregate (A, C) and category-specific (B, D) views for both reproducibility tiers and direction concordance

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

**Visual elements (Panel C - top right)**:
- Ridge plot (overlapping density curves) showing distribution of direction concordance values
- Two match types with distinct colors:
  - Exact match (olive green)
  - Fuzzy match (orange)
- Semi-transparent area fills (70% opacity) to show overlapping distributions
- Smooth monotone interpolation for density curves
- Light gray outlines on density areas
- X-axis: Direction Concordance (-1 to 1)
- Y-axis: Density (axis hidden)
- No grid lines for clean visualization
- Width: 400px
- Height: 30px (reduced height for compact display)
- No legend (shared legend shown in Panel D)
- Vertically aligned with Panel A via 80px spacer above

**Visual elements (Panel D - bottom right)**:
- Faceted ridge plots (stacked density curves) showing distribution of direction concordance values per category
- Each category has its own row (facet) with overlapping density curves for exact and fuzzy matches
- Two match types with distinct colors:
  - Exact match (olive green)
  - Fuzzy match (orange)
- Semi-transparent area fills (70% opacity) to show overlapping distributions
- Smooth monotone interpolation for density curves
- Light gray outlines on density areas
- X-axis: Direction Concordance (-1 to 1)
- Y-axis: None (density axis hidden for clean faceted display)
- No grid lines for clean visualization
- Categories stacked vertically with labels on the left
- Width: 400px
- Height: 42px per category row
- Legend: Match Type (bottom, horizontal orientation)
- Facet spacing: 0 (tight vertical stacking)
- Row headers: Category names, left-aligned

**Key features**:
- Left panel (A+B) groups tier distribution metrics for visual coherence
- Right panel (C+D) groups concordance distribution metrics for visual coherence
- Panel C vertically aligned with Panel A using invisible spacer for balanced layout
- Panel A provides high-level summary of overall reproducibility distribution
- Panel B shows how reproducibility varies by disease category
- Panel C shows overall distribution of concordance values using ridge plot style with overlapping densities
- Panel D uses faceted ridge plots to compare distributions across categories
- Ridge plots provide intuitive visualization of distribution shapes and overlaps
- Overlapping distributions make it easy to compare exact vs fuzzy match quality
- Grid lines removed from subplots C and D for cleaner visualization
- Uses "Outcome Category" (not "Disease Category") for consistency
- Categories: cancer, autoimmune, cardiovascular, metabolic, other, psychiatric
- Reproducibility tiers: high, moderate, low, discordant
- Match types: exact, fuzzy
- Text labels improve readability by showing exact pair counts in panels A and B
- Independent color scales for different legend types
- Compact vertical design maximizes information density

**Data sources**:
- Panel A: `data/artifacts/manuscript-tables/cs1_tier_distribution.csv`
- Panel B: Computed from `pair_reproducibility_metrics.csv`
- Panel C: Extracted from `pair_reproducibility_metrics.csv` (individual concordance values, all categories)
- Panel D: Extracted from `pair_reproducibility_metrics.csv` (individual concordance values by category)

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
