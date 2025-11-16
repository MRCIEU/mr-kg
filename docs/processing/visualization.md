# Visualization guide for MR-KG

This guide explains how to use the Jupyter notebooks for visualizing MR-KG
summary statistics and creating publication-ready figures.

## Overview

The MR-KG project includes interactive Jupyter notebooks that transform
summary statistics into publication-quality visualizations using Altair.
These notebooks provide both exploratory analysis and manuscript-ready
figures.

**Available notebooks:**

1. **`1-mr-literature.ipynb`**: MR literature corpus characteristics
2. **`2-mr-kg.ipynb`**: Comprehensive MR-KG database visualizations

**Key features:**

- Interactive visualizations with tooltips and zooming
- Consistent styling and color schemes
- Export capabilities for manuscripts (JSON, PNG, SVG)
- Clear data validation and error messages
- Reproducible analysis workflow

## Prerequisites

### Data dependencies

All notebooks require pre-computed summary statistics. Generate them before
running notebooks:

```bash
cd processing
just generate-all-summary-stats
```

This creates the required CSV files in:
- `data/processed/overall-stats/`
- `data/processed/trait-profiles/analysis/`
- `data/processed/evidence-profiles/analysis/`

### Environment setup

Ensure the processing environment is installed:

```bash
cd processing
uv sync
```

The environment includes:
- Altair 6.0.0 (visualization library)
- pandas 2.3.1+ (data manipulation)
- DuckDB 1.3.2+ (database queries)
- yiutils (project utilities)

## Quick start

### Starting Jupyter

From the processing directory:

```bash
cd processing
jupyter notebook notebooks/
```

Or use JupyterLab:

```bash
jupyter lab notebooks/
```

Your browser will open showing the notebooks directory.

### Running a notebook

1. Open the desired notebook (e.g., `1-mr-literature.ipynb`)
2. Select **Kernel > Restart & Run All** to execute all cells
3. Review visualizations as they appear
4. Optionally uncomment export cells to save figures

### Troubleshooting startup

If notebooks fail to start:

```bash
# Ensure Jupyter is installed
uv add --dev jupyter

# Try running from project root
cd /path/to/mr-kg
jupyter notebook processing/notebooks/
```

## Notebook 1: MR Literature Corpus Analysis

**File:** `processing/notebooks/1-mr-literature.ipynb`

**Purpose:** Visualize temporal characteristics of the MR literature dataset.

### What it visualizes

**Data sources:**
- PubMed metadata from `vector_store.db`
- Temporal statistics from `overall-stats/temporal-statistics.csv`

**Coverage:** ~15,635 papers from the MR literature corpus

### Key plots

#### Plot 1: Temporal distribution (bar chart)

Shows the number of papers published per year.

**Features:**
- Bar chart with color gradient by paper count
- Temporal x-axis with year formatting
- Interactive tooltips showing year and paper count

**Interpretation:**
- Identifies peak publication years
- Shows growth trajectory of MR research
- Reveals temporal gaps or surges in publications

#### Plot 2: Cumulative growth (line chart)

Shows cumulative paper count over time.

**Features:**
- Line chart with point markers
- Cumulative sum calculation
- Interactive tooltips

**Interpretation:**
- Reveals overall growth trajectory
- Shows acceleration or deceleration periods
- Useful for understanding field maturation

### Summary statistics table

The notebook computes and displays:

- **Total papers**: Complete corpus size
- **Year range**: Earliest to latest publication
- **Year span**: Duration of coverage
- **Average papers per year**: Mean publication rate
- **Recent vs earlier papers**: Distribution over time periods
- **Percentage in last 5 years**: Recent publication concentration

### Example output

```text
Total Papers:             15,635
Year Range:               1990 - 2024
Year Span:                35 years
Average Papers per Year:  446.7
Papers in Last 5 Years:   8,234
Papers Before 2020:       7,401
Percentage Last 5 Years:  52.7%
```

### Customization tips

**Changing time ranges:**

Modify the SQL query in the data loading cell:

```python
# Original: BETWEEN 1990 AND 2025
# Modified: BETWEEN 2000 AND 2024
query = """
SELECT 
    CAST(SUBSTR(pub_date, 1, 4) AS INTEGER) as year,
    COUNT(*) as paper_count
FROM mr_pubmed_data
WHERE pub_date IS NOT NULL 
    AND LENGTH(pub_date) >= 4
    AND CAST(SUBSTR(pub_date, 1, 4) AS INTEGER) BETWEEN 2000 AND 2024
GROUP BY year
ORDER BY year
"""
```

**Changing color schemes:**

Modify the `scale` parameter in chart encoding:

```python
# Original: scheme="viridis"
# Alternatives: "blues", "greens", "reds", "plasma", "turbo"
color=alt.Color(
    "paper_count:Q",
    scale=alt.Scale(scheme="blues"),
    legend=None,
)
```

## Notebook 2: MR-KG Database Characteristics

**File:** `processing/notebooks/2-mr-kg.ipynb`

**Purpose:** Comprehensive visualization of all MR-KG databases and
similarity metrics.

### Structure

The notebook is organized into four major sections:

- **Section A**: Overall database characteristics
- **Section B**: Trait profile similarity
- **Section C**: Evidence profile similarity
- **Section D**: Cross-database comparison

### Section A: Overall Database Characteristics

#### Plot 1: Database entity counts

Horizontal bar chart showing total entities in MR-KG.

**Metrics:**
- Unique papers (PMIDs)
- Unique traits
- Model extraction results

**Interpretation:**
- Overall database scale
- Completeness of extraction pipeline
- Relationship between papers, traits, and results

#### Plot 2: Per-model extraction statistics

Grouped bar chart comparing extraction performance across models.

**Metrics:**
- Extraction count per model
- Unique papers processed
- Average traits per extraction

**Interpretation:**
- Model extraction efficiency
- Coverage differences between models
- Quality indicators for model selection

#### Plot 3: Top journals

Horizontal bar chart of the 20 most frequent journals.

**Metrics:**
- Paper count per journal
- Percentage of total corpus

**Interpretation:**
- Source distribution of MR literature
- Journal specialization in MR methods
- Potential publication venue identification

### Section B: Trait Profile Similarity

#### Plot 4: Similarity score distributions

Multi-panel density plots showing trait similarity distributions by model.

**Metrics:**
- Semantic similarity (SciSpacy embeddings)
- Jaccard similarity (set overlap)

**Features:**
- Faceted by model (2 columns)
- Overlaid density curves for both metrics
- Opacity for visual separation

**Interpretation:**
- Distribution shape reveals similarity patterns
- Model-specific extraction characteristics
- Comparison between semantic and lexical similarity

**Understanding distributions:**
- **Right-skewed**: Most pairs have low similarity (diverse studies)
- **Left-skewed**: Most pairs have high similarity (homogeneous studies)
- **Bimodal**: Two distinct similarity clusters
- **Uniform**: Evenly distributed similarities

#### Plot 5: Trait count distribution

Histogram showing number of traits extracted per study.

**Interpretation:**
- Typical trait count per study
- Outliers with unusually high trait counts
- Data quality indicators (e.g., over-extraction)

**Typical patterns:**
- Most studies have 2-10 traits
- Long tail of studies with many traits
- Very few single-trait studies

#### Plot 6: Similarity metric correlation

Scatter plot with regression lines showing correlation between semantic
and Jaccard similarity.

**Metrics:**
- X-axis: Semantic similarity (embedding-based)
- Y-axis: Jaccard similarity (set-based)
- Color: Model

**Interpretation:**
- Strong positive correlation expected (both measure similarity)
- Deviations reveal cases where methods disagree
- Model-specific patterns in correlation strength

**When metrics disagree:**
- **High semantic, low Jaccard**: Paraphrased traits (same concept,
  different words)
- **Low semantic, high Jaccard**: Lexically similar but semantically
  different
- **Both low**: Truly dissimilar trait pairs
- **Both high**: Strong agreement on similarity

### Section C: Evidence Profile Similarity

Evidence profiles are sparser than trait profiles because they require
quantitative MR results (effect sizes, p-values, etc.).

#### Plot 7: Evidence similarity metrics

Multi-panel violin plots showing distributions of evidence-level metrics.

**Metrics:**
- **Direction concordance**: Agreement on effect direction (positive/negative)
- **Effect size similarity**: Magnitude similarity (when both available)
- **Statistical consistency**: Agreement on statistical significance
- **Evidence overlap**: Proportion of matched trait pairs

**Features:**
- Horizontal violin plots (density + mirroring)
- Separate panel per metric
- Color by model

**Interpretation:**

**Direction concordance:**
- Values near 1.0: High agreement on causal direction
- Values near 0.5: Random agreement (concerning)
- Values near 0.0: Systematic disagreement (very concerning)

**Effect size similarity:**
- Requires both studies to report quantitative effects
- Higher values indicate replicable effect magnitudes
- Missing data common (many studies lack effect sizes)

**Statistical consistency:**
- Agreement on statistical significance thresholds
- Values near 1.0: Consistent conclusions
- Lower values: Publication bias or heterogeneity

**Evidence overlap:**
- Proportion of trait pairs with matched MR results
- Higher overlap enables better evidence comparison
- Low overlap limits analysis to trait-level similarity

#### Plot 8: Data completeness by model

Stacked bar chart showing proportion of high/medium/low completeness.

**Completeness categories:**
- **High** (green): >= 75% of fields populated
- **Medium** (orange): 50-75% populated
- **Low** (red): < 50% populated

**Interpretation:**
- Models differ in extraction completeness
- Low completeness limits evidence profile analysis
- Affects interpretation of similarity metrics

**Factors affecting completeness:**
- Reporting standards in original papers
- Model extraction capabilities
- Result structure variability

#### Plot 9: Matched pairs distribution

Box plot showing distribution of matched trait pairs per study pair.

**Interpretation:**
- **Median**: Typical overlap between studies
- **Outliers**: Study pairs with unusually high overlap
- **Model differences**: Extraction consistency

**High matched pairs suggest:**
- Studies on similar research questions
- Overlapping trait sets (e.g., common outcomes)
- Potential for evidence synthesis

### Section D: Cross-Database Comparison

#### Plot 10: Model coverage comparison

Side-by-side bar charts comparing trait vs evidence profile coverage.

**Metrics:**
- Total combinations (study pairs) per model
- Separate panels for trait and evidence profiles

**Interpretation:**
- Evidence profiles consistently sparser
- Model coverage varies substantially
- Some models better suited for evidence extraction

**Why evidence profiles are sparser:**
- Require quantitative MR results (not all studies report)
- Need structured result formats for extraction
- Matching criteria more stringent

## Customization guide

### Modifying Altair plots

All visualizations use Altair's declarative syntax. Key customization
points:

#### Changing figure sizes

```python
.properties(
    width=800,    # Default: 600-700
    height=500,   # Default: 400
    title="Your Title Here"
)
```

#### Adjusting color schemes

Altair supports many built-in schemes:

```python
# Sequential (single hue)
scale=alt.Scale(scheme="blues")     # blues, greens, reds, etc.

# Sequential (multi-hue)
scale=alt.Scale(scheme="viridis")   # viridis, plasma, inferno, etc.

# Diverging
scale=alt.Scale(scheme="blueorange") # blueorange, redblue, etc.

# Categorical
scale=alt.Scale(scheme="category10") # category10, category20, etc.
```

For custom colors:

```python
scale=alt.Scale(
    domain=["High", "Medium", "Low"],
    range=["#2ca02c", "#ff7f0e", "#d62728"]
)
```

#### Modifying tooltips

Add or remove tooltip fields:

```python
tooltip=[
    alt.Tooltip("model:N", title="Model"),
    alt.Tooltip("value:Q", title="Value", format=".2f"),
    alt.Tooltip("percentage:Q", title="Percentage", format=".1%"),
]
```

Tooltip format strings:
- `.2f`: 2 decimal places
- `,.0f`: Thousands separator, no decimals
- `.1%`: Percentage with 1 decimal

#### Changing axis formatting

```python
x=alt.X(
    "year:T",
    title="Publication Year",
    axis=alt.Axis(
        format="%Y",        # Year format
        labelAngle=-45,     # Rotate labels
        tickCount=10,       # Number of ticks
    )
)
```

### Themes

Altair supports several built-in themes:

```python
# Available themes
alt.themes.enable("default")     # Default Altair theme
alt.themes.enable("dark")        # Dark background
alt.themes.enable("ggplot2")     # R's ggplot2 style
alt.themes.enable("quartz")      # Quartz style
alt.themes.enable("vox")         # Vox Media style
```

Apply themes in the setup cell:

```python
alt.themes.enable("ggplot2")
```

### Font sizes and styling

Create a custom theme:

```python
def custom_theme():
    return {
        "config": {
            "title": {"fontSize": 16, "font": "Arial"},
            "axis": {
                "labelFontSize": 12,
                "titleFontSize": 14,
                "labelFont": "Arial",
                "titleFont": "Arial",
            },
            "legend": {
                "labelFontSize": 12,
                "titleFontSize": 14,
            },
        }
    }

alt.themes.register("custom", custom_theme)
alt.themes.enable("custom")
```

## Export options

### Saving as JSON (recommended)

JSON specs preserve full interactivity and can be embedded in web pages:

```python
chart.save("output.json")
```

To view saved JSON specs:
- Use Vega Editor: https://vega.github.io/editor/
- Embed in HTML with Vega-Embed
- Load in other notebooks

### Saving as PNG/SVG (for manuscripts)

Requires additional dependencies:

```bash
# Install altair_saver and dependencies
uv add --dev altair_saver vl-convert-python
```

Then export:

```python
# PNG (raster)
chart.save("output.png", scale_factor=2.0)  # 2x resolution

# SVG (vector, editable)
chart.save("output.svg")
```

### Batch export in notebooks

Both notebooks include optional export cells at the end. To use:

1. Uncomment the export cells
2. Run the cells
3. Figures saved to `data/processed/figures/`

Example structure:

```python
output_dir = PROCESSED_DIR / "figures" / "mr-kg"
output_dir.mkdir(parents=True, exist_ok=True)

entity_chart.save(str(output_dir / "entity_counts.json"))
model_chart.save(str(output_dir / "model_statistics.json"))
# ... additional charts
```

### Export for LaTeX/manuscripts

For publication-quality figures:

1. **Export as SVG** (vector format, scalable)
2. **Use high resolution PNG** (if raster required)
3. **Save both JSON and PNG/SVG** (archive interactive + static)

```python
# Save both formats
chart.save("figure.svg")
chart.save("figure.png", scale_factor=3.0)  # 3x for print
chart.save("figure.json")  # Archive interactive version
```

## Troubleshooting

### Missing data files

**Error:** `FileNotFoundError: Data file not found`

**Solution:**

```bash
cd processing
just generate-all-summary-stats
```

Ensure all three analysis steps complete successfully:
1. Overall database statistics
2. Trait profile analysis
3. Evidence profile analysis

### Import errors

**Error:** `ModuleNotFoundError: No module named 'altair'`

**Solution:**

```bash
cd processing
uv sync
```

Ensure you're running Jupyter from the processing environment:

```bash
cd processing
uv run jupyter notebook notebooks/
```

### Altair data transformer errors

**Error:** `MaxRowsError: The number of rows in your dataset is greater than
the maximum allowed`

**Solution:** Already handled in setup cells:

```python
alt.data_transformers.enable("default", max_rows=None)
```

If error persists, use data transformer debugging:

```python
# Check current transformer
print(alt.data_transformers.active)

# Try vegafusion (for very large datasets)
alt.data_transformers.enable("vegafusion")
```

### Database connection errors

**Error:** `DuckDBPyConnection error`

**Solution:**

1. Verify database exists:
   ```bash
   ls -lh data/db/vector_store.db
   ```

2. Check database is not corrupted:
   ```bash
   duckdb data/db/vector_store.db "SELECT COUNT(*) FROM mr_pubmed_data;"
   ```

3. Ensure read-only connection (in notebooks):
   ```python
   conn = duckdb.connect(str(vector_db_path), read_only=True)
   ```

### Empty or missing plots

**Problem:** Cells execute but no plots appear

**Solutions:**

1. **Check data availability:**
   ```python
   print(f"Data shape: {df.shape}")
   print(f"Empty: {df.empty}")
   display(df.head())
   ```

2. **Verify Altair installation:**
   ```python
   import altair as alt
   print(alt.__version__)  # Should be 6.0.0+
   ```

3. **Try rendering explicitly:**
   ```python
   chart.display()  # or
   display(chart)
   ```

4. **Check for null values:**
   ```python
   print(df.isnull().sum())
   ```

### Slow rendering

**Problem:** Large datasets cause slow plot rendering

**Solutions:**

1. **Sample data for exploration:**
   ```python
   sampled_df = df.sample(n=1000, random_state=42)
   chart = alt.Chart(sampled_df)...
   ```

2. **Aggregate before plotting:**
   ```python
   aggregated = df.groupby("category").mean().reset_index()
   chart = alt.Chart(aggregated)...
   ```

3. **Use data transformer:**
   ```python
   alt.data_transformers.enable("json")  # Faster for large data
   ```

## Integration with analysis pipeline

### Workflow

The complete workflow from raw data to visualizations:

```mermaid
flowchart LR
    A[Raw Data] --> B[Processing Pipeline]
    B --> C[DuckDB Databases]
    C --> D[Summary Statistics Scripts]
    D --> E[CSV Outputs]
    E --> F[Jupyter Notebooks]
    F --> G[Visualizations]
    G --> H[Manuscript Figures]
```

### When to regenerate visualizations

Regenerate notebooks when:

1. **Database updates**: New extractions added to vector store
2. **Analysis changes**: Modified summary statistics scripts
3. **Parameter tuning**: Changed similarity thresholds or metrics
4. **Model updates**: New extraction models added

### Automation

Automate visualization generation:

```bash
# Generate all stats and export plots
cd processing

# Step 1: Generate summary statistics
just generate-all-summary-stats

# Step 2: Execute notebooks programmatically
uv run jupyter nbconvert \
    --to notebook \
    --execute \
    notebooks/1-mr-literature.ipynb \
    --output 1-mr-literature-executed.ipynb

uv run jupyter nbconvert \
    --to notebook \
    --execute \
    notebooks/2-mr-kg.ipynb \
    --output 2-mr-kg-executed.ipynb

# Step 3: Export to HTML for sharing
uv run jupyter nbconvert \
    --to html \
    notebooks/1-mr-literature-executed.ipynb

uv run jupyter nbconvert \
    --to html \
    notebooks/2-mr-kg-executed.ipynb
```

For scheduled regeneration, create a shell script:

```bash
#!/bin/bash
# regenerate_visualizations.sh

set -e

cd "$(dirname "$0")"
cd processing

echo "Generating summary statistics..."
just generate-all-summary-stats

echo "Executing notebooks..."
uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb

echo "Visualization regeneration complete!"
```

### Version control

**What to commit:**
- Notebook files (`.ipynb`)
- Notebook outputs (if small and informative)
- Documentation

**What not to commit:**
- Large PNG/SVG exports (use Git LFS or exclude)
- Executed notebook copies (can regenerate)
- Temporary output directories

**.gitignore entries:**

```gitignore
# Notebook outputs
data/processed/figures/
notebooks/*-executed.ipynb
notebooks/.ipynb_checkpoints/

# Export formats
*.png
*.svg
*.pdf
```

## Best practices

### Reproducibility

1. **Always restart kernel** before final execution
2. **Run all cells in order** (don't skip cells)
3. **Document parameter changes** in markdown cells
4. **Save notebook with outputs** for archival
5. **Include random seeds** for sampling operations

### Code organization

1. **One main visualization per cell**
2. **Separate data preparation from plotting**
3. **Add markdown headers** before each plot
4. **Include brief interpretations** after plots
5. **Group related visualizations** in sections

### Performance

1. **Load data once** in dedicated cells
2. **Cache expensive computations**
3. **Sample large datasets** for exploration
4. **Use efficient data structures** (avoid copying)
5. **Close database connections** after use

### Sharing

When sharing notebooks:

1. **Clear sensitive outputs** if needed
2. **Include data availability notes**
3. **Document dependencies** in environment file
4. **Provide minimal working example** if possible
5. **Export to HTML** for non-technical users

## Additional resources

### Altair documentation

- Official docs: https://altair-viz.github.io/
- Example gallery: https://altair-viz.github.io/gallery/
- API reference: https://altair-viz.github.io/user_guide/API.html

### Jupyter documentation

- Jupyter notebook: https://jupyter-notebook.readthedocs.io/
- JupyterLab: https://jupyterlab.readthedocs.io/
- nbconvert: https://nbconvert.readthedocs.io/

### MR-KG documentation

- Processing pipeline: @docs/processing/pipeline.md
- Summary statistics: @docs/processing/summary-statistics.md
- Database schema: @docs/processing/db-schema.md
- Trait similarity: @docs/processing/trait-profile-similarity.md
- Evidence similarity: @docs/processing/evidence-profile-similarity.md

### Support

For issues with:
- **Notebooks**: Check this guide's troubleshooting section
- **Data processing**: See @docs/processing/pipeline.md
- **Database queries**: See @docs/processing/db-schema.md
- **Analysis scripts**: See script docstrings and help messages
