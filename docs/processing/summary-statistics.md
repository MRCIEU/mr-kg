# Summary statistics for manuscripts

This guide explains how to generate comprehensive summary statistics and
publication-ready tables from the MR-KG databases.

## Overview

The summary statistics workflow consolidates data from all three MR-KG
databases to create manuscript-ready outputs:

1. **Vector store database** (`vector_store.db`): Overall MR-KG characteristics
2. **Trait profile database** (`trait_profile_db.db`): Trait similarity metrics
3. **Evidence profile database** (`evidence_profile_db.db`): Evidence
   concordance metrics

## Quick start

Generate all summary statistics and LaTeX tables in one command:

```bash
cd processing
just generate-all-summary-stats
```

This executes three scripts in sequence:

1. `generate-overall-database-stats.py`: Extract overall statistics
2. `analyze-trait-summary-stats.py`: Compute trait profile metrics
3. `generate-manuscript-summary-table.py`: Create LaTeX tables

## Visualization notebooks

After generating summary statistics, explore the data interactively using
Jupyter notebooks:

```bash
cd processing
jupyter notebook notebooks/
```

### Available notebooks

**1-mr-literature.ipynb**

- Temporal distribution of MR literature
- Publication trends and growth analysis
- Basic corpus statistics

**2-mr-kg.ipynb**

- Comprehensive database visualizations
- Similarity metric distributions
- Model comparison dashboards
- Data quality assessments

All plots are created using Altair and can be exported for manuscript
inclusion.

See @docs/processing/visualization.md for detailed visualization
documentation.

## Three-step workflow

### Step 1: Overall database statistics

Extract comprehensive statistics from the vector store database.

Command:

```bash
just generate-overall-stats
```

Direct script execution:

```bash
uv run scripts/analysis/generate-overall-database-stats.py \
  --vector-db ../data/db/vector_store.db \
  --output-dir ../data/processed/overall-stats
```

Outputs:

```text
data/processed/overall-stats/
├── database-summary.csv
├── database-summary.json
├── model-statistics.csv
├── temporal-statistics.csv
├── trait-usage-statistics.csv
└── journal-statistics.csv
```

Key metrics:

- Total unique papers (PMIDs)
- Total unique traits
- Number of extraction models
- Temporal coverage (year range)
- Average results per paper
- Per-model extraction counts
- Trait usage patterns (exposure vs outcome)
- Journal distribution

### Step 2: Trait profile statistics

Generate summary statistics for trait similarity profiles.

Command:

```bash
just analyze-trait-summary-stats
```

Direct script execution:

```bash
uv run scripts/analysis/analyze-trait-summary-stats.py \
  --trait-db ../data/db/trait_profile_db.db \
  --output-dir ../data/processed/trait-profiles/analysis
```

Outputs:

```text
data/processed/trait-profiles/analysis/
├── summary-stats-by-model.csv
├── similarity-distributions.csv
├── metric-correlations.csv
└── trait-count-distributions.csv
```

Key metrics:

- Total PMID-model combinations
- Total pairwise trait comparisons
- Semantic similarity distributions (mean, median, percentiles)
- Jaccard similarity distributions
- Correlation between semantic and Jaccard metrics
- Trait count statistics per study

### Step 3: Manuscript table generation

Consolidate all statistics into LaTeX-formatted tables.

Command:

```bash
just generate-manuscript-tables
```

Direct script execution:

```bash
uv run scripts/analysis/generate-manuscript-summary-table.py \
  --overall-dir ../data/processed/overall-stats \
  --trait-dir ../data/processed/trait-profiles/analysis \
  --evidence-dir ../data/processed/evidence-profiles/analysis \
  --output-dir ../data/processed/manuscript-tables
```

Prerequisites:

Must have run:

- `generate-overall-database-stats.py` (Step 1)
- `analyze-trait-summary-stats.py` (Step 2)
- `analyze-evidence-summary-stats.py` (from evidence profile pipeline)

Outputs:

```text
data/processed/manuscript-tables/
├── summary-table-full.tex                        # Aggregated across all models
├── summary-table-full-deepseek-r1-distilled.tex
├── summary-table-full-gpt-4-1.tex
├── summary-table-full-gpt-5.tex
├── summary-table-full-llama3.tex
├── summary-table-full-llama3-2.tex
├── summary-table-full-o4-mini.tex
└── summary-table-data.json
```

The script generates:

- **1 aggregated table** with statistics across all models
- **6 per-model tables** with model-specific statistics
- **1 JSON file** with all data for custom processing

## Output formats

### CSV files

All statistics are output as CSV files for easy import into R, Python,
or spreadsheet software.

Example structure (database-summary.csv):

```csv
total_unique_pmids,total_unique_traits,total_model_results,temporal_range_start,temporal_range_end,avg_results_per_pmid
15635,75121,50402,1990,2024,3.22
```

### JSON files

JSON format provides structured data for programmatic access.

Example structure (database-summary.json):

```json
{
  "total_unique_pmids": 15635,
  "total_unique_traits": 75121,
  "total_model_results": 50402,
  "temporal_range_start": 1990,
  "temporal_range_end": 2024,
  "avg_results_per_pmid": 3.22
}
```

### LaTeX tables

Two types of LaTeX tables are generated:

Aggregated version (summary-table-full.tex):

- Complete statistics from all three databases
- Aggregated across all models
- Three sections: Overall, Trait Profile, Evidence Profile
- Suitable for supplementary materials or appendices

Per-model versions (summary-table-full-{model}.tex):

- Model-specific statistics from all three databases
- One table per model (6 tables total)
- Same three-section structure as aggregated table
- Shows model-specific extraction counts, trait similarities, and evidence concordance
- Suitable for detailed model comparisons in appendices

Example LaTeX structure (per-model table):

```latex
\begin{table}[htbp]
\centering
\caption{MR-KG Summary Statistics - GPT-4-1 Model}
\label{tab:mrkg-summary-gpt-4-1}
\begin{tabular}{lr}
\hline
\multicolumn{2}{l}{\textbf{Model: GPT-4-1}} \\
\hline
Papers processed (PMIDs) & 15,626 \\
Total extraction results & 70,930 \\
Unique traits extracted & 80,761 \\
Average results per paper & 4.54 \\
\hline
\multicolumn{2}{l}{\textbf{Trait Profile Similarity}} \\
\hline
Total PMID-model combinations & 15,626 \\
Total pairwise comparisons & 156,260 \\
Semantic similarity (mean) & 0.734 \\
Semantic similarity (median) & 0.749 \\
Jaccard similarity (mean) & 0.088 \\
Jaccard similarity (median) & 0.042 \\
\hline
\multicolumn{2}{l}{\textbf{Evidence Profile Similarity}} \\
\hline
Total PMID-model combinations & 1,579 \\
Total pairwise comparisons & 2,066 \\
Direction concordance (mean) & 0.521 \\
Direction concordance (median) & 1.000 \\
Composite similarity (mean) & 0.419 \\
Composite similarity (median) & 0.385 \\
Data completeness (mean) & 0.649 \\
\hline
\end{tabular}
\end{table}
```

## Customization

### Adjusting table formats

The generated LaTeX tables include comments for customization:

```latex
% MR-KG Summary Statistics Table (Full Version)
% Generated by generate-manuscript-summary-table.py
% Customizable: adjust column widths, fonts, spacing as needed
```

Common customizations:

Change column alignment:

```latex
\begin{tabular}{lr}    % left-aligned, right-aligned
```

Add column separators:

```latex
\begin{tabular}{l|r}   % vertical line between columns
```

Adjust caption:

```latex
\caption{Your custom caption here}
```

Change label for cross-referencing:

```latex
\label{tab:your-custom-label}
```

### Using JSON data for custom tables

The `summary-table-data.json` file provides all raw data for creating
custom tables in any format:

```python
import json
import pandas as pd

with open('data/processed/manuscript-tables/summary-table-data.json') as f:
    data = json.load(f)

overall = data['overall']
trait_profile = data['trait_profile']
evidence_profile = data['evidence_profile']

df = pd.DataFrame({
    'Metric': ['Total Papers', 'Total Traits', ...],
    'Value': [overall['total_unique_pmids'], overall['total_unique_traits'], ...]
})
```

## Metrics reference

### Overall database metrics

| Metric                  | Description                               | Source Table        |
| ----------------------- | ----------------------------------------- | ------------------- |
| total_unique_pmids      | Unique papers in database                 | mr_pubmed_data      |
| total_unique_traits     | Unique trait labels                       | trait_embeddings    |
| total_model_results     | Model extraction records                  | model_results       |
| total_unique_models     | Number of extraction models               | model_results       |
| temporal_range_start    | Earliest publication year                 | mr_pubmed_data      |
| temporal_range_end      | Latest publication year                   | mr_pubmed_data      |
| avg_results_per_pmid    | Average model results per paper           | Computed aggregation|

### Trait profile metrics

| Metric                   | Description                                 | Range |
| ------------------------ | ------------------------------------------- | ----- |
| total_combinations       | PMID-model combinations                     | Count |
| total_similarity_pairs   | Pairwise trait comparisons                  | Count |
| mean_semantic_similarity | Average embedding-based similarity          | 0-1   |
| median_semantic_similarity| Median embedding-based similarity           | 0-1   |
| mean_jaccard_similarity  | Average set overlap                         | 0-1   |
| median_jaccard_similarity| Median set overlap                          | 0-1   |
| corr_semantic_jaccard    | Correlation between semantic and Jaccard    | -1 to 1|

### Evidence profile metrics

| Metric                      | Description                              | Range |
| --------------------------- | ---------------------------------------- | ----- |
| total_combinations          | PMID-model combinations with evidence    | Count |
| total_similarity_pairs      | Pairwise evidence comparisons            | Count |
| mean_direction_concordance  | Average agreement in effect directions   | -1 to 1|
| median_direction_concordance| Median direction agreement               | -1 to 1|
| mean_composite_direction    | Average weighted composite similarity    | 0-1   |
| median_composite_direction  | Median weighted composite similarity     | 0-1   |

## Interpretation guide

### Semantic vs Jaccard similarity

Semantic similarity (embedding-based):

- Captures conceptual relationships between traits
- Range: 0 (completely different) to 1 (identical)
- Example: "body mass index" and "obesity" have high semantic similarity

Jaccard similarity (set-based):

- Measures exact trait overlap
- Range: 0 (no overlap) to 1 (identical sets)
- Formula: |intersection| / |union|
- Example: Studies with 3 common traits out of 5 total have Jaccard = 0.6

### Direction concordance

Measures agreement in causal effect directions between studies:

- Range: -1 (complete disagreement) to +1 (complete agreement)
- Calculation: (agreeing pairs - disagreeing pairs) / total pairs
- Values near 0 indicate mixed or inconsistent evidence
- Negative values indicate contradictory findings

### Composite similarity

Weighted combination of multiple evidence similarity metrics:

- Range: 0 (no similarity) to 1 (perfect similarity)
- Incorporates direction, effect size, statistical consistency, evidence
  overlap
- Two variants: equal weights vs direction-weighted
- Higher values indicate more consistent evidence patterns

## Troubleshooting

### Missing input files

Error: "Overall statistics not found"

Solution:

```bash
# Run Step 1 first
just generate-overall-stats
```

Error: "Trait model stats not found"

Solution:

```bash
# Run Step 2 first
just analyze-trait-summary-stats
```

Error: "Evidence model stats not found"

Solution:

```bash
# Generate evidence profile statistics
just analyze-evidence-summary-stats
```

### Database not found

Error: "Vector store database not found"

Solution:

```bash
# Build vector store database
just build-main-db
```

Error: "Trait database not found"

Solution:

```bash
# Build trait profile database
just build-trait-profile-db
```

### Validation

Verify output completeness:

```bash
# Check overall stats
ls -lh data/processed/overall-stats/

# Check trait stats
ls -lh data/processed/trait-profiles/analysis/

# Check manuscript tables
ls -lh data/processed/manuscript-tables/
```

Expected file counts:

- overall-stats/: 6 files
- trait-profiles/analysis/: 4 files
- manuscript-tables/: 8 files (1 aggregated + 6 per-model + 1 JSON)

## Integration with manuscript

### LaTeX workflow

1. Generate tables:

```bash
just generate-manuscript-tables
```

2. Copy table files to manuscript directory:

```bash
# Copy aggregated table
cp data/processed/manuscript-tables/summary-table-full.tex \
   /path/to/manuscript/tables/

# Copy specific model tables
cp data/processed/manuscript-tables/summary-table-full-gpt-4-1.tex \
   /path/to/manuscript/tables/
```

3. Include in manuscript:

```latex
% Aggregated statistics
\input{tables/summary-table-full.tex}

% Model-specific statistics
\input{tables/summary-table-full-gpt-4-1.tex}
```

4. Reference in text:

```latex
% Reference aggregated table
As shown in Table \ref{tab:mrkg-summary-full}, the MR-KG database
contains 15,635 papers...

% Reference model-specific table
Model-specific performance (Table \ref{tab:mrkg-summary-gpt-4-1})
shows that GPT-4-1 extracted...
```

### Word/Google Docs workflow

1. Generate tables:

```bash
just generate-manuscript-tables
```

2. Open CSV files in Excel/Google Sheets
3. Format as needed
4. Copy and paste into document
5. Cite data from `summary-table-data.json` as needed

## Advanced usage

### Filtering by model

To generate statistics for specific models, modify the SQL queries in the
analysis scripts:

```python
query = """
SELECT ...
FROM model_results mr
WHERE mr.model = 'gpt-4-1'  -- Add model filter
...
"""
```

### Custom percentiles

To add custom percentile calculations, modify the distribution queries:

```python
query = """
SELECT
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY similarity) as p10,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY similarity) as p90,
    ...
"""
```

### Temporal stratification

To analyze statistics by time period:

```python
query = """
SELECT
    CASE
        WHEN publication_year < 2015 THEN 'Early'
        ELSE 'Recent'
    END as era,
    COUNT(*) as paper_count,
    ...
GROUP BY era
"""
```

## Related documentation

- Pipeline overview: @docs/processing/pipeline.md
- Database schema: @docs/processing/db-schema.md
- Trait similarity methodology: @docs/processing/trait-profile-similarity.md
- Evidence similarity methodology:
  @docs/processing/evidence-profile-similarity.md
- Data structure: @docs/DATA.md
