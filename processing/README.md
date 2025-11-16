# MR-KG Processing Pipeline

ETL processing pipeline that creates MR-KG databases from raw LLM results and
EFO ontology data.

For complete pipeline documentation, see @docs/processing/pipeline.md.

## Overview

The processing pipeline transforms raw LLM extraction results and EFO ontology
data into queryable DuckDB databases through five main stages:

1. Preprocessing: Normalize traits and EFO terms, create indices
2. Embedding: Generate 200-dim vectors using SciSpacy (HPC)
3. Database build: Create vector_store.db with embeddings and results
4. Similarity: Compute trait profile similarities (HPC)
5. Profile databases: Create trait_profile_db.db and evidence_profile_db.db
   for network analysis of similarity between studies

## Setup

### Models

Download and extract SciSpacy model:

```bash
cd models
wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
tar -xzf en_core_sci_lg-0.5.4.tar.gz
```

### Data

Download EFO ontology:

```bash
wget https://github.com/EBISPOT/efo/releases/download/v3.80.0/efo.json
mv efo.json data/raw/efo/
```

### Environment

Install dependencies:

```bash
cd processing
uv sync
```

### HPC configuration

For HPC batch job submissions:

```bash
export ACCOUNT_CODE=your-hpc-account
```

## Quick start

Run complete pipeline:

```bash
just pipeline-full
```

## Development tools

### Code quality

```bash
just ruff
just ty
```

### Database documentation

```bash
just generate-schema-docs
```

Generates comprehensive schema docs with ERD diagrams at
docs/processing/db-schema.md (vector_store, trait_profile, evidence_profile).
Includes: table schemas, relationships, statistics, DuckDB versions, indexes.

### Summary statistics

Generate manuscript-ready summary statistics and LaTeX tables:

```bash
just generate-all-summary-stats
```

Consolidates statistics from all three databases (vector_store,
trait_profile, evidence_profile) into CSV files and LaTeX-formatted tables.
See @docs/processing/summary-statistics.md for complete documentation.

Individual steps:

```bash
just generate-overall-stats           # Overall database statistics
just analyze-trait-summary-stats      # Trait profile metrics
just generate-manuscript-tables       # LaTeX table generation
```

For manual step-by-step execution and detailed documentation, see
@docs/processing/pipeline.md.
