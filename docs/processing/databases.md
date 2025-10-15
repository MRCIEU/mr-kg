# Vector stores and database architecture

See @processing/README.md for complete pipeline overview and workflow.

## Overview

MR-KG uses DuckDB databases to store vectorized trait data, EFO ontology
embeddings, and model results. The databases support semantic similarity
search and trait profile analysis for Mendelian randomization studies.

## Database types

### Vector store database

**Location**: `data/db/vector_store.db`

**Purpose**: Primary database containing trait embeddings, EFO term
embeddings, model extraction results, and optimized views for similarity
search.

**Key capabilities**:
- Semantic search across traits using 200-dimensional embeddings
- EFO term mapping and similarity lookup
- Model result querying by PMID and extraction model
- Cosine similarity computation using DuckDB's built-in functions

**Schema reference**: See @docs/processing/db-schema.md for complete
table definitions, indexes, and views.

### Trait profile database

**Location**: `data/db/trait_profile_db.db`

**Purpose**: Stores precomputed trait-to-trait similarity scores and
study similarity profiles for network analysis.

**Key capabilities**:
- Fast lookup of similar studies based on trait profiles
- Model-specific similarity comparisons
- Both semantic and Jaccard similarity metrics
- Support for study network construction

**Concepts**: See @docs/processing/trait-similarity.md for details on
trait profile similarity methodology.

## Embedding model

**Model**: SciSpaCy `en_core_sci_lg` (v0.5.4)
**Dimensions**: 200
**Use cases**:
- Trait label vectorization
- EFO term vectorization
- Semantic similarity computation

The embedding model provides domain-specific scientific language
representations optimized for biomedical text.

## Database operations

### Building databases

```bash
just build-main-db
just build-trait-profile-db
```

See @processing/README.md for complete build workflow and prerequisites.

### Querying databases

Example using `query-database.py`:

```bash
cd processing

uv run python scripts/main-db/query-database.py \
    --database ../data/db/vector_store.db \
    --query-trait "coffee intake" \
    --limit 10
```

### Validation

Validate database schema:

```bash
uv run python scripts/main-db/validate-database.py \
    --database ../data/db/vector_store.db
```

### Schema documentation

Generate updated schema documentation:

```bash
just generate-schema-docs
```

## Data sources

See @docs/data.md for complete data structure documentation.
