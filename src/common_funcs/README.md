# Common Functions

Shared utility functions and data access layer for the MR-KG project.

For project overview, see @../../README.md and @../../DEV.md.

## Overview

This package provides:

- **Repositories**: Database access layer for DuckDB databases (vector_store.db, trait_profile_db.db, evidence_profile_db.db)
- **Schema**: Shared Pydantic models for data structures
- **Database utilities**: Connection management and helpers

The repositories provide a unified interface for both the API and webapp to access the three MR-KG databases.
See @../../docs/processing/databases.md for database architecture and @../../docs/processing/db-schema.md for schema details.

## Installation

This package is installed as a local editable dependency in both the API and
webapp:

```toml
# In pyproject.toml
[tool.uv.sources]
common_funcs = { path = "../src/common_funcs", editable = true }
```

## Repositories

The `repositories` module provides database access functions for all three
DuckDB databases:

### Vector store repository

Functions for searching studies and traits:

```python
from common_funcs.repositories import (
    search_traits,
    search_studies,
    get_studies,
    get_study_extraction,
    get_available_models,
)

# Search traits by prefix
traits = search_traits("blood", limit=10)

# Search studies by query
studies = search_studies(query="hypertension", model="gpt-5", limit=20)

# Get specific studies by PMIDs
studies = get_studies(pmids=["12345678", "87654321"], model="gpt-5")

# Get extraction results for a study
extraction = get_study_extraction(pmid="12345678", model="gpt-5")

# Get available extraction models
models = get_available_models()
```

### Trait profile repository

Functions for trait-based similarity:

```python
from common_funcs.repositories import get_similar_by_trait

# Get studies with similar trait profiles
similar = get_similar_by_trait(pmid="12345678", model="gpt-5", limit=10)
```

### Evidence profile repository

Functions for evidence-based similarity:

```python
from common_funcs.repositories import get_similar_by_evidence

# Get studies with similar evidence profiles
similar = get_similar_by_evidence(pmid="12345678", model="gpt-5", limit=10)
```

### Statistics repository

Functions for aggregate statistics:

```python
from common_funcs.repositories import (
    get_overall_statistics,
    get_model_similarity_stats,
    get_model_evidence_stats,
    get_metric_availability,
)

# Get overall resource statistics
stats = get_overall_statistics()

# Get trait similarity statistics per model
trait_stats = get_model_similarity_stats()

# Get evidence similarity statistics per model
evidence_stats = get_model_evidence_stats()

# Get metric availability
metrics = get_metric_availability()
```

## Configuration

Database paths are configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_PATH` | `data/db/vector_store.db` | Path to vector store |
| `TRAIT_PROFILE_PATH` | `data/db/trait_profile_db.db` | Path to trait profiles |
| `EVIDENCE_PROFILE_PATH` | `data/db/evidence_profile_db.db` | Path to evidence profiles |

Access configuration programmatically:

```python
from common_funcs.repositories.config import get_settings

settings = get_settings()
print(settings.vector_store_path)
```

## Connection management

The repositories use thread-safe connection management with automatic
connection pooling per database and per thread:

```python
from common_funcs.repositories import get_vector_store_connection

# Get a connection (thread-safe, cached per thread)
conn = get_vector_store_connection()
```

Connections are cached per thread and reused for efficiency.

## Return types

All repository functions return plain Python dictionaries (not Pydantic
models) for maximum flexibility.
The consuming applications (API, webapp) can convert to their own models as
needed.

## Development

### Running tests

```bash
cd src/common_funcs
just test
```

### Formatting and linting

```bash
just fmt
just lint
```
