# MR-KG Webapp

Streamlit-based web interface for exploring Mendelian Randomization studies
with direct DuckDB database access.

For project overview and setup instructions, see @../README.md and @../DEV.md.

## Quick start

### Docker (recommended)

Start the webapp from the project root:

```bash
# IMPORTANT: Run from project root, not from webapp/ directory
cd /path/to/mr-kg
just dev
```

The webapp will be available at http://localhost:8501

Note: The webapp and API services are independent.
The webapp accesses DuckDB databases directly and does not require the API.

### Local development

```bash
# From project root
cd webapp
uv sync
just dev
```

Ensure the DuckDB databases exist at the expected paths:

- `data/db/vector_store.db`
- `data/db/trait_profile_db.db`
- `data/db/evidence_profile_db.db`

## Pages

### Landing page (app.py)

The main entry point providing:

- Resource description and overview
- Key statistics display (studies, traits, models, extractions)
- Navigation buttons to search pages and info page
- External links to GitHub and documentation

### Search by trait (pages/1_Search_by_Trait.py)

Find studies investigating a specific trait:

- Trait autocomplete with prefix matching (min 2 characters)
- Model selector dropdown to filter by extraction model
- Results table showing matching studies
- Click-to-navigate to study details

A trait can appear as either an exposure or outcome in MR studies.
The search returns all studies containing the selected trait in either role.

### Search by study (pages/2_Search_by_Study.py)

Find studies by title or PMID:

- Study autocomplete with substring matching (min 2 characters)
- Displays PMID and truncated title in results
- Model selector for subsequent viewing
- Click-to-navigate to study details

### Study info (pages/3_Study_Info.py)

Comprehensive study information with four collapsible panels:

**Panel 1: Study details (always visible)**

- PMID with link to PubMed
- Title, publication year, journal
- Full abstract text

**Panel 2: Extraction results (collapsed by default)**

- Extraction model used
- Count of exposure-outcome pairs and traits
- Detailed results for each pair:
  - Exposure and outcome names
  - Effect sizes (beta, odds ratio, hazard ratio)
  - 95% confidence interval
  - P-value and direction
- Raw metadata JSON viewer (nested expander)

**Panel 3: Similar studies by trait profile (collapsed, lazy-loaded)**

- Query study trait count
- Table of similar studies with:
  - PMID (clickable to navigate)
  - Title
  - Semantic similarity (cosine similarity of trait embeddings)
  - Jaccard similarity (set overlap of traits)
  - Trait count

**Panel 4: Similar studies by evidence profile (collapsed, lazy-loaded)**

- Query study result count
- Table of similar studies with:
  - PMID (clickable to navigate)
  - Title
  - Direction concordance (color-coded: green/orange/red)
  - Matched pairs count

### Info page (pages/4_Info.py)

Resource statistics and documentation:

- Overall statistics (papers, traits, models, extractions)
- Model statistics for trait similarity
- Model statistics for evidence similarity
- List of available extraction models
- Methodology explanations for similarity metrics
- Links to detailed documentation

## Components

### Model selector (components/model_selector.py)

Dropdown widget for selecting the extraction model.

```python
from components.model_selector import model_selector, AVAILABLE_MODELS

selected_model = model_selector(key="my_selector")
```

Available models:

- deepseek-r1-distilled
- gpt-4-1
- gpt-4o
- gpt-5 (default)
- llama3
- llama3-2
- o4-mini

### Study table (components/study_table.py)

Displays study search results with selection capability.

```python
from components.study_table import study_table

selected_pmid = study_table(studies)  # Returns PMID or None
```

Displays: PMID, title (truncated), year, journal.
Each row has a "View" button that returns the selected PMID.

### Similarity display (components/similarity_display.py)

Two functions for displaying similarity results:

```python
from components.similarity_display import (
    trait_similarity_table,
    evidence_similarity_table,
)

# Trait similarity with semantic and Jaccard scores
selected = trait_similarity_table(similar_studies)

# Evidence similarity with direction concordance
selected = evidence_similarity_table(similar_studies)
```

Evidence similarity uses color coding for concordance values:

- Green: >= 0.5
- Orange: >= 0
- Red: < 0

## Database client

The webapp accesses DuckDB databases directly through `services/db_client.py`,
which uses the shared repository layer from `common_funcs`.

### Available functions

| Function | Description | Caching |
|----------|-------------|---------|
| `search_studies(q, trait, model, limit, offset)` | Search for studies | None |
| `get_extraction(pmid, model)` | Get extraction results | None |
| `get_similar_by_trait(pmid, model, limit)` | Get trait-similar studies | None |
| `get_similar_by_evidence(pmid, model, limit)` | Get evidence-similar studies | None |
| `autocomplete_traits(q, limit)` | Get trait suggestions | st.cache_data TTL 300s |
| `autocomplete_studies(q, limit)` | Get study suggestions | st.cache_data TTL 300s |
| `get_statistics()` | Get resource statistics | st.cache_data TTL 3600s |
| `check_database_health()` | Check database availability | None |

Similarity functions use session state for caching within the Study Info page
to avoid redundant database calls when panels are re-rendered.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_PATH` | `data/db/vector_store.db` | Path to vector store database |
| `TRAIT_PROFILE_PATH` | `data/db/trait_profile_db.db` | Path to trait profile database |
| `EVIDENCE_PROFILE_PATH` | `data/db/evidence_profile_db.db` | Path to evidence profile database |
| `DEFAULT_MODEL` | `gpt-5` | Default extraction model |

## Project structure

```text
webapp/
+-- .streamlit/
|   +-- config.toml           # Streamlit configuration
+-- pages/
|   +-- 1_Search_by_Trait.py  # Trait search page
|   +-- 2_Search_by_Study.py  # Study search page
|   +-- 3_Study_Info.py       # Study details page
|   +-- 4_Info.py             # Resource info page
+-- components/
|   +-- model_selector.py     # Model selection widget
|   +-- similarity_display.py # Similarity results display
|   +-- study_table.py        # Study list table
+-- services/
|   +-- db_client.py          # Database client functions
+-- tests/
|   +-- conftest.py           # Test fixtures
|   +-- test_db_client.py     # Database client tests (TODO)
+-- app.py                    # Main application entry
+-- config.py                 # Application configuration
+-- Dockerfile
+-- justfile
+-- pyproject.toml
```

## Development

### Running tests

```bash
just test
```

### Code formatting

```bash
just fmt
```

### Linting

```bash
just lint
```

## Dependencies

Key dependencies:

- Streamlit: Web framework
- DuckDB: Direct database access
- common_funcs: Shared repository layer (local dependency)
- pydantic-settings: Configuration management

See `pyproject.toml` for the full dependency list.
