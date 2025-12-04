# MR-KG Webapp

Streamlit-based web interface for exploring Mendelian Randomization studies
through the MR-KG API.

## Quick start

### Docker (recommended)

Start both API and webapp services from the project root:

```bash
# IMPORTANT: Run from project root, not from webapp/ directory
cd /path/to/mr-kg
just dev
```

The webapp will be available at http://localhost:8501

### Local development

Important: The webapp requires the API to be running first.

Terminal 1 (start API first):

```bash
# From project root
cd api
uv sync
just dev
```

Terminal 2 (start webapp):

```bash
# From project root
cd webapp
uv sync
just dev
```

If the API is not running, the webapp will fail with a connection error:
"Connection refused [Errno 61]"

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

## API client

The webapp communicates with the API backend through `services/api_client.py`.

### Available functions

| Function | Description | Caching |
|----------|-------------|---------|
| `search_studies(q, trait, model, limit, offset)` | Search for studies | TTL 300s |
| `get_extraction(pmid, model)` | Get extraction results | None |
| `get_similar_by_trait(pmid, model, limit)` | Get trait-similar studies | None |
| `get_similar_by_evidence(pmid, model, limit)` | Get evidence-similar studies | None |
| `autocomplete_traits(q, limit)` | Get trait suggestions | TTL 300s |
| `autocomplete_studies(q, limit)` | Get study suggestions | TTL 300s |
| `get_statistics()` | Get resource statistics | TTL 3600s |
| `check_health()` | Check API health | None |

Similarity functions use session state for caching within the Study Info page
to avoid redundant API calls when panels are re-rendered.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | API backend URL |
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
|   +-- api_client.py         # API client functions
+-- tests/
|   +-- conftest.py           # Test fixtures
|   +-- test_api_client.py    # API client tests
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
- httpx: HTTP client for API communication
- pandas: Data handling for tables
- pydantic-settings: Configuration management

See `pyproject.toml` for the full dependency list.
