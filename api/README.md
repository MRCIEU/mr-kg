# MR-KG API

FastAPI backend service providing RESTful endpoints for accessing Mendelian
Randomization study data, extraction results, and similarity metrics.

## Quick start

### Docker (recommended)

From the project root:

```bash
just dev
```

The API will be available at http://localhost:8000

### Local development

```bash
cd api
uv sync
just dev
```

## API documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc
- OpenAPI spec: http://localhost:8000/api/openapi.json

## Endpoints overview

### Studies

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/studies` | GET | Search and list studies with optional filtering |
| `/api/studies/{pmid}/extraction` | GET | Get extraction results for a specific study |
| `/api/studies/autocomplete` | GET | Get study autocomplete suggestions |
| `/api/traits/autocomplete` | GET | Get trait autocomplete suggestions |
| `/api/statistics` | GET | Get resource-wide statistics |

### Similarity

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/studies/{pmid}/similar/trait` | GET | Get similar studies by trait profile |
| `/api/studies/{pmid}/similar/evidence` | GET | Get similar studies by evidence profile |

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with database status |

## Endpoint reference

### GET /api/studies

Search and list studies with optional filtering.

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `q` | string | No | - | - | Search query for title or PMID |
| `trait` | string | No | - | - | Filter by trait label |
| `model` | string | No | gpt-5 | - | Filter by extraction model |
| `limit` | int | No | 20 | 1-100 | Maximum results to return |
| `offset` | int | No | 0 | >= 0 | Pagination offset |

### GET /api/studies/{pmid}/extraction

Get extraction results for a specific study.
Returns 404 if the study is not found for the specified model.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pmid` | string | Yes | - | PubMed ID (path parameter) |
| `model` | string | No | gpt-5 | Extraction model |

### GET /api/studies/{pmid}/similar/trait

Get similar studies by trait profile similarity.
Returns studies ranked by semantic similarity of trait embeddings.

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `pmid` | string | Yes | - | - | PubMed ID (path parameter) |
| `model` | string | No | gpt-5 | - | Extraction model |
| `limit` | int | No | 10 | 1-50 | Maximum similar studies |

### GET /api/studies/{pmid}/similar/evidence

Get similar studies by evidence profile similarity.
Returns studies ranked by direction concordance of matched exposure-outcome pairs.

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `pmid` | string | Yes | - | - | PubMed ID (path parameter) |
| `model` | string | No | gpt-5 | - | Extraction model |
| `limit` | int | No | 10 | 1-50 | Maximum similar studies |

### GET /api/traits/autocomplete

Get trait autocomplete suggestions using prefix matching.

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `q` | string | Yes | - | min 2 chars | Search term for prefix match |
| `limit` | int | No | 20 | 1-50 | Maximum suggestions |

### GET /api/studies/autocomplete

Get study autocomplete suggestions using substring matching in titles.

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `q` | string | Yes | - | min 2 chars | Search term for substring match |
| `limit` | int | No | 20 | 1-50 | Maximum suggestions |

### GET /api/statistics

Get resource-wide statistics.
No parameters required.

### GET /api/health

Health check endpoint returning database connectivity status.
No parameters required.

## Response schemas

### StudiesResponse

```json
{
  "total": 100,
  "limit": 20,
  "offset": 0,
  "studies": [
    {
      "pmid": "12345678",
      "title": "Study title",
      "pub_date": "2024-01-15",
      "journal": "Journal Name",
      "model": "gpt-5"
    }
  ]
}
```

### ExtractionResponse

```json
{
  "pmid": "12345678",
  "model": "gpt-5",
  "title": "Study title",
  "pub_date": "2024-01-15",
  "journal": "Journal Name",
  "abstract": "Study abstract...",
  "traits": [
    {
      "trait_index": 0,
      "trait_label": "body mass index",
      "trait_id_in_result": "exposure"
    }
  ],
  "results": [
    {
      "exposure": "body mass index",
      "outcome": "type 2 diabetes",
      "beta": 0.25,
      "odds_ratio": 1.28,
      "hazard_ratio": null,
      "ci_lower": 1.15,
      "ci_upper": 1.42,
      "p_value": 0.001,
      "direction": "positive"
    }
  ],
  "metadata": {}
}
```

### TraitSimilarityResponse

```json
{
  "query_pmid": "12345678",
  "query_model": "gpt-5",
  "query_title": "Study title",
  "query_trait_count": 5,
  "similar_studies": [
    {
      "pmid": "87654321",
      "title": "Similar study title",
      "trait_profile_similarity": 0.85,
      "trait_jaccard_similarity": 0.60,
      "trait_count": 4
    }
  ]
}
```

### EvidenceSimilarityResponse

```json
{
  "query_pmid": "12345678",
  "query_model": "gpt-5",
  "query_title": "Study title",
  "query_result_count": 3,
  "similar_studies": [
    {
      "pmid": "87654321",
      "title": "Similar study title",
      "direction_concordance": 0.75,
      "matched_pairs": 2,
      "match_type_exact": true,
      "match_type_fuzzy": false,
      "match_type_efo": false
    }
  ]
}
```

### TraitAutocompleteResponse

```json
{
  "traits": ["body mass index", "blood pressure", "BMI"]
}
```

### StudyAutocompleteResponse

```json
{
  "studies": [
    {"pmid": "12345678", "title": "Study about BMI..."},
    {"pmid": "23456789", "title": "Another BMI study..."}
  ]
}
```

### StatisticsResponse

```json
{
  "overall": {
    "total_papers": 15635,
    "total_traits": 75121,
    "total_models": 7,
    "total_extractions": 50402
  },
  "model_similarity_stats": [
    {
      "model": "gpt-5",
      "extractions": 8400,
      "avg_traits": 5.2,
      "similarities": 84000
    }
  ],
  "model_evidence_stats": [
    {
      "model": "gpt-5",
      "extractions": 8400,
      "avg_results": 3.1,
      "similarities": 42000
    }
  ]
}
```

### HealthResponse

```json
{
  "status": "healthy",
  "databases": {
    "vector_store": true,
    "trait_profile": true,
    "evidence_profile": true
  }
}
```

## Project structure

```text
api/
+-- app/
|   +-- routers/
|   |   +-- studies.py      # Study endpoints
|   |   +-- similar.py      # Similarity endpoints
|   +-- repositories/
|   |   +-- vector_store.py      # Vector store database access
|   |   +-- trait_profile.py     # Trait profile database access
|   |   +-- evidence_profile.py  # Evidence profile database access
|   |   +-- statistics.py        # Statistics queries
|   +-- main.py             # Application entry point
|   +-- models.py           # Pydantic response models
|   +-- config.py           # Application configuration
|   +-- database.py         # Database connections
+-- tests/
|   +-- conftest.py         # Test fixtures
|   +-- test_studies.py     # Studies endpoint tests
|   +-- test_similar.py     # Similarity endpoint tests
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

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_PATH` | `data/db/vector_store.db` | Path to vector store database |
| `TRAIT_PROFILE_PATH` | `data/db/trait_profile_db.db` | Path to trait profile database |
| `EVIDENCE_PROFILE_PATH` | `data/db/evidence_profile_db.db` | Path to evidence profile database |
| `DEFAULT_MODEL` | `gpt-5` | Default extraction model |
| `LOG_LEVEL` | `INFO` | Logging level |

## Dependencies

Key dependencies:

- FastAPI: Web framework
- DuckDB: Database engine
- Pydantic: Data validation
- httpx: HTTP client (for testing)

See `pyproject.toml` for the full dependency list.
