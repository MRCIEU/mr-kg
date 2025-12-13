# MR-KG: Mendelian Randomization Knowledge Graph

This repository implements MR-KG (Mendelian Randomization Knowledge Graph), a
system for processing and exploring Mendelian Randomization studies through
large language model-extracted trait information and vector similarity search.

## Components

- **API (FastAPI)**: RESTful backend providing programmatic access to MR data
- **Webapp (Streamlit)**: User-facing interface for interactive exploration
- **Processing pipeline**: ETL pipeline that creates DuckDB databases from raw
  LLM results and EFO ontology data

## Quick start

Clone the repository and set up the development environment:

```bash
git clone https://github.com/MRCIEU/mr-kg
cd mr-kg
just setup-dev
```

Start the web services using Docker:

```bash
# Run from project root
just dev
```

Access the services:

- Webapp: http://localhost:8501
- API documentation: http://localhost:8000/docs

For detailed development instructions including local development without
Docker, see @DEV.md

## Web services

MR-KG provides web services for:

- Searching and exploring MR studies by trait or study title
- Viewing detailed extraction results from multiple LLM models
- Discovering similar studies through trait profile and evidence profile
  similarity metrics
- Accessing resource-wide statistics

### API endpoints

| Endpoint                          | Description                           |
|-----------------------------------|---------------------------------------|
| GET /studies                  | Search and list studies               |
| GET /studies/{pmid}/extraction| Get extraction results for a study    |
| GET /studies/{pmid}/similar/trait | Find similar studies by trait     |
| GET /studies/{pmid}/similar/evidence | Find similar studies by evidence |
| GET /traits/autocomplete      | Trait name suggestions                |
| GET /studies/autocomplete     | Study title suggestions               |
| GET /statistics               | Resource-wide statistics              |
| GET /health                   | Service health check                  |

Full API documentation available at `/docs` when the service is running.

### Webapp pages

- **Search by Trait**: Find studies investigating specific traits
- **Search by Study**: Search studies by title
- **Study Info**: View extraction details and similar studies
- **Info**: Resource statistics and methodology documentation

## Project structure

```text
mr-kg/
+-- api/                    # FastAPI backend service
+-- webapp/                 # Streamlit frontend service
+-- processing/             # ETL processing pipeline
+-- data/                   # Data files (gitignored)
|   +-- db/                 # DuckDB databases
+-- docs/                   # Documentation
+-- src/                    # Shared utilities
+-- tests/                  # Integration and performance tests
+-- docker-compose.yml      # Development orchestration
+-- docker-compose.prod.yml # Production orchestration
+-- justfile                # Task runner commands
```

## Documentation

- Development guide: @DEV.md
- Data structure: @docs/DATA.md
- Key terms and concepts: @docs/GLOSSARY.md
- Processing pipeline: @docs/processing/pipeline.md
- Case study analyses: @docs/processing/case-studies.md
- API documentation: @api/README.md
- Webapp documentation: @webapp/README.md

## Citation

TBC
