# MR-KG Development Guide

This guide covers project setup, development workflows, and deployment for
MR-KG (Mendelian Randomization Knowledge Graph), a system for processing and
exploring Mendelian Randomization studies through LLM-extracted trait
information and vector similarity search.

## Project overview

MR-KG consists of three main components:

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

Start the web services:

```bash
just dev
```

Access the services:

- Webapp: http://localhost:8501
- API docs: http://localhost:8000/api/docs

## Prerequisites

- Python 3.12+ and the uv package manager
- just task runner
- Docker and Docker Compose

For the processing pipeline:

- Additional Python dependencies managed via uv
- Access to HPC resources for large-scale embedding jobs

## Setup

### Environment files

The `just setup-dev` command creates .env files from templates.
You can also create them manually:

Development (.env.development):

```env
WEBAPP_PORT=8501
API_PORT=8000
PYTHON_ENV=development
HOT_RELOAD=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text
```

Production (.env.production):

```env
WEBAPP_PORT=8501
API_PORT=8000
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Data prerequisites

MR-KG services expect local DuckDB databases:

- data/db/vector_store.db
- data/db/trait_profile_db.db
- data/db/evidence_profile_db.db

You can produce these via the processing pipeline.
See:

- Processing pipeline documentation: @docs/processing/pipeline.md
- Data structures and schema details: @docs/DATA.md

## Development workflows

### API development

The API is built with FastAPI and provides RESTful endpoints for accessing
study data, extraction results, and similarity metrics.

Docker-based development (recommended):

```bash
# Start both API and webapp in Docker
just dev
```

Local development:

```bash
cd api
uv sync
just dev
```

The API is accessible at http://localhost:8000 with documentation at
http://localhost:8000/api/docs

Running API tests:

```bash
cd api
just test
```

Key files:

- `api/app/main.py`: FastAPI application entry point
- `api/app/routers/`: API endpoint definitions
- `api/app/repositories/`: Database access layer
- `api/app/models.py`: Pydantic response models

### Webapp development

The webapp is built with Streamlit and communicates with the API backend.

Docker-based development (recommended):

```bash
# Start both API and webapp in Docker
just dev
```

Local development (requires API to be running):

```bash
cd webapp
uv sync
just dev
```

The webapp is accessible at http://localhost:8501

Running webapp tests:

```bash
cd webapp
just test
```

Key files:

- `webapp/app.py`: Main application entry point
- `webapp/pages/`: Streamlit page definitions
- `webapp/services/api_client.py`: API client
- `webapp/components/`: Reusable UI components

### Processing pipeline

The processing pipeline runs data transformations to create the databases.
See @docs/processing/pipeline.md for detailed documentation.

Key commands:

```bash
cd processing

# Run formatting
just fmt

# Run linting
just lint

# Run tests
just test

# Execute specific pipeline scripts
uv run python scripts/main-processing/<script_name>.py
```

For HPC batch jobs and embedding generation:

- See scripts/bc4/ for SLURM batch job templates
- See @docs/processing/trait-profile-similarity.md for trait similarity

### Common utilities

The src/common_funcs/ directory contains shared utilities used across
components.
See @src/common_funcs/README.md for details.

## Docker commands

All Docker commands use the top-level justfile:

Development:

```bash
just dev              # Start development stack (API + webapp)
just dev-logs         # View development logs
```

Production:

```bash
just prod             # Deploy production stack
just prod-logs        # View production logs
just prod-update      # Update production deployment
```

Build:

```bash
just build-dev        # Build development images
just build-prod       # Build production images
just build webapp     # Build specific service
```

Health and maintenance:

```bash
just health           # Check service health (API + webapp)
just status           # View container status
just clean            # Clean up Docker resources
just usage            # View Docker resource usage
```

Database:

```bash
just backup           # Create database backup
just list-backups     # List available backups
```

## Testing

### Unit tests

Each component has its own test suite:

```bash
# API tests
cd api && just test

# Webapp tests
cd webapp && just test

# Processing tests
cd processing && just test
```

### Integration tests

Integration tests verify the complete API workflow with real databases.
They require the API service to be running:

```bash
# Start services
just dev

# Run integration tests (from project root)
API_URL=http://localhost:8000 python -m pytest tests/integration -v
```

Integration tests are automatically skipped if the API is unavailable.

### Performance tests

Performance tests measure API response times against targets:

```bash
# Start services
just dev

# Run performance tests
python tests/performance/test_response_times.py
```

Performance targets:

- Autocomplete: <300ms
- Study search: <500ms
- Similarity queries: <1000ms

## Environment variables

### Docker Compose variables

- `WEBAPP_PORT`: Host port for webapp (default: 8501)
- `API_PORT`: Host port for API (default: 8000)

### API variables

- `VECTOR_STORE_PATH`: Path to vector store database
- `TRAIT_PROFILE_PATH`: Path to trait profile database
- `EVIDENCE_PROFILE_PATH`: Path to evidence profile database
- `DEFAULT_MODEL`: Default extraction model (default: gpt-5)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

### Webapp variables

- `API_URL`: API backend URL (default: http://localhost:8000)
- `DEFAULT_MODEL`: Default extraction model (default: gpt-5)

### Development variables

- `PYTHON_ENV`: Python environment mode (default: development)
- `HOT_RELOAD`: Enable hot reload for development (default: true)
- `LOG_FORMAT`: Log format (text/json)

## Deployment

### Docker deployment

The production stack uses Docker Compose with the docker-compose.prod.yml file.

Prerequisites:

- Docker Engine and Docker Compose installed
- Production environment file created (.env.production)
- Required databases present under data/db/

Deploy:

```bash
# Build and start production stack
just prod

# View logs
just prod-logs

# Update deployment
just prod-update

# Stop stack
just prod-down
```

Access the services:

- Webapp: http://localhost:8501
- API: http://localhost:8000

### Resource limits

Production containers have resource limits configured in docker-compose.prod.yml:

API:

- Memory: 512M limit, 256M reservation
- CPU: 0.5 limit, 0.25 reservation

Webapp:

- Memory: 1G limit, 512M reservation
- CPU: 0.5 limit, 0.25 reservation

Adjust these as needed for your workload.

### Health checks

Both services include health check endpoints:

- API: http://localhost:8000/api/health
- Webapp: http://localhost:8501/_stcore/health

Check service health:

```bash
just health
```

## Project structure

```text
mr-kg/
+-- README.md                  # Project overview
+-- DEV.md                     # This file
+-- api/                       # FastAPI backend service
|   +-- app/                   # Application code
|   |   +-- routers/           # API endpoints
|   |   +-- repositories/      # Database access
|   |   +-- main.py            # Application entry
|   |   +-- models.py          # Response models
|   +-- tests/                 # Unit tests
|   +-- Dockerfile
|   +-- justfile
|   +-- pyproject.toml
|   +-- README.md
+-- webapp/                    # Streamlit frontend
|   +-- pages/                 # Page definitions
|   +-- components/            # UI components
|   +-- services/              # API client
|   +-- tests/                 # Unit tests
|   +-- Dockerfile
|   +-- justfile
|   +-- pyproject.toml
|   +-- README.md
+-- processing/                # ETL processing pipeline
|   +-- scripts/               # Processing scripts
|   +-- README.md
+-- tests/                     # Integration and performance tests
|   +-- integration/           # Integration tests
|   +-- performance/           # Performance tests
+-- docs/
|   +-- DATA.md                # Data structure documentation
|   +-- GLOSSARY.md            # Key terms
|   +-- processing/            # Processing pipeline docs
+-- data/                      # Data files (gitignored)
|   +-- raw/                   # Raw input data
|   +-- processed/             # Processed data
|   +-- db/                    # DuckDB databases
+-- src/
|   +-- common_funcs/          # Shared utilities
+-- docker-compose.yml         # Development compose file
+-- docker-compose.prod.yml    # Production compose file
+-- justfile                   # Task runner commands
+-- .env.development           # Development environment
+-- .env.production            # Production environment
```

## Component documentation

- API: @api/README.md
- Webapp: @webapp/README.md
- Processing pipeline: @docs/processing/pipeline.md
- Common utilities: @src/common_funcs/README.md
- Data structure: @docs/DATA.md
- Key terms and concepts: @docs/GLOSSARY.md

## Data documentation

For detailed information about the data structure, database schema, and data
processing:

- Data overview: @docs/DATA.md
- Vector stores: @docs/processing/databases.md
- Database schema: @docs/processing/db-schema.md
- Trait profile similarity: @docs/processing/trait-profile-similarity.md
- Evidence profile similarity: @docs/processing/evidence-profile-similarity.md
