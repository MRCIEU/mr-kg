# MR-KG Development Guide

This guide covers project setup, development workflows, and deployment for
MR-KG (Mendelian Randomization Knowledge Graph), a system for processing and
exploring Mendelian Randomization studies through LLM-extracted trait
information and vector similarity search.

## Project overview

MR-KG consists of three main components:

- **API (FastAPI)**: RESTful backend providing programmatic access to MR data
- **Webapp (Streamlit)**: User-facing interface for interactive exploration
  (standalone, accesses DuckDB databases directly)
- **Processing pipeline**: ETL pipeline that creates DuckDB databases from raw
  LLM results and EFO ontology data

The webapp and API are independent services.
Both access the same DuckDB databases through a shared repository layer
(`src/common_funcs/common_funcs/repositories/`).

## Quick start

Clone the repository and set up the development environment:

```bash
git clone https://github.com/MRCIEU/mr-kg
cd mr-kg
just setup-dev
```

Start the web services using Docker (recommended):

```bash
# IMPORTANT: Run this from the project root, not from api/ or webapp/
just dev
```

This starts both the API and webapp services in Docker containers.

Access the services:

- Webapp: http://localhost:8501
- API docs: http://localhost:8000/api/docs

Note: The `just dev` command behavior depends on your current directory:

- **From project root**: Starts both API and webapp in Docker containers
- **From api/ directory**: Starts API only in local development mode
- **From webapp/ directory**: Starts webapp only in local development mode

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

For local development (without Docker), each service needs its own .env file:

API (.env in api/ directory):

```env
VECTOR_STORE_PATH=../data/db/vector_store.db
TRAIT_PROFILE_PATH=../data/db/trait_profile_db.db
EVIDENCE_PROFILE_PATH=../data/db/evidence_profile_db.db
DEFAULT_MODEL=gpt-5
LOG_LEVEL=DEBUG
```

Webapp (.env in webapp/ directory):

```env
VECTOR_STORE_PATH=../data/db/vector_store.db
TRAIT_PROFILE_PATH=../data/db/trait_profile_db.db
EVIDENCE_PROFILE_PATH=../data/db/evidence_profile_db.db
DEFAULT_MODEL=gpt-5
```

Note: The database paths in the .env files use `../` to reference the project
root from the api/ or webapp/ directories.

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

### Docker-based development (recommended)

The recommended approach is to use Docker Compose to run both services
together.
This handles all networking and dependencies automatically.

From the project root:

```bash
# Start both services
just dev

# View logs
just dev-logs

# Stop services
just dev-down
```

Services are accessible at:

- Webapp: http://localhost:8501
- API: http://localhost:8000
- API docs: http://localhost:8000/api/docs

### Local development (without Docker)

For local development, you can run the API and webapp independently.
The webapp accesses DuckDB databases directly and does not require the API.

API (for programmatic access):

```bash
cd api
uv sync
just dev
```

Webapp (standalone):

```bash
cd webapp
uv sync
just dev
```

Ensure the DuckDB databases exist at the expected paths before starting either
service.

### API development

The API is built with FastAPI and provides RESTful endpoints for accessing
study data, extraction results, and similarity metrics.

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

The webapp is built with Streamlit and accesses DuckDB databases directly
through the shared repository layer.

Running webapp tests:

```bash
cd webapp
just test
```

Key files:

- `webapp/app.py`: Main application entry point
- `webapp/pages/`: Streamlit page definitions
- `webapp/services/db_client.py`: Database client using common_funcs
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

Important: All Docker commands must be run from the project root directory,
not from within api/ or webapp/ subdirectories.

Development:

```bash
# From project root only
just dev              # Start development stack (API + webapp)
just dev-logs         # View development logs
just dev-down         # Stop development stack
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

- `VECTOR_STORE_PATH`: Path to vector store database
- `TRAIT_PROFILE_PATH`: Path to trait profile database
- `EVIDENCE_PROFILE_PATH`: Path to evidence profile database
- `DEFAULT_MODEL`: Default extraction model (default: gpt-5)

### Development variables

- `PYTHON_ENV`: Python environment mode (default: development)
- `HOT_RELOAD`: Enable hot reload for development (default: true)
- `LOG_FORMAT`: Log format (text/json)

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
|   +-- services/              # Database client
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

## Troubleshooting

### API returns "Database not found" errors

**Symptom**: API returns 500 errors with messages like "Vector store database
not found: data/db/vector_store.db"

**Cause**: The API is looking for databases in the wrong location due to
incorrect relative paths.

**Solution**:

1. Ensure you have a .env file in the api/ directory:

   ```bash
   cd api
   cat .env
   ```

   It should contain:

   ```env
   VECTOR_STORE_PATH=../data/db/vector_store.db
   TRAIT_PROFILE_PATH=../data/db/trait_profile_db.db
   EVIDENCE_PROFILE_PATH=../data/db/evidence_profile_db.db
   ```

2. If the .env file is missing or incorrect, run:

   ```bash
   # From project root
   just setup-dev
   ```

3. Verify databases exist at the project root:

   ```bash
   ls -la data/db/
   ```

4. Check API health:

   ```bash
   curl http://localhost:8000/api/health
   ```

   All databases should show `true`.

### Webapp shows "Database not found" error

**Symptom**: Webapp fails with "Database not found" or similar error.

**Cause**: The webapp cannot find the DuckDB databases.

**Solution**:

1. Ensure you have a .env file in the webapp/ directory:

   ```bash
   cd webapp
   cat .env
   ```

   It should contain:

   ```env
   VECTOR_STORE_PATH=../data/db/vector_store.db
   TRAIT_PROFILE_PATH=../data/db/trait_profile_db.db
   EVIDENCE_PROFILE_PATH=../data/db/evidence_profile_db.db
   ```

2. Verify databases exist at the project root:

   ```bash
   ls -la data/db/
   ```

3. If the .env file is missing or incorrect, run:

   ```bash
   # From project root
   just setup-dev
   ```

### Docker services fail to start

**Symptom**: `just dev` fails or containers exit immediately.

**Cause**: Missing environment files or database files.

**Solution**:

1. Run setup:

   ```bash
   just setup-dev
   ```

2. Ensure databases exist:

   ```bash
   ls -la data/db/
   ```

3. Check container logs:

   ```bash
   just dev-logs
   ```

4. Verify Docker Compose configuration:

   ```bash
   docker-compose config
   ```
