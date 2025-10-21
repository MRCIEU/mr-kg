# MR-KG Development Guide

This guide covers project setup, development workflows, and deployment for
MR-KG (Mendelian Randomization Knowledge Graph), a system for processing and
exploring Mendelian Randomization studies through LLM-extracted trait
information and vector similarity search.

## Project overview

MR-KG consists of two main components:

- **Processing pipeline**: ETL pipeline that creates DuckDB databases from raw
  LLM results and EFO ontology data
- **Webapp**: Streamlit interface for exploring the processed data

## Quick start

Clone the repository and set up the development environment:

```bash
git clone https://github.com/MRCIEU/mr-kg
cd mr-kg
just setup-dev
```

Start the webapp:

```bash
just dev
```

Access the webapp at http://localhost:8501

## Prerequisites

- Python 3.12+ and the uv package manager
- just task runner
- Docker and Docker Compose

For the processing pipeline:

- Additional Python dependencies managed via uv
- Access to HPC resources for large-scale embedding jobs

## Setup

### Environment files

The `just setup-dev` command creates .env files from templates. You can also
create them manually:

Development (.env.development):

```env
WEBAPP_PORT=8501
PYTHON_ENV=development
HOT_RELOAD=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text
```

Production (.env.production):

```env
WEBAPP_PORT=8501
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Data prerequisites

MR-KG services expect local DuckDB databases:

- data/db/vector_store.db
- data/db/trait_profile_db.db

You can produce these via the processing pipeline. See:

- Processing pipeline documentation: @docs/processing/overview.md
- Data structures and schema details: @docs/DATA.md

## Development workflows

### Webapp development

Docker-based development (recommended):

```bash
# Start webapp in Docker
just dev
```

Local development:

```bash
cd webapp
just local-run
```

The webapp is accessible at http://localhost:8501

### Processing pipeline

The processing pipeline runs data transformations to create the databases.
See @docs/processing/overview.md for detailed documentation.

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
- See @docs/processing/trait-similarity.md for trait similarity computation

### Common utilities

The src/common_funcs/ directory contains shared utilities used across
components. See @src/common_funcs/README.md for details.

## Docker commands

All Docker commands use the top-level justfile:

Development:

```bash
just dev              # Start development stack
```

Production:

```bash
just prod             # Deploy production stack
```

Build:

```bash
just build-dev        # Build development images
just build-prod       # Build production images
just build webapp     # Build specific service
```

Health and maintenance:

```bash
just health           # Check service health
just status           # View container status
just clean            # Clean up Docker resources
just usage            # View Docker resource usage
```

Database:

```bash
just backup           # Create database backup
just list-backups     # List available backups
```

## Environment variables

### Docker Compose variables

- `WEBAPP_PORT`: Host port for webapp (default: 8501)

### Development variables

- `PYTHON_ENV`: Python environment mode (default: development)
- `HOT_RELOAD`: Enable hot reload for development (default: true)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `LOG_FORMAT`: Log format (text/json)

### Production variables

- `LOG_LEVEL`: Logging level (default: INFO)
- `LOG_FORMAT`: Log format (default: json)

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

Access the webapp at <http://localhost:8501>

### Resource limits

The production webapp container has the following resource limits:

- Memory: 1G limit, 512M reservation
- CPU: 0.5 limit, 0.25 reservation
- Logs: 100MB max size, 3 files

Adjust these in docker-compose.prod.yml as needed for your workload.

### Health checks

The webapp includes a health check endpoint:

- Endpoint: <http://localhost:8501/_stcore/health>
- Interval: 30s
- Timeout: 10s
- Retries: 3

Check service health:

```bash
just health
```

## Project structure

```text
mr-kg/
├── README.md                  # Project overview
├── DEV.md                     # This file
├── docs/
│   ├── DATA.md                # Data structure documentation
│   └── processing/            # Processing pipeline docs
│       ├── overview.md
│       ├── databases.md
│       ├── db-schema.md       # Auto-generated schema
│       └── trait-similarity.md
├── data/                      # Data files (gitignored)
│   ├── raw/                   # Raw input data
│   ├── processed/             # Processed data
│   └── db/                    # DuckDB databases
├── processing/                # ETL processing pipeline
│   ├── scripts/               # Processing scripts
│   └── README.md
├── webapp/                    # Streamlit webapp
│   └── README.md
├── src/
│   └── common_funcs/          # Shared utilities
├── docker-compose.yml         # Development compose file
├── docker-compose.prod.yml    # Production compose file
├── justfile                   # Task runner commands
└── .env.development           # Development environment template
```

## Component documentation

- Processing pipeline: @docs/processing/overview.md
- Webapp: @webapp/README.md
- Common utilities: @src/common_funcs/README.md
- Data structure: @docs/DATA.md

## Data documentation

For detailed information about the data structure, database schema, and data
processing:

- Data overview: @docs/DATA.md
- Vector stores: @docs/processing/databases.md
- Database schema: @docs/processing/db-schema.md
- Trait similarity: @docs/processing/trait-similarity.md
