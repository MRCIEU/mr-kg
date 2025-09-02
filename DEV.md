# Development Guide

This repository implements MR-KG (Mendelian Randomization Knowledge Graph), a fullstack system for processing and exploring Mendelian Randomization studies through large language model-extracted trait information and vector similarity search.

## Architecture Overview

The system consists of four main components:

1. **ETL Processing Pipeline** → DuckDB vector stores
2. **FastAPI Backend** → RESTful API with vector similarity search
3. **Vue.js Frontend** → Modern interactive web interface
4. **Legacy Streamlit App** → Compatible visualization interface (maintained for transition)

### Technology Stack

- **Backend**: FastAPI, DuckDB, Python/uv, Pydantic
- **Frontend**: Vue.js 3, TypeScript, Pinia, Tailwind CSS, Vite
- **Legacy Interface**: Streamlit for existing workflows
- **Data Processing**: Python/uv, HPC batch processing, spaCy embeddings
- **Infrastructure**: Docker, Docker Compose, justfile task runners

### Docs

- `@DEV.md` (this document): Development setup and guidelines
- `@docs/DOCKER.md`: Docker deployment and configuration
- `@docs/ARCHITECTURE.md`: System architecture and design patterns
- `@docs/DATA.md`: Data structure and database schema details
- `@backend/README.md`: Backend API development guide
- `@frontend/README.md`: Frontend development guide
- `@processing/README.md`: ETL pipeline documentation

### Important Files to Examine

- `@src/common_funcs/common_funcs/schema/database_schema.py`: Complete data model
- `@backend/app/main.py`: FastAPI application entry point
- `@frontend/src/main.ts`: Vue.js application entry point
- `@processing/scripts/main-processing/preprocess-traits.py`: Core trait processing
- `@webapp/app.py`: Legacy Streamlit application

## Repository Structure

```text
mr-kg/
├── backend/                 # FastAPI REST API server
│   ├── app/                 # FastAPI application
│   │   ├── api/             # API route handlers (v1 endpoints)
│   │   ├── core/            # Core configuration, database, middleware
│   │   ├── models/          # Pydantic data models
│   │   ├── services/        # Business logic services
│   │   └── utils/           # Utility functions
│   ├── tests/               # Backend test suite
│   └── justfile             # Backend task runner
├── frontend/                # Vue.js TypeScript interface
│   ├── src/                 # Vue.js application source
│   │   ├── components/      # Reusable Vue components
│   │   ├── views/           # Page-level components
│   │   ├── stores/          # Pinia state management
│   │   ├── services/        # API service layer
│   │   ├── types/           # TypeScript definitions
│   │   └── router/          # Vue Router configuration
│   └── justfile             # Frontend task runner
├── webapp/                  # Legacy Streamlit application
│   ├── pages/               # Streamlit page modules
│   ├── resources/           # Database resources
│   └── app.py               # Streamlit entry point
├── processing/              # ETL pipeline scripts and workflows
│   ├── scripts/             # Processing pipeline scripts
│   │   ├── main-processing/ # Core trait and EFO processing
│   │   ├── main-db/         # Database building scripts
│   │   ├── trait-profile/   # Similarity computation
│   │   └── bc4/             # HPC batch job scripts
│   └── justfile             # Processing task runner
├── data/                    # Data storage (raw, processed, databases)
│   ├── raw/                 # Source datasets (EFO, LLM results, PubMed)
│   ├── processed/           # Intermediate processing artifacts
│   ├── db/                  # DuckDB databases
│   └── assets/              # Schemas and reference files
├── src/
│   ├── common_funcs/        # Shared schemas and database utilities
│   └── yiutils/             # General-purpose utilities (submodule)
└── models/                  # ML models (spaCy, embeddings)
```

## Quick Start

### Prerequisites

- Python 3.12+ with [uv](https://github.com/astral-sh/uv) package manager
- Node.js 18+ with npm/yarn
- [just](https://github.com/casey/just) task runner
- Docker and Docker Compose (for containerized development)

### Full Stack Development Setup

1. **Clone and setup the repository**:

   ```bash
   git clone https://github.com/MRCIEU/mr-kg
   cd mr-kg

   # Setup environment files
   just setup-dev
   ```

2. **Start the development stack**:

   ```bash
   # Start all services (backend, frontend, legacy webapp)
   just dev

   # Or individually:
   just backend-dev    # FastAPI backend only
   just frontend-dev   # Vue.js frontend only
   just webapp-dev     # Streamlit webapp only
   ```

3. **Access the applications**:

   - **Frontend**: <http://localhost:3000> (Vue.js interface)
   - **Backend API**: <http://localhost:8000> (FastAPI with docs at /docs)
   - **Legacy Webapp**: <http://localhost:8501> (Streamlit interface)

### Component-Specific Development

#### Backend Development Commands

```bash
cd backend

# Setup and development
just install          # Install dependencies
just env-setup        # Setup environment
just dev              # Start development server

# Code quality
just fmt              # Format code with ruff
just lint             # Lint code with ruff
just ty               # Type check with ty
just check            # Run all quality checks

# Testing
just test             # Run tests with pytest
just test-cov         # Run tests with coverage
```

#### Frontend Development Commands

```bash
cd frontend

# Setup and development
just install          # Install dependencies
just env-setup        # Setup environment
just dev              # Start development server

# Build
just build            # Build for production
just preview          # Preview production build

# Code quality
just format           # Format code with prettier
just lint             # Lint code with eslint
just type-check       # Type check with vue-tsc
just check            # Run all quality checks
```

#### Legacy Webapp Development Commands

```bash
cd webapp

# Development
just local-run        # Start local development
just docker-run       # Run in Docker container

# Code quality
just ruff             # Format and lint
```

## Data Processing Pipeline

The system requires preprocessing raw LLM results into vector databases before the web interfaces can be used.

### Setup Requirements

1. **Models setup**:

   ```bash
   cd models
   wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
   # Extract the model files
   ```

2. **Data setup**:

   ```bash
   # Download EFO ontology
   wget https://github.com/EBISPOT/efo/releases/download/v3.80.0/efo.json
   mv efo.json data/raw/
   ```

3. **Processing environment**:

   ```bash
   cd processing
   uv sync               # Install dependencies
   ```

### Processing Workflow

```bash
cd processing

# Full pipeline (includes HPC batch jobs)
just pipeline-full

# Or step by step:
# ---- Preprocessing ----
just preprocess-traits    # Extract unique traits, create indices
just preprocess-efo       # Process EFO ontology

# ---- Embedding Generation (HPC) ----
just embed-traits         # Generate trait embeddings (SLURM batch)
just embed-efo           # Generate EFO embeddings (SLURM batch)
just aggregate-embeddings # Combine HPC results

# ---- Database Building ----
just build-main-db        # Create vector_store.db

# ---- Similarity Analysis (HPC) ----
just compute-trait-similarities     # Compute similarities (SLURM batch)
just aggregate-trait-similarities   # Combine similarity results
just build-trait-profile-db         # Create trait_profile_db.db

# ---- Development Tools ----
just ruff                # Format and lint
just ty                  # Type checking
just describe-db         # Database inspection
```

### Key Processing Scripts

- `preprocess-traits.py`: Extracts unique trait labels from all models, creates trait indices
- `preprocess-efo.py`: Processes EFO ontology JSON to extract term IDs and labels
- `embed-traits.py`: Generates embeddings for trait labels using spaCy models
- `build-main-database.py`: Creates vector_store.db with trait/EFO embeddings and model results
- `compute-trait-similarity.py`: Computes pairwise trait similarities
- `build-trait-profile-database.py`: Creates trait_profile_db.db for similarity analysis

### HPC Integration

The pipeline uses SLURM batch jobs for computationally intensive tasks:

- Environment variable `ACCOUNT_CODE` required for HPC submissions
- Results stored in `data/output/` with experiment IDs
- `scripts/bc4/*.sbatch`: SLURM job definitions

## Development Workflows

### Full Stack Development

```bash
# Top-level commands for full stack
just dev              # Start all development services
just dev-down         # Stop development stack
just dev-logs         # View development logs

# Build and testing
just build-dev        # Build development images
just test-backend     # Run backend tests
just health           # Check service health
just status           # Docker container status
```

### Database Development

Both backend and webapp use DuckDB databases:

#### Vector Store Database (`vector_store.db`)

- `trait_embeddings`: Trait vectors indexed by canonical trait indices
- `efo_embeddings`: EFO term vectors for semantic mapping
- `model_results`: Raw LLM outputs with metadata
- `model_result_traits`: Links between studies and extracted traits
- Views for PMIDs analysis and trait-based filtering

#### Trait Profile Database (`trait_profile_db.db`)

- `trait_similarities`: Precomputed pairwise trait similarity scores
- `trait_profiles`: Aggregated trait profiles for studies
- Views for similarity analysis and ranking

#### Database Inspection

```bash
# Backend or processing directories
just describe-db      # Generate complete database schema
# Outputs to data/assets/database_schema/database_info.txt
```

### API Development

The FastAPI backend provides comprehensive REST endpoints:

#### API Structure

```text
/api/v1/
├── health/          # Health check endpoints
├── system/          # System information
├── core/           # Core utilities (ping, version, echo)
├── traits/         # Trait search and exploration
├── studies/        # Study metadata and relationships
└── similarities/   # Vector similarity computation
```

#### API Documentation

- **Interactive docs**: <http://localhost:8000/docs> (Swagger UI)
- **ReDoc**: <http://localhost:8000/redoc>
- **OpenAPI JSON**: <http://localhost:8000/openapi.json>

#### Testing API Endpoints

```bash
cd backend

# Test individual endpoints
just test-endpoints       # Test core endpoints
just test-db-health      # Test database health
just validate-schema     # Validate OpenAPI schema

# Development tools
just docs                # Open API documentation
```

### Frontend Development

The Vue.js frontend provides interactive interfaces for:

#### Pages and Features

- **Home**: Overview and navigation to main features
- **Traits**: Browse and search trait labels with filtering
- **Studies**: View study metadata and find similar studies
- **Similarities**: Analyze trait and study similarity relationships
- **About**: Project information and methodology

#### Development Patterns

- **Composition API**: Vue 3's modern composition API
- **TypeScript**: Full type safety throughout the application
- **Pinia**: State management for centralized application state
- **Vue Router**: Single-page application navigation
- **Tailwind CSS**: Utility-first CSS framework

#### State Management

- **Application Store**: Global UI state and configuration
- **Traits Store**: Trait data, filtering, and search state
- **Studies Store**: Study data and metadata management
- **Similarities Store**: Similarity analysis and results

## Code Quality and Standards

### Backend Standards

- **ruff**: Code formatting and linting
- **ty**: Type checking
- **pytest**: Testing framework with comprehensive coverage
- **Pydantic**: Type-safe data models and validation
- **FastAPI**: Automatic OpenAPI documentation

### Frontend Standards

- **ESLint**: JavaScript/TypeScript linting with Vue.js rules
- **Prettier**: Code formatting
- **Vue TSC**: TypeScript type checking
- **TypeScript Strict Mode**: Full type safety

### Processing Standards

- **ruff**: Code formatting and linting
- **ty**: Type checking
- **Comprehensive logging**: loguru for structured logging
- **Schema validation**: TypedDict for type-safe data processing

### Quality Checks

```bash
# Backend
cd backend && just check

# Frontend
cd frontend && just check

# Processing
cd processing && just ruff && just ty
```

## Docker Development

### Development Environment

```bash
# Quick start
just start            # Setup and start development environment

# Manual steps
just setup-dev        # Create environment files
just dev              # Start development stack
```

### Container Structure

- **Backend**: FastAPI with hot reload, volume mounting
- **Frontend**: Vite dev server with hot reload
- **Legacy Webapp**: Streamlit with Docker profile
- **Shared Volumes**: Database files, logs, development code

### Environment Profiles

- **Local Development**: Direct file system access
- **Docker Development**: Container-based with volume mounting
- **Production**: Optimized builds with security hardening

## Environment Configuration

### Required Environment Variables

#### Backend

```bash
DEBUG=true                    # Development mode
HOST=0.0.0.0
PORT=8000
DB_PROFILE=local              # or 'docker'
VECTOR_STORE_PATH=./data/db/vector_store.db
TRAIT_PROFILE_PATH=./data/db/trait_profile_db.db
```

#### Frontend

```bash
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_APP_TITLE=MR-KG Explorer
VITE_APP_DESCRIPTION=Mendelian Randomization Knowledge Graph
```

#### Processing

```bash
ACCOUNT_CODE=your-hpc-account  # Required for HPC submissions
```

## Integration Points

### Common Functions Integration

The `src/common_funcs/` module provides shared functionality:

- **Database Schemas**: Complete database schema definitions
- **Data Models**: TypedDict definitions for all data formats
- **Validation**: Schema validation and reporting utilities
- **Database Utils**: Connection management and path resolution

### Legacy Compatibility

- Maintains compatibility with existing Streamlit application
- No impact on processing pipeline workflows
- Uses existing database schemas without modification
- Works with current data organization structure
