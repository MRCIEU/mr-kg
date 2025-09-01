# Development Guidelines

This repository implements MR-KG (Mendelian Randomization Knowledge Graph), a system for processing and exploring structural PubMed literature data extracted by large language models. The project builds vector databases from LLM-extracted trait information and provides a Streamlit web interface for exploration.

## Architecture Overview

The system follows a multi-stage ETL pipeline:

1. Raw LLM results → Processing pipeline → DuckDB vector stores
2. Vector stores → Streamlit web application → Interactive exploration

Key technologies:

- DuckDB for vector storage and similarity search
- Streamlit for web interface
- HPC batch processing for embeddings computation
- Python with uv package management

## Repository Structure

```text
mr-kg/
├── data/                    # Data storage (raw, processed, databases)
│   ├── raw/                 # Source datasets (EFO, LLM results, PubMed)
│   ├── processed/           # Intermediate processing artifacts
│   ├── db/                  # DuckDB databases
│   └── assets/              # Schemas and reference files
├── processing/              # ETL pipeline scripts and workflows
├── webapp/                  # Streamlit web application
├── src/
│   ├── common_funcs/        # Shared schemas and database utilities
│   └── yiutils/             # General-purpose utilities (submodule)
└── models/                  # ML models (spaCy, embeddings)
```

## Core Components

### Processing Pipeline (`processing/`)

The ETL pipeline processes raw LLM results into queryable vector databases through several stages:

#### Main Processing Scripts

- `preprocess-traits.py`: Extracts unique trait labels from all models, creates trait indices, and links exposure/outcome traits in model results
- `preprocess-efo.py`: Processes EFO ontology JSON to extract term IDs and labels
- `embed-traits.py`: Generates embeddings for trait labels using spaCy models (HPC batch job)
- `embed-efo.py`: Generates embeddings for EFO terms (HPC batch job)
- `aggregate-embeddings.py`: Combines embedding results from HPC chunks into single files

#### Database Building Scripts

- `build-main-database.py`: Creates vector_store.db with trait/EFO embeddings and model results
- `compute-trait-similarity.py`: Computes pairwise trait similarities (HPC batch job)
- `build-trait-profile-database.py`: Creates trait_profile_db.db for similarity analysis

#### Key Logic Flow

1. Trait preprocessing extracts unique traits across all models and assigns canonical indices
2. EFO preprocessing extracts ontology terms for semantic mapping
3. Embedding generation creates vector representations using spaCy models
4. Database building combines embeddings with structured data for vector search
5. Trait similarity computation enables finding related traits and studies

#### HPC Integration

The pipeline uses SLURM batch jobs for computationally intensive tasks:

- `scripts/bc4/*.sbatch`: SLURM job definitions
- Environment variable `ACCOUNT_CODE` required for HPC submissions
- Results stored in `data/output/` with experiment IDs

### Web Application (`webapp/`)

Streamlit application for interactive exploration of the processed data.

#### Core Structure

- `app.py`: Main application entry point, handles profile configuration (local/docker)
- `pages/`: Multi-page application modules
  - `explore_results.py`: Study details and similar studies finder
  - `explore_traits.py`: Trait search and metadata exploration
  - `trait_similarities.py`: Trait similarity analysis and visualization
  - `about.py`: Project information and documentation

#### Key Functionality

- Database profile management (local paths vs. containerized paths)
- Cached database connections for performance
- Interactive filtering by models, traits, and similarity thresholds
- Real-time similarity search using vector embeddings

#### Database Integration

The webapp connects to two DuckDB databases:
- `vector_store.db`: Main data with embeddings and model results
- `trait_profile_db.db`: Precomputed similarity matrices and aggregations

### Common Functions (`src/common_funcs/`)

Shared library providing schema definitions and database utilities used across processing and webapp components.

#### Schema Modules

- `database_schema.py`: Complete database schema definitions with table structures, indexes, and views
- `raw_data_schema.py`: TypedDict definitions for raw LLM output format
- `processed_data_schema.py`: Schemas for processed data with trait linkings
- `efo_schema.py`: EFO ontology data structures
- `embedding_schema.py`: Vector embedding data formats
- `mr_pubmed_schema.py`: PubMed corpus data structures
- `trait_profile_schema.py`: Trait similarity and profile schemas

#### Database Utilities

- `database_utils/utils.py`: Connection management, path resolution
- `database_schema_utils.py`: Schema validation and reporting utilities

#### Key Data Flow

Raw LLM results → Schema validation → Trait linking → Vector embedding → Database storage → Web interface queries

### Utilities (`src/yiutils/`)

Git submodule providing general-purpose utilities:
- `project_utils.py`: Project root finding and path resolution
- `chunking.py`: Data chunking for batch processing
- `failsafe.py`: Error handling and retry mechanisms

## Data Architecture

### Data Flow Overview

1. Raw data ingestion: EFO ontology + LLM results + PubMed corpus
2. Preprocessing: Trait extraction, deduplication, indexing
3. Embedding generation: Vector representations using spaCy models
4. Database construction: Structured storage with vector search capabilities
5. Web interface: Interactive exploration and similarity search

### Key Databases

#### vector_store.db

- `trait_embeddings`: Trait vectors indexed by canonical trait indices
- `efo_embeddings`: EFO term vectors for semantic mapping
- `model_results`: Raw LLM outputs with metadata
- `model_result_traits`: Links between studies and extracted traits
- Views for PMIDs analysis and trait-based filtering

#### trait_profile_db.db

- `trait_similarities`: Precomputed pairwise trait similarity scores
- `trait_profiles`: Aggregated trait profiles for studies
- Views for similarity analysis and ranking

### Schema Validation

The system includes comprehensive schema validation:
- JSON Schema validation for raw data
- Database schema compliance checking
- Type safety through TypedDict definitions
- Validation reporting and error logging

## Development Workflows

### Setup Requirements

1. Python environment with uv package manager
2. HPC access with SLURM for embedding computation
3. Database dependencies (DuckDB)
4. Model downloads (spaCy, embeddings)

### Task Runner Commands

Both `processing/` and `webapp/` use justfile for task management:

#### Processing Commands
```bash
# Main processing workflow
just preprocess-traits preprocess-efo
just embed-traits embed-efo          # HPC batch jobs
just aggregate-embeddings

# Database building
just build-main-db
just compute-trait-similarities      # HPC batch job
just aggregate-trait-similarities
just build-trait-profile-db

# Development
just ruff                           # Format and lint
just ty                             # Type checking
just describe-db                    # Database inspection
```

#### Webapp Commands
```bash
just local-run                     # Local development
just docker-run                    # Docker deployment
just describe-db                   # Database inspection
just ruff                          # Format and lint
```

### Environment Configuration

Required environment variables:
- `ACCOUNT_CODE`: HPC account for SLURM submissions

### Code Quality

Both components use:
- `ruff`: Code formatting and linting
- `ty`: Type checking
- Comprehensive docstrings and type annotations
- Schema validation for data integrity

## Key Entry Points for Agents

### Understanding the System
1. Start with `DATA.md` for data structure overview
2. Review `processing/README.md` for pipeline workflow
3. Examine `src/common_funcs/common_funcs/schema/database_schema.py` for data model

### Important Scripts to Examine
- `processing/scripts/main-processing/preprocess-traits.py`: Core trait processing logic
- `processing/scripts/main-db/build-main-database.py`: Database construction
- `webapp/app.py`: Web application entry point
- `src/common_funcs/common_funcs/schema/`: All schema definitions

### Database Inspection
- Use `just describe-db` in either processing/ or webapp/ directories
- Outputs complete database schema to `data/assets/database_schema/database_info.txt`
- Includes table structures, row counts, and view definitions

### Configuration Files
- `processing/justfile`: ETL workflow commands
- `webapp/justfile`: Web application commands
- `docker-compose.yml`: Container deployment
- `.env`: Environment variables (HPC configuration)

### Development Patterns
- All scripts include `--dry-run` flags for safe testing
- Comprehensive logging with loguru
- Type-safe data processing with TypedDict schemas
- Modular design with clear separation of concerns
