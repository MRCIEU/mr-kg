# MR-KG: Mendelian Randomization Knowledge Graph

A comprehensive fullstack system for processing, storing, and exploring Mendelian Randomization studies through large language model-extracted trait information and vector similarity search.

## System Overview

MR-KG transforms raw PubMed literature data extracted by large language models into an interactive knowledge graph platform with three main components:

- **ETL Processing Pipeline**: Converts LLM results into structured vector databases
- **FastAPI Backend**: RESTful API providing trait exploration and similarity search
- **Vue.js Frontend**: Modern web interface for interactive data exploration
- **Legacy Streamlit App**: Compatible visualization interface (maintained for transition)

## Key Features

- Vector similarity search across 50,000+ trait mentions from PubMed literature
- Study analysis with trait-outcome relationships from multiple LLM models
- Interactive exploration of trait similarities and study connections
- Real-time vector search for discovering related traits and studies
- Comprehensive API for programmatic access to the knowledge graph

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Built databases in `data/db/` directory (see [Processing Setup](#processing-setup))

### Development Environment

```bash
# Start the full development stack
just start

# Or step by step:
just setup-dev    # Setup environment files
just dev          # Start all services
```

Access the applications:
- **Frontend**: http://localhost:3000 (Vue.js interface)
- **Backend API**: http://localhost:8000 (FastAPI with docs at /docs)
- **Legacy Webapp**: http://localhost:8501 (Streamlit interface)

### Production Deployment

```bash
# Deploy production stack
just setup-prod
just prod
```

For detailed deployment instructions, see [DOCKER.md](DOCKER.md).

## Processing Setup

The system requires preprocessing raw LLM results into vector databases before using the web interfaces.

### Models Setup

```bash
cd models
wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
# Extract the model files
```

### Data Setup

```bash
# Download EFO ontology
wget https://github.com/EBISPOT/efo/releases/download/v3.80.0/efo.json
mv efo.json data/raw/
```

### Processing Pipeline

```bash
cd processing

# Setup environment
uv sync

# Run the full processing pipeline
just pipeline-full

# Or run individual steps:
just preprocess-traits preprocess-efo
just embed-traits embed-efo          # HPC batch jobs
just aggregate-embeddings
just build-main-db
just compute-trait-similarities      # HPC batch job
just build-trait-profile-db
```

For detailed processing instructions, see [processing/README.md](processing/README.md).

## Architecture

### Technology Stack

- **Backend**: FastAPI with DuckDB vector databases, Python/uv
- **Frontend**: Vue.js 3 with TypeScript, Pinia state management, Tailwind CSS
- **Legacy Interface**: Streamlit for existing visualization workflows
- **Databases**: DuckDB with vector similarity search capabilities
- **Infrastructure**: Docker containers with development/production configurations

### Component Structure

```
mr-kg/
├── backend/          # FastAPI REST API server
├── frontend/         # Vue.js TypeScript interface
├── webapp/           # Legacy Streamlit application
├── processing/       # ETL pipeline and data processing
├── data/            # Vector databases and processed data
└── src/common_funcs/ # Shared schemas and utilities
```

### Data Flow

1. **Raw Data**: PubMed abstracts + LLM-extracted trait relationships
2. **Processing Pipeline**: Trait extraction, embedding generation, database creation
3. **Vector Databases**: DuckDB stores with similarity search capabilities
4. **API Layer**: FastAPI provides RESTful access to processed data
5. **Web Interface**: Vue.js frontend for interactive exploration

## API Access

The FastAPI backend provides comprehensive REST endpoints:

- **Traits API**: Search and explore trait information
- **Studies API**: Access study metadata and relationships
- **Similarities API**: Vector similarity computation and ranking
- **System API**: Health checks and system information

Interactive API documentation: http://localhost:8000/docs

## Documentation

- **[DEV.md](DEV.md)**: Comprehensive development setup and guidelines
- **[DOCKER.md](DOCKER.md)**: Docker deployment and configuration
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and design patterns
- **[DATA.md](DATA.md)**: Data structure and database schema details
- **[backend/README.md](backend/README.md)**: Backend API development guide
- **[frontend/README.md](frontend/README.md)**: Frontend development guide
- **[processing/README.md](processing/README.md)**: ETL pipeline documentation

## Development Workflow

### Backend Development

```bash
cd backend
just dev              # Start development server
just test             # Run test suite
just check            # Code quality checks
```

### Frontend Development

```bash
cd frontend
just dev              # Start development server
just build            # Build for production
just check            # Run linting and type checking
```

### Full Stack Development

```bash
# Top-level commands for full stack
just dev              # Start all development services
just test-backend     # Run backend tests
just build-dev        # Build development images
just health           # Check service health
```

## Contributing

This system is designed for exploring Mendelian Randomization literature data. Key areas for contribution:

- **Data Processing**: Improve trait extraction and similarity algorithms
- **API Development**: Add new endpoints for specialized queries
- **Frontend Features**: Enhance visualization and interaction capabilities
- **Performance**: Optimize vector search and database operations

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## Citation

When using MR-KG in research, please cite the associated methodology papers and acknowledge the data sources:

- PubMed/MEDLINE for literature corpus
- EFO (Experimental Factor Ontology) for trait standardization
- Large language model providers for extraction services

## License

This project maintains compatibility with existing MR research infrastructure while providing modern fullstack capabilities for trait relationship exploration.
