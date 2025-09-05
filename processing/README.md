# MR-KG Processing Pipeline

Complete guide for the ETL processing pipeline that creates MR-KG databases from raw LLM results and EFO ontology data.

See @docs/SETTING-UP.md for initial repo setup.
See @docs/ENV.md for environment variables (ACCOUNT_CODE and others).

## Overview

The processing pipeline transforms raw inputs into vectorized databases consumed by the web stack:

```
Raw inputs -> Preprocessing -> Embedding -> Database build

- Raw inputs
  - LLM extraction results
  - EFO ontology JSON
  - PubMed metadata
- Preprocessing
  - Trait normalization and deduplication
  - EFO term parsing and linking
  - Index construction
- Embedding (HPC or batch)
  - spaCy-based vectorization of traits and EFO terms
  - Chunked jobs and aggregation of partial outputs
- Database build
  - Create DuckDB files and optimized tables
  - Materialize views for query performance
```

## Quick Start - Complete Pipeline

Run the complete processing pipeline with a single command:

```bash
just pipeline-full
```

This executes all steps in correct order, including HPC batch job submissions.

## Prerequisites

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

### Processing Environment

```bash
cd processing
uv sync               # Install dependencies
```

## Manual Step-by-Step Process

For more control or debugging, run individual steps:

### Main Processing

#### Preprocessing

Extract and normalize traits and EFO data:

```bash
just preprocess-traits    # Extract unique traits, create indices
just preprocess-efo       # Process EFO ontology
```

#### Embedding Generation

Generate embeddings for traits and EFO terms:

```bash
just embed-traits         # Generate trait embeddings (SLURM batch)
just embed-efo           # Generate EFO embeddings (SLURM batch)
```

#### Aggregate Embeddings

Combine HPC embedding results:

```bash
just aggregate-embeddings # Combine HPC results
```

### Database Building

Create the main vector store database:

```bash
just build-main-db        # Create vector_store.db
```

### Trait Profile Analysis

#### Compute Similarities

Generate trait-trait similarity matrices:

```bash
just compute-trait-similarities     # Compute similarities (SLURM batch)
```

#### Aggregate and Build Profile Database

Process similarity results and build final database:

```bash
just aggregate-trait-similarities   # Combine similarity results
just build-trait-profile-db         # Create trait_profile_db.db
```

## Key Processing Scripts

### Core Scripts

- `preprocess-traits.py`: Extracts unique trait labels from all models, creates trait indices
- `preprocess-efo.py`: Processes EFO ontology JSON to extract term IDs and labels
- `embed-traits.py`: Generates embeddings for trait labels using spaCy models
- `build-main-database.py`: Creates vector_store.db with trait/EFO embeddings and model results
- `compute-trait-similarity.py`: Computes pairwise trait similarities
- `build-trait-profile-database.py`: Creates trait_profile_db.db for similarity analysis

### Script Locations

```
processing/scripts/
├── main-processing/     # Core trait and EFO processing
│   ├── preprocess-traits.py
│   ├── preprocess-efo.py
│   ├── embed-traits.py
│   └── embed-efo.py
├── main-db/            # Database building scripts
│   ├── build-main-database.py
│   └── query-database.py
├── trait-profile/      # Similarity computation
│   ├── compute-trait-similarity.py
│   ├── aggregate-trait-similarities.py
│   └── build-trait-profile-database.py
└── bc4/               # HPC batch job scripts
    ├── compute-trait-similarity.sbatch
    ├── embed-efo.sbatch
    └── embed-traits.sbatch
```

## HPC Integration

The pipeline uses SLURM batch jobs for computationally intensive tasks:

### Requirements

- Environment variable `ACCOUNT_CODE` required for HPC submissions
- Results stored in `data/output/` with experiment IDs
- SLURM job definitions in `scripts/bc4/*.sbatch`

### HPC Workflow

1. **Trait Embedding**: Distributes trait embedding generation across HPC nodes
2. **EFO Embedding**: Generates EFO term embeddings in parallel
3. **Similarity Computation**: Computes pairwise trait similarities using HPC resources
4. **Result Aggregation**: Combines distributed results into final databases

### HPC Configuration

```bash
# Required environment variable
export ACCOUNT_CODE=your-hpc-account

# Batch job submission
sbatch scripts/bc4/embed-traits.sbatch
sbatch scripts/bc4/embed-efo.sbatch
sbatch scripts/bc4/compute-trait-similarity.sbatch
```

## Output Databases

### Vector Store Database (`data/db/vector_store.db`)

- `trait_embeddings`: Trait vectors indexed by canonical trait indices
- `efo_embeddings`: EFO term vectors for semantic mapping
- `model_results`: Raw LLM outputs with metadata
- `model_result_traits`: Links between studies and extracted traits
- `query_combinations`: PMID and model metadata
- Views for PMIDs analysis and trait-based filtering

### Trait Profile Database (`data/db/trait_profile_db.db`)

- `trait_similarities`: Precomputed pairwise trait similarity scores
- `trait_profiles`: Aggregated trait profiles for studies
- `similarity_views`: Optimized read views

## Development Tools

### Code Quality

```bash
just ruff                # Format and lint
just ty                  # Type checking
```

### Database Inspection

```bash
just describe-db         # Database inspection
# Outputs to data/assets/database_schema/database_info.txt
```

### Pipeline Status

```bash
# Check processing status and intermediate files
ls data/processed/       # Intermediate processing artifacts
ls data/output/          # HPC job outputs
ls data/db/             # Final database files
```

## Troubleshooting

### Common Issues

- **Missing models**: Ensure spaCy models are downloaded and extracted
- **HPC account**: Set `ACCOUNT_CODE` environment variable for batch jobs
- **Disk space**: Processing generates large intermediate files
- **Memory**: Embedding generation requires substantial RAM

### Debug Mode

Run individual scripts with increased verbosity:

```bash
cd processing
uv run python scripts/main-processing/preprocess-traits.py --verbose
uv run python scripts/main-db/build-main-database.py --debug
```

## Data Flow

```
Raw Data Sources:
├── data/raw/efo.json                    # EFO ontology
├── data/raw/llm_results/               # LLM extraction outputs
└── data/raw/pubmed/                    # PubMed metadata

Preprocessing:
├── data/processed/traits/              # Normalized traits
├── data/processed/efo/                 # Processed EFO terms
└── data/processed/indices/             # Trait indices

Embeddings:
├── data/output/embeddings/traits/      # Trait embeddings
└── data/output/embeddings/efo/         # EFO embeddings

Final Databases:
├── data/db/vector_store.db            # Main vector database
└── data/db/trait_profile_db.db        # Similarity analysis database
```

## Performance Considerations

### Resource Requirements

- **RAM**: 16GB+ recommended for embedding generation
- **Storage**: 100GB+ for intermediate files and final databases
- **CPU**: Multi-core beneficial for parallel processing
- **HPC**: SLURM environment for large-scale processing

### Optimization

- Use HPC batch jobs for embedding generation
- Process in chunks to manage memory usage
- Leverage DuckDB's columnar storage for query performance
- Precompute similarities to reduce runtime query complexity

## References

- Shared schema definitions: @src/common_funcs/common_funcs/schema/database_schema.py
- Environment configuration: @docs/ENV.md
- System architecture: @docs/ARCHITECTURE.md
- Data structure details: @docs/DATA.md
