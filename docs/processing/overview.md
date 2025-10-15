# Processing overview

ETL processing pipeline that creates MR-KG databases from raw LLM results and EFO ontology data.

## Purpose

The processing pipeline transforms raw inputs into vectorized databases consumed by the web stack:

1. Preprocessing: Normalize traits and EFO terms, create indices
2. Embedding: Generate 200-dim vectors using SciSpaCy (HPC)
3. Database build: Create vector_store.db with embeddings and results
4. Similarity: Compute trait profile similarities (HPC)
5. Profile database: Create trait_profile_db.db for network analysis

## Quick reference

For complete pipeline workflow and commands, see @processing/README.md.

## Architecture documentation

- Vector stores: @docs/processing/databases.md
- Database schema: @docs/processing/db-schema.md
- Trait similarity: @docs/processing/trait-similarity.md

## Key features

- SciSpaCy embeddings (200-dimensional vectors)
- DuckDB databases for efficient vector similarity search
- HPC integration for computationally intensive tasks
- Precomputed similarity matrices for fast queries
- Comprehensive schema validation

## Pipeline stages

### 1. Preprocessing

Extract and normalize traits and EFO data:
- preprocess-traits: Extract unique traits, create indices
- preprocess-efo: Process EFO ontology

### 2. Embedding generation

Generate embeddings for traits and EFO terms:
- embed-traits: Generate trait embeddings (SLURM batch)
- embed-efo: Generate EFO embeddings (SLURM batch)

### 3. Database building

Create the main vector store database:
- build-main-db: Create vector_store.db

### 4. Trait profile analysis

Generate trait-trait similarity matrices:
- compute-trait-similarities: Compute similarities (SLURM batch)
- aggregate-trait-similarities: Combine similarity results
- build-trait-profile-db: Create trait_profile_db.db

## Output databases

### Vector store database

Path: data/db/vector_store.db

Primary database containing trait embeddings, EFO embeddings, and model extraction results with optimized views for similarity search.

### Trait profile database

Path: data/db/trait_profile_db.db

Precomputed trait-to-trait similarities for study network analysis.

## HPC integration

The pipeline uses SLURM batch jobs for computationally intensive tasks:

- Trait embedding: Distributes trait embedding generation across HPC nodes
- EFO embedding: Generates EFO term embeddings in parallel
- Similarity computation: Computes pairwise trait similarities using HPC resources
- Result aggregation: Combines distributed results into final databases

Requirements:
- Environment variable ACCOUNT_CODE required for HPC submissions
- Results stored in data/output/ with experiment IDs
- SLURM job definitions in scripts/bc4/*.sbatch

## Development

See @processing/README.md for:
- Complete pipeline workflow
- Prerequisites and setup
- Manual step-by-step process
- HPC configuration
- Development tools
- Troubleshooting guide
