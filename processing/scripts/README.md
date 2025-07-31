# Vector Store for Trait and EFO Similarity Search

This directory contains scripts for building and querying a DuckDB-based vector store that enables similarity search across traits from model results and EFO (Experimental Factor Ontology) terms.

## Overview

The vector store contains:
- **Trait embeddings**: 200-dimensional vectors for traits extracted from model results
- **EFO embeddings**: 200-dimensional vectors for EFO ontology terms  
- **Model results**: Links between traits in model results and their embedding indices
- **Similarity search capabilities**: Find similar traits and EFO terms using cosine similarity

## Scripts

### `build-vector-store.py`

Creates the DuckDB database with all embeddings and model results data.

**Usage:**
```bash
# Dry run to check files exist
python scripts/build-vector-store.py --dry-run

# Create database with automatic timestamp
python scripts/build-vector-store.py

# Create database with custom name  
python scripts/build-vector-store.py --database-name my-database
```

**Database Schema:**
- `trait_embeddings`: trait embeddings with id, label, and 200-dim vector
- `efo_embeddings`: EFO embeddings with id, label, and 200-dim vector  
- `model_results`: model result metadata (model, pmid)
- `model_traits`: trait occurrences in model results with linkings
- `trait_similarity_search`: view for trait-to-trait similarity
- `trait_efo_similarity_search`: view for trait-to-EFO similarity

### `query-vector-store.py`

Query the vector store for similarity searches and data exploration.

**Usage:**
```bash
# List all models and their statistics
python scripts/query-vector-store.py --database database-{timestamp}.db --list-models

# Find similar traits
python scripts/query-vector-store.py --database database-{timestamp}.db --query-trait "coffee intake" --limit 10

# Find similar EFO terms
python scripts/query-vector-store.py --database database-{timestamp}.db --query-efo "coffee intake" --limit 10

# Find trait by linked_index
python scripts/query-vector-store.py --database database-{timestamp}.db --trait-by-index 0

# List all available traits (first 50)
python scripts/query-vector-store.py --database database-{timestamp}.db --list-traits
```

## Example Workflow

1. **Build the database:**
   ```bash
   cd processing
   python scripts/build-vector-store.py
   ```

2. **Find similar traits:**
   ```bash
   python scripts/query-vector-store.py --database database-1753991544.db --query-trait "coffee intake" --limit 5
   ```
   
   Output:
   ```
   Top 5 most similar traits to 'coffee intake':
   Similarity  Trait ID      Trait Label
   0.9505      trait_30163   Decaffeinated coffee intake
   0.9451      trait_16552   caffeine intake from coffee
   0.9337      trait_16429   lower coffee intake
   0.9287      trait_691     coffee consumption
   0.9279      trait_5457    tea intake
   ```

3. **Find related EFO terms:**
   ```bash
   python scripts/query-vector-store.py --database database-1753991544.db --query-efo "coffee intake" --limit 5
   ```
   
   Output:
   ```
   Top 5 most similar EFO terms to 'coffee intake':
   Similarity  EFO ID                                      EFO Label
   0.9287      http://www.ebi.ac.uk/efo/EFO_0004330       coffee consumption
   0.8751      http://www.ebi.ac.uk/efo/EFO_0010097       sugar sweetened beverage consumption measurement
   0.8733      http://www.ebi.ac.uk/efo/EFO_0006781       coffee consumption measurement
   ```

## Data Sources

- **Trait embeddings**: `data/processed/embeddings/traits.json`
- **EFO embeddings**: `data/processed/embeddings/efo.json`  
- **Model results**: `data/processed/model_results/processed_model_results.json`

## Database Location

Databases are saved to: `data/db/database-{timestamp}.db`

## Dependencies

- DuckDB (>= 1.3.2)
- Common functions package
- Loguru for logging
- yiutils for project utilities

## Performance

The database includes indexes on commonly queried columns for optimal performance:
- Trait and EFO labels
- Model trait mappings
- Linked indices

Similarity searches use DuckDB's built-in `array_cosine_similarity` function for efficient vector comparisons.
