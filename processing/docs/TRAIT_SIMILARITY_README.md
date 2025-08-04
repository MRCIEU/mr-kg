# Trait Profile Similarity Computation

This directory contains scripts for computing trait profile similarities between PMID-model combinations using parallel processing with SLURM job arrays.

## Overview

The trait profile similarity computation has been moved out of the main database build process to improve performance and enable parallel processing. Instead of computing all pairwise similarities during database creation (which would be ~26,165² = 684 million comparisons), we use a chunked approach that:

1. Processes PMID-model combinations in parallel chunks
2. Computes similarities only for assigned chunks
3. Keeps only the top-k most similar results for each combination
4. Aggregates results from all chunks

## Files

### Core Scripts

- `compute-trait-similarity.py`: Main computation script for processing chunks
- `aggregate-trait-similarities.py`: Combines results from all chunks
- `bc4/compute-trait-similarity.sbatch`: SLURM batch script for parallel execution

### Similarity Metrics

The computation uses two similarity metrics:

1. **Semantic Similarity**: Average of maximum cosine similarities between trait embeddings
   - Uses trait vector embeddings from the database
   - Computes similarity between trait profiles semantically
   - Range: 0.0 to 1.0

2. **Jaccard Similarity**: Set similarity of trait indices
   - Intersection over union of trait index sets
   - Measures overlap of exact traits mentioned
   - Range: 0.0 to 1.0

## Usage

### 1. Build the Database First

Ensure you have built the vector store database without similarity computation:

```bash
cd processing
uv run scripts/build-vector-store.py
```

### 2. Submit SLURM Job Array

Submit the parallel computation job:

```bash
cd processing/scripts/bc4
sbatch --account=<your-account> compute-trait-similarity.sbatch
```

The script is configured to:
- Use 20 array tasks (0-19)
- Process combinations in chunks
- Keep top 10 similar results per combination
- Use 16GB memory and 24-hour time limit per task

### 3. Monitor Progress

Check job status:

```bash
squeue -u $USER
```

Check output logs:

```bash
# Check SLURM output
ls ../data/output/slurm-*.out

# Check job-specific logs
ls ../data/output/bc4-<job-id>/logs/
```

### 4. Aggregate Results

Once all chunks complete, combine them:

```bash
cd processing
uv run scripts/aggregate-trait-similarities.py \
  --input-dir data/output/bc4-<job-id>/results \
  --expected-chunks 20
```

This creates:
- `data/processed/similarities/trait_similarities.json`: Combined results
- `data/processed/similarities/aggregation_stats.json`: Validation statistics

## Configuration

### Adjusting Parallelization

To change the number of chunks, modify both:

1. **SLURM script** (`compute-trait-similarity.sbatch`):
   ```bash
   #SBATCH --array=0-N  # Change N to desired number - 1
   ```

2. **Array length parameter**:
   ```bash
   --array-length N  # Change N to match array size
   ```

### Adjusting Top-K Results

Change the number of similar results kept per combination:

```bash
--top-k 10  # Default is 10, can be adjusted
```

### Memory and Time Limits

For larger datasets, you may need to adjust:

```bash
#SBATCH --mem=32G      # Increase memory if needed
#SBATCH --time=48:00:00 # Increase time limit if needed
```

## Testing

### Dry Run

Test the setup without processing:

```bash
# Test computation script
uv run scripts/compute-trait-similarity.py --dry-run

# Test aggregation script
uv run scripts/aggregate-trait-similarities.py \
  --input-dir /path/to/test/dir \
  --expected-chunks 5
```

### Local Testing

Test with a small chunk locally:

```bash
uv run scripts/compute-trait-similarity.py \
  --array-length 5 \
  --array-id 0 \
  --top-k 5 \
  --output-dir test_output
```

## Output Format

### Individual Chunk Files

Each chunk produces a file: `trait_similarities_chunk_<id>.json`

```json
[
  {
    "query_pmid": "12345678",
    "query_model": "gpt-4-turbo",
    "query_title": "Title of the paper",
    "query_trait_count": 3,
    "top_similarities": [
      {
        "similar_pmid": "87654321",
        "similar_model": "deepseek-r1",
        "similar_title": "Similar paper title",
        "trait_profile_similarity": 0.85,
        "trait_jaccard_similarity": 0.60,
        "query_trait_count": 3,
        "similar_trait_count": 4
      }
    ]
  }
]
```

### Aggregated Results

The final aggregated file contains all records from all chunks combined.

## Performance Notes

- **Chunk Size**: 20 chunks works well for ~26K combinations
- **Memory Usage**: ~16GB per task handles typical workloads
- **Processing Time**: ~1-24 hours per chunk depending on data size
- **Top-K Filtering**: Keeps memory usage manageable by limiting results

## Troubleshooting

### Common Issues

1. **Database Not Found**: Ensure database was built successfully
2. **Memory Errors**: Increase memory allocation in SLURM script
3. **Time Limits**: Increase time limit for larger chunks
4. **Missing Chunks**: Check failed jobs and resubmit specific array indices

### Resubmitting Failed Jobs

Resubmit specific array indices:

```bash
sbatch --account=<account> --array=5,7,12 compute-trait-similarity.sbatch
```

### Checking Results

Validate chunk completeness:

```bash
# Count chunk files
ls data/output/bc4-<job-id>/results/trait_similarities_chunk_*.json | wc -l

# Check for missing chunks
for i in {0..19}; do
  if [ ! -f "data/output/bc4-<job-id>/results/trait_similarities_chunk_${i}.json" ]; then
    echo "Missing chunk: $i"
  fi
done
```
