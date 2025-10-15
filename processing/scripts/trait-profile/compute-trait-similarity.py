"""Compute trait profile similarities between the extracted data of two studies
(i.e. PMID-model combinations).

The "trait profile" here refers to the list of extracted
exposure and outcome traits of a study.
And the trait profile similarity is then the similarity of two studies based on
the topics of their researched traits as represented by the trait profiles.

This script processes PMID-model combinations to compute trait profile similarities:
1. Loads PMID-model combinations from the database
2. Computes pairwise trait profile similarities using embeddings within the same model
3. Keeps only top 10 most similar results for each combination (from same model only)
4. Processes data in chunks for parallel processing using SLURM job arrays
5. Outputs similarity data for downstream analysis

The similarity computation uses two metrics:
- Semantic similarity: Average of maximum cosine similarities between trait embeddings
- Jaccard similarity: Set similarity of trait indices (intersection over union)

IMPORTANT: For each query combination, similarities are only computed with other
combinations from the SAME model (e.g., gpt-4-1 results are compared only with other gpt-4-1 results).

NOTE: This should be run as part of a HPC array job.
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
from loguru import logger
from tqdm import tqdm
from yiutils.chunking import calculate_chunk_start_end
from yiutils.project_utils import find_project_root

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments with dry_run, array_length, and array_id options
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ---- --dry-run ----
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually processing",
    )

    # ---- --array-length ----
    parser.add_argument(
        "--array-length",
        type=int,
        default=10,
        help="Total number of array chunks for parallel processing",
    )

    # ---- --array-id ----
    parser.add_argument(
        "--array-id",
        type=int,
        default=0,
        help="Current array chunk ID (0-based indexing)",
    )

    # ---- --output-dir ----
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for similarity files",
    )

    # ---- --database-path ----
    parser.add_argument(
        "--database-path",
        type=str,
        help="Path to the DuckDB database file",
    )

    # ---- --top-k ----
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top similar results to keep for each combination",
    )

    # ---- --workers ----
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for multiprocessing (default: 4)",
    )

    return parser.parse_args()


def find_latest_database() -> Path:
    """Find the most recent database file in the data/db directory.

    Returns:
        Path to the most recent database file
    """
    db_dir = DATA_DIR / "db"
    if not db_dir.exists():
        raise FileNotFoundError(f"Database directory not found: {db_dir}")

    db_files = list(db_dir.glob("database-*.db"))
    if not db_files:
        raise FileNotFoundError(f"No database files found in: {db_dir}")

    # Sort by modification time and return the most recent
    latest_db = max(db_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using latest database: {latest_db}")
    return latest_db


def load_pmid_model_combinations(
    conn: duckdb.DuckDBPyConnection,
) -> List[Tuple]:
    """Load all PMID-model combinations with their traits from the database.

    Args:
        conn: DuckDB connection

    Returns:
        List of tuples containing (pmid, model, title, traits)
    """
    logger.info("Loading PMID-model combinations from database...")

    result = conn.execute("""
        SELECT pmid, model, title, traits
        FROM pmid_model_analysis
        WHERE pmid IS NOT NULL
        ORDER BY pmid, model
    """).fetchall()

    logger.info(f"Loaded {len(result)} PMID-model combinations")
    return result


def compute_trait_profile_similarity(
    conn: duckdb.DuckDBPyConnection,
    query_traits: List[Dict],
    similar_traits: List[Dict],
) -> float:
    """Compute semantic similarity between two trait profiles.

    Args:
        conn: DuckDB connection
        query_traits: List of trait dictionaries for query
        similar_traits: List of trait dictionaries for comparison

    Returns:
        Average maximum cosine similarity between trait embeddings
    """
    if not query_traits or not similar_traits:
        return 0.0

    query_trait_indices = [t["trait_index"] for t in query_traits]
    similar_trait_indices = [t["trait_index"] for t in similar_traits]

    if not query_trait_indices or not similar_trait_indices:
        return 0.0

    # Use SQL to compute average maximum similarity
    query_placeholders = ",".join("?" * len(query_trait_indices))
    similar_placeholders = ",".join("?" * len(similar_trait_indices))

    sim_result = conn.execute(
        f"""
        SELECT AVG(max_sim) FROM (
            SELECT MAX(array_cosine_similarity(te1.vector, te2.vector)) as max_sim
            FROM trait_embeddings te1
            CROSS JOIN trait_embeddings te2
            WHERE te1.trait_index IN ({query_placeholders})
              AND te2.trait_index IN ({similar_placeholders})
            GROUP BY te1.trait_index
        )
    """,
        query_trait_indices + similar_trait_indices,
    ).fetchone()

    return sim_result[0] if sim_result and sim_result[0] else 0.0


def compute_jaccard_similarity(
    query_traits: List[Dict], similar_traits: List[Dict]
) -> float:
    """Compute Jaccard similarity between two trait sets.

    Args:
        query_traits: List of trait dictionaries for query
        similar_traits: List of trait dictionaries for comparison

    Returns:
        Jaccard similarity coefficient (intersection over union)
    """
    if not query_traits and not similar_traits:
        return 1.0  # Both empty

    if not query_traits or not similar_traits:
        return 0.0  # One empty

    query_set = set(t["trait_index"] for t in query_traits)
    similar_set = set(t["trait_index"] for t in similar_traits)

    intersection = len(query_set & similar_set)
    union = len(query_set | similar_set)

    return intersection / union if union > 0 else 0.0


def compute_similarities_for_single_query(args_tuple) -> Dict:
    """Worker function to compute similarities for a single query combination.

    This function is designed to work with multiprocessing.Pool.
    Only compares with results from the same model as the query.

    Args:
        args_tuple: Tuple containing (db_path, query_data, all_pmid_models, top_k)
            - db_path: Path to the database file
            - query_data: Tuple of (query_pmid, query_model, query_title, query_traits)
            - all_pmid_models: List of all PMID-model combinations (will be filtered by model)
            - top_k: Number of top similar results to keep

    Returns:
        Dictionary containing similarity results for the query (only from same model)
    """
    db_path, query_data, all_pmid_models, top_k = args_tuple
    query_pmid, query_model, query_title, query_traits = query_data

    # Create a new database connection for this worker
    with duckdb.connect(str(db_path), read_only=True) as conn:
        similarities = []

        # Filter to only compare with results from the same model
        same_model_combinations = [
            (pmid, model, title, traits)
            for pmid, model, title, traits in all_pmid_models
            if model == query_model
        ]

        for (
            similar_pmid,
            similar_model,
            similar_title,
            similar_traits,
        ) in same_model_combinations:
            # Skip self-comparison
            if query_pmid == similar_pmid and query_model == similar_model:
                continue

            # Compute trait profile similarity
            trait_profile_sim = compute_trait_profile_similarity(
                conn, query_traits, similar_traits
            )

            # Compute Jaccard similarity
            jaccard_sim = compute_jaccard_similarity(
                query_traits, similar_traits
            )

            similarities.append(
                {
                    "similar_pmid": similar_pmid,
                    "similar_model": similar_model,
                    "similar_title": similar_title,
                    "trait_profile_similarity": trait_profile_sim,
                    "trait_jaccard_similarity": jaccard_sim,
                    "query_trait_count": len(query_traits)
                    if query_traits
                    else 0,
                    "similar_trait_count": len(similar_traits)
                    if similar_traits
                    else 0,
                }
            )

        # Sort by trait profile similarity and keep top-k
        similarities.sort(
            key=lambda x: x["trait_profile_similarity"], reverse=True
        )
        top_similarities = similarities[:top_k]

        # Create record for this query
        return {
            "query_pmid": query_pmid,
            "query_model": query_model,
            "query_title": query_title,
            "query_trait_count": len(query_traits) if query_traits else 0,
            "top_similarities": top_similarities,
        }


def compute_similarities_for_chunk(
    db_path: Path,
    pmid_models_chunk: List[Tuple],
    all_pmid_models: List[Tuple],
    top_k: int = 10,
    workers: int = 4,
) -> List[Dict]:
    """Compute similarities for a chunk of PMID-model combinations using multiprocessing.

    For each query combination, only compares with results from the same model.

    Args:
        db_path: Path to the DuckDB database file
        pmid_models_chunk: Chunk of query PMID-model combinations
        all_pmid_models: All PMID-model combinations for comparison (filtered by model in workers)
        top_k: Number of top similar results to keep
        workers: Number of worker processes to use

    Returns:
        List of similarity records for the chunk (each containing same-model similarities only)
    """
    # Prepare arguments for each worker
    worker_args = [
        (db_path, query_data, all_pmid_models, top_k)
        for query_data in pmid_models_chunk
    ]

    logger.info(
        f"Starting multiprocessing with {workers} workers for {len(pmid_models_chunk)} queries"
    )

    # Use multiprocessing to compute similarities in parallel
    with multiprocessing.Pool(processes=workers) as pool:
        # Use imap for progress tracking
        similarity_records = list(
            tqdm(
                pool.imap(compute_similarities_for_single_query, worker_args),
                total=len(worker_args),
                desc="Computing similarities",
            )
        )

    logger.info(
        f"Completed similarity computation for {len(similarity_records)} queries"
    )
    return similarity_records


def main():
    """Main function to compute trait profile similarities.

    This function:
    1. Loads PMID-model combinations from the database
    2. Processes them in chunks for parallel processing
    3. Computes trait profile similarities for the assigned chunk
    4. Keeps only the top-k most similar results for each combination
    """
    # Parse command line arguments
    args = make_args()

    logger.info("Checking file paths and basic setup...")

    # Determine database path
    if args.database_path:
        db_path = Path(args.database_path)
    else:
        db_path = find_latest_database()

    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        return 1

    logger.info(f"✓ Database path exists: {db_path}")

    # Check if this is a dry run
    if args.dry_run:
        logger.info("Dry run completed. Exiting without processing.")
        return 0

    # Connect to database (read-only, allow multiple connections)
    logger.info("Connecting to database...")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        logger.info("Database connected successfully.")

        # Load all PMID-model combinations
        all_pmid_models = load_pmid_model_combinations(conn)

        # Process combinations into chunks
        total_combinations = len(all_pmid_models)
        start_idx, end_idx = calculate_chunk_start_end(
            chunk_id=args.array_id,
            num_chunks=args.array_length,
            data_length=total_combinations,
        )

        if start_idx is None or end_idx is None:
            logger.warning(
                f"Chunk {args.array_id} is out of range. No combinations to process."
            )
            return 0

        logger.info(
            f"Processing chunk {args.array_id}/{args.array_length}: "
            f"combinations [{start_idx} to {end_idx}) (total: {end_idx - start_idx})"
        )

        # Extract chunk for processing
        pmid_models_chunk = all_pmid_models[start_idx:end_idx]

        logger.info(
            f"Chunk contains {len(pmid_models_chunk)} combinations to process."
        )
        logger.info(
            f"Using {args.workers} worker processes for multiprocessing"
        )

        # Compute similarities for this chunk
        start_time = time.time()
        similarity_records = compute_similarities_for_chunk(
            db_path,
            pmid_models_chunk,
            all_pmid_models,
            args.top_k,
            args.workers,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Similarity computation completed in {processing_time:.2f} seconds"
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"trait_similarities_chunk_{args.array_id}.json"
        )

        logger.info(
            f"Writing {len(similarity_records)} records to: {output_path}"
        )
        with output_path.open("w") as f:
            json.dump(similarity_records, f, indent=2)

        logger.info("Processing completed successfully!")

    return 0


if __name__ == "__main__":
    exit(main())
