"""Aggregate trait similarity results from SLURM job array chunks.

This script combines the output from multiple trait similarity computation chunks:
1. Loads all chunk files from the specified directory
2. Combines them into a single comprehensive dataset
3. Validates the completeness of the results
4. Outputs the aggregated data for downstream analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from loguru import logger
from yiutils.project_utils import find_project_root

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing chunk files from SLURM job array",
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for aggregated results (default: data/processed/similarities/trait_similarities.json)",
    )
    
    parser.add_argument(
        "--expected-chunks",
        type=int,
        help="Expected number of chunk files (for validation)",
    )
    
    return parser.parse_args()


def load_chunk_files(input_dir: Path) -> List[Dict]:
    """Load all trait similarity chunk files from the input directory.
    
    Args:
        input_dir: Directory containing chunk files
        
    Returns:
        List of all similarity records from all chunks
    """
    chunk_files = list(input_dir.glob("trait_similarities_chunk_*.json"))
    
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in: {input_dir}")
    
    logger.info(f"Found {len(chunk_files)} chunk files")
    
    all_records = []
    
    for chunk_file in sorted(chunk_files):
        logger.info(f"Loading chunk file: {chunk_file.name}")
        
        with chunk_file.open("r") as f:
            chunk_data = json.load(f)
        
        logger.info(f"  - Loaded {len(chunk_data)} records from {chunk_file.name}")
        all_records.extend(chunk_data)
    
    logger.info(f"Total records loaded: {len(all_records)}")
    return all_records


def validate_results(records: List[Dict], expected_chunks: int | None = None) -> Dict:
    """Validate the aggregated results for completeness.
    
    Args:
        records: List of similarity records
        expected_chunks: Expected number of chunks (optional)
        
    Returns:
        Validation statistics
    """
    logger.info("Validating aggregated results...")
    
    # Basic statistics
    total_records = len(records)
    unique_queries = set()
    total_similarities = 0
    
    for record in records:
        query_key = (record["query_pmid"], record["query_model"])
        unique_queries.add(query_key)
        total_similarities += len(record["top_similarities"])
    
    stats = {
        "total_records": total_records,
        "unique_query_combinations": len(unique_queries),
        "total_similarity_entries": total_similarities,
        "average_similarities_per_query": total_similarities / total_records if total_records > 0 else 0,
    }
    
    logger.info("Validation statistics:")
    logger.info(f"  - Total records: {stats['total_records']}")
    logger.info(f"  - Unique query combinations: {stats['unique_query_combinations']}")
    logger.info(f"  - Total similarity entries: {stats['total_similarity_entries']}")
    logger.info(f"  - Average similarities per query: {stats['average_similarities_per_query']:.2f}")
    
    if expected_chunks:
        logger.info(f"  - Expected chunks: {expected_chunks}")
    
    return stats


def main():
    """Main function to aggregate trait similarity results."""
    args = make_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    logger.info(f"Input directory: {input_dir}")
    
    # Load all chunk files
    try:
        all_records = load_chunk_files(input_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Validate results
    stats = validate_results(all_records, args.expected_chunks)
    
    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_dir = DATA_DIR / "processed" / "similarities"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "trait_similarities.json"
    
    # Save aggregated results
    logger.info(f"Saving aggregated results to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w") as f:
        json.dump(all_records, f, indent=2)
    
    logger.info("Aggregation completed successfully!")
    
    # Save validation statistics
    stats_path = output_path.parent / "aggregation_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Validation statistics saved to: {stats_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
