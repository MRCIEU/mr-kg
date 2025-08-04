"""Aggregate embeddings from trait and EFO embedding results.

This script processes embedding results to:
1. Read in embedding data from HPC array job outputs
2. Process and convert data to common EmbeddingRecord structure
3. Export aggregated embeddings for downstream analysis
"""

import argparse
import json
from pathlib import Path
from typing import List

from common_funcs.schema.embedding_schema import (
    EmbeddingRecord,
    RawEfoRecord,
    RawTraitRecord,
)
from loguru import logger
from pydash import py_
from yiutils.project_utils import find_project_root

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments with input directories and dry_run option
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # ---- input directories ----
    parser.add_argument(
        "--trait-results-dir",
        type=Path,
        required=True,
        help="Directory containing trait embedding results",
    )
    parser.add_argument(
        "--efo-results-dir",
        type=Path,
        required=True,
        help="Directory containing EFO embedding results",
    )
    # ---- --dry-run ----
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually processing",
    )
    res = parser.parse_args()
    return res


def read_trait_file(file_path: Path) -> List[RawTraitRecord]:
    """Read a single trait embedding file.

    Args:
        file_path: Path to the trait embedding JSON file

    Returns:
        List of raw trait records from the file
    """
    with file_path.open("r") as f:
        data = json.load(f)
    return data


def read_efo_file(file_path: Path) -> List[RawEfoRecord]:
    """Read a single EFO embedding file.

    Args:
        file_path: Path to the EFO embedding JSON file

    Returns:
        List of raw EFO records from the file
    """
    with file_path.open("r") as f:
        data = json.load(f)
    return data


def convert_trait_record(record: RawTraitRecord) -> EmbeddingRecord:
    """Convert a raw trait record to EmbeddingRecord format.

    Args:
        record: Raw trait record containing trait and vector

    Returns:
        Processed embedding record
    """
    res = {
        "id": f"trait_{record['index']}",
        "label": record["trait"],
        "vector": record["vector"],
    }
    return res


def convert_efo_record(record: RawEfoRecord) -> EmbeddingRecord:
    """Convert a raw EFO record to EmbeddingRecord format.

    Args:
        record: Raw EFO record containing id, label and vector

    Returns:
        Processed embedding record
    """
    res = {
        "id": record["id"],
        "label": record["label"],
        "vector": record["vector"],
    }
    return res


def main():
    """Main function to aggregate embeddings from trait and EFO results.

    This function:
    1. Reads in embedding data from HPC array job outputs
    2. Processes data to common EmbeddingRecord structure
    3. Exports aggregated embeddings for downstream analysis
    """
    # Parse command line arguments
    args = make_args()

    logger.info("Checking file paths and basic setup...")

    # ==== init ====
    # Check trait results directory
    trait_results_dir = args.trait_results_dir
    if trait_results_dir.exists():
        logger.info(f"✓ Trait results directory exists: {trait_results_dir}")
    else:
        logger.error(f"✗ Trait results directory missing: {trait_results_dir}")

    # Check EFO results directory
    efo_results_dir = args.efo_results_dir
    if efo_results_dir.exists():
        logger.info(f"✓ EFO results directory exists: {efo_results_dir}")
    else:
        logger.error(f"✗ EFO results directory missing: {efo_results_dir}")

    # Check if this is a dry run - validate setup without processing
    if args.dry_run:
        logger.info("Dry run completed. Exiting without processing.")
        return

    # ==== Read in data ====

    # ---- read raw trait data ----
    logger.info("Reading trait embedding data...")

    logger.info(f"Reading trait embeddings from: {trait_results_dir}")

    trait_files = list(trait_results_dir.glob("trait_vectors_chunk_*.json"))
    logger.info(f"Found {len(trait_files)} trait embedding files")

    raw_trait_data: List[RawTraitRecord] = (
        py_.chain(trait_files).map(read_trait_file).flatten().value()
    )
    logger.info(f"Total trait records loaded: {len(raw_trait_data)}")

    # ---- read raw efo data ----
    logger.info("Reading EFO embedding data...")

    logger.info(f"Reading EFO embeddings from: {efo_results_dir}")

    efo_files = list(efo_results_dir.glob("efo_vectors_chunk_*.json"))
    logger.info(f"Found {len(efo_files)} EFO embedding files")

    raw_efo_data: List[RawEfoRecord] = (
        py_.chain(efo_files).map(read_efo_file).flatten().value()
    )
    logger.info(f"Total EFO records loaded: {len(raw_efo_data)}")

    # ==== Process data ====
    logger.info("Processing trait data...")
    trait_embeddings: List[EmbeddingRecord] = (
        py_.chain(raw_trait_data).map(convert_trait_record).value()
    )

    logger.info("Processing EFO data...")
    efo_embeddings: List[EmbeddingRecord] = (
        py_.chain(raw_efo_data).map(convert_efo_record).value()
    )

    # ==== Export data ====
    output_dir = DATA_DIR / "processed" / "embeddings"
    logger.info(f"Creating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trait_output_path = output_dir / "traits.json"
    efo_output_path = output_dir / "efo.json"

    # Export trait embeddings
    logger.info(
        f"Writing {len(trait_embeddings)} trait embeddings to: {trait_output_path}"
    )
    with trait_output_path.open("w") as f:
        json.dump(trait_embeddings, f, indent=2)

    # Export EFO embeddings
    logger.info(
        f"Writing {len(efo_embeddings)} EFO embeddings to: {efo_output_path}"
    )
    with efo_output_path.open("w") as f:
        json.dump(efo_embeddings, f, indent=2)

    logger.info("Embeddings have been written successfully")


if __name__ == "__main__":
    main()
