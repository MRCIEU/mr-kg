"""Generate embeddings for EFO term labels using SciSpacy models.

This script processes EFO term labels to generate embeddings:
1. Loads EFO term labels from processed JSON data
2. Uses SciSpacy models to generate embeddings
3. Processes EFO terms in chunks for parallel processing
4. Outputs embeddings for downstream analysis

NOTE: this should be run as part of a HPC array job.
"""

import argparse
import json
from pathlib import Path

import spacy
from loguru import logger
from tqdm import tqdm
from yiutils.chunking import calculate_chunk_start_end
from yiutils.project_utils import find_project_root

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = (
    MODELS_DIR
    / "en_core_sci_lg-0.5.4"
    / "en_core_sci_lg"
    / "en_core_sci_lg-0.5.4"
)
INPUT_EFO_PATH = DATA_DIR / "processed" / "efo" / "efo_terms.json"
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
    # ---- --aray-length ----
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
        help="Output directory for embeddings files",
    )
    res = parser.parse_args()
    return res


def main():
    """Main function to generate embeddings for EFO term labels.

    This function:
    1. Loads SciSpacy model for biomedical text processing
    2. Loads EFO term labels from processed JSON data
    3. Processes EFO terms in chunks for parallel processing
    4. Generates embeddings for the assigned chunk
    """
    # Parse command line arguments
    args = make_args()

    logger.info("Checking file paths and basic setup...")

    # Check SciSpacy model path
    if MODEL_PATH.exists():
        logger.info(f"✓ SciSpacy model path exists: {MODEL_PATH}")
    else:
        logger.error(f"✗ SciSpacy model path missing: {MODEL_PATH}")

    # Check EFO terms JSON path
    if INPUT_EFO_PATH.exists():
        logger.info(f"✓ EFO terms JSON path exists: {INPUT_EFO_PATH}")
    else:
        logger.error(f"✗ EFO terms JSON path missing: {INPUT_EFO_PATH}")

    # Check if this is a dry run - validate setup without processing
    if args.dry_run:
        logger.info("Dry run completed. Exiting without processing.")
        return

    # Load SciSpacy model
    nlp = spacy.load(str(MODEL_PATH))
    logger.info("SciSpacy model loaded successfully.")

    # Load EFO term labels
    logger.info("Loading EFO term labels...")
    with INPUT_EFO_PATH.open("r") as f:
        efo_terms = json.load(f)
    logger.info(f"Loaded {len(efo_terms)} EFO term labels.")

    # Process EFO terms into chunks
    total_efo_terms = len(efo_terms)
    start_idx, end_idx = calculate_chunk_start_end(
        chunk_id=args.array_id,
        num_chunks=args.array_length,
        data_length=total_efo_terms,
    )

    if start_idx is None or end_idx is None:
        logger.warning(
            f"Chunk {args.array_id} is out of range. No EFO terms to process."
        )
        return

    logger.info(
        f"Processing chunk {args.array_id}/{args.array_length}: "
        f"EFO terms [{start_idx} to {end_idx}) (total: {end_idx - start_idx})"
    )

    # Extract EFO term chunk for processing
    efo_chunk = efo_terms[start_idx:end_idx]

    logger.info(f"Chunk contains {len(efo_chunk)} EFO terms to process.")

    # Process EFO terms in this chunk
    for record in tqdm(efo_chunk, desc="Processing EFO terms"):
        efo_label = record["label"]
        doc = nlp(efo_label)
        vector = list(doc.vector.astype(float))
        record["vector"] = vector

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"efo_vectors_chunk_{args.array_id}.json"
    logger.info(f"Write to output file: {output_path}")
    with output_path.open("w") as f:
        json.dump(efo_chunk, f, indent=2)


if __name__ == "__main__":
    main()
