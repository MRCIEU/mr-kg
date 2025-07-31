"""Generate embeddings for trait labels using SciSpacy models.

This script processes trait labels to generate embeddings:
1. Loads trait labels from processed CSV data
2. Uses SciSpacy models to generate embeddings
3. Processes traits in chunks for parallel processing
4. Outputs embeddings for downstream analysis
"""

import argparse
import pandas as pd
import spacy
from loguru import logger
from yiutils.project_utils import find_project_root
from yiutils.chunking import calculate_chunk_start_end

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "en_core_sci_lg-0.5.4" / "en_core_sci_lg"
INPUT_TRAITS_PATH = DATA_DIR / "traits" / "unique_traits.csv"


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
    res = parser.parse_args()
    return res


def main():
    """Main function to generate embeddings for trait labels.

    This function:
    1. Loads SciSpacy model for biomedical text processing
    2. Loads trait labels from processed CSV data
    3. Processes traits in chunks for parallel processing
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

    # Check traits CSV path
    if INPUT_TRAITS_PATH.exists():
        logger.info(f"✓ Traits CSV path exists: {INPUT_TRAITS_PATH}")
    else:
        logger.error(f"✗ Traits CSV path missing: {INPUT_TRAITS_PATH}")

    # Check if this is a dry run - validate setup without processing
    if args.dry_run:
        logger.info("Dry run completed. Exiting without processing.")
        return

    # Load SciSpacy model
    nlp = spacy.load(str(MODEL_PATH))
    logger.info("SciSpacy model loaded successfully.")

    # Load trait labels
    logger.info("Loading trait labels...")
    traits_df = pd.read_csv(INPUT_TRAITS_PATH)
    logger.info(f"Loaded {len(traits_df)} trait labels.")

    # Process trait labels into chunks
    total_traits = len(traits_df)
    start_idx, end_idx = calculate_chunk_start_end(
        chunk_id=args.array_id,
        num_chunks=args.array_length,
        data_length=total_traits
    )

    if start_idx is None or end_idx is None:
        logger.warning(f"Chunk {args.array_id} is out of range. No traits to process.")
        return

    logger.info(
        f"Processing chunk {args.array_id}/{args.array_length}: "
        f"traits [{start_idx} to {end_idx }) (total: {end_idx - start_idx})"
    )

    # Extract trait chunk for processing
    trait_chunk = traits_df.iloc[start_idx:end_idx]
    trait_labels = trait_chunk['trait'].tolist()

    logger.info(f"Chunk contains {len(trait_labels)} traits to process.")

    # Process traits in this chunk
    for idx, trait in enumerate(trait_labels):
        logger.info(f"Processing trait {idx + 1}/{len(trait_labels)}: {trait}")
        # TODO: Generate embeddings using nlp(trait) and save results


if __name__ == "__main__":
    main()
