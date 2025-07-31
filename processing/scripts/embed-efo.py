"""Generate embeddings for EFO term labels using SciSpacy models.

This script processes EFO term labels to generate embeddings:
1. Loads EFO term labels from processed JSON data
2. Uses SciSpacy models to generate embeddings
3. Processes all EFO terms in one go
4. Outputs embeddings for downstream analysis
"""

import argparse
import json
from pathlib import Path

import spacy
from loguru import logger
from tqdm import tqdm
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
        Parsed command line arguments with dry_run and output_dir options
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # ---- --dry-run ----
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually processing",
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
    3. Generates embeddings for all EFO terms
    4. Outputs embeddings as JSON
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

    # Process all EFO terms
    logger.info("Processing all EFO terms...")
    for record in tqdm(efo_terms, desc="Processing EFO terms"):
        efo_label = record["label"]
        doc = nlp(efo_label)
        vector = list(doc.vector.astype(float))
        record["vector"] = vector

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "efo_vectors.json"
    logger.info(f"Write to output file: {output_path}")
    with output_path.open("w") as f:
        json.dump(efo_terms, f, indent=2)


if __name__ == "__main__":
    main()
