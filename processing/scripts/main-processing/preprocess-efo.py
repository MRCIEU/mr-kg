"""Process EFO ontology data to extract term labels.

This script processes EFO (Experimental Factor Ontology) data to:
1. Extract term IDs and labels from the EFO JSON file
2. Filter out terms without labels
3. Export the processed terms as JSON
"""

import argparse
import json
from typing import List, Optional

from common_funcs.schema.efo_schema import EfoTermRecord
from loguru import logger
from pydash import py_
from yiutils.project_utils import find_project_root

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
EFO_INPUT_PATH = DATA_DIR / "raw" / "efo" / "efo.json"


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments with dry_run option
    """
    parser = argparse.ArgumentParser(description=__doc__)
    # ---- --dry-run ----
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually processing",
    )
    res = parser.parse_args()
    return res


def extract_term(node) -> Optional[EfoTermRecord]:
    """Extract term record from a single node.

    Args:
        node: EFO node containing id and lbl fields

    Returns:
        EFO term record with id and label, or None if no label present
    """
    if "lbl" not in node:
        return None
    res = {"id": node["id"], "label": node["lbl"]}
    return res


def main():
    """Main function to process EFO ontology data and extract term labels.

    This function:
    1. Parses command line arguments
    2. Extracts EFO term labels from the ontology file
    3. Exports the processed terms to JSON
    """
    # ==== init ====
    args = make_args()

    # ==== dry-run ====
    if EFO_INPUT_PATH.exists():
        logger.info(f"✓ EFO input path exists: {EFO_INPUT_PATH}")
    else:
        logger.error(f"✗ EFO input path missing: {EFO_INPUT_PATH}")
        return
    if args.dry_run:
        logger.info("Dry run completed. Exiting without processing.")
        return

    # ==== Extract EFO labels ====
    logger.info(f"Reading EFO data from: {EFO_INPUT_PATH}")
    with EFO_INPUT_PATH.open("r") as f:
        data = json.load(f)
    nodes = data["graphs"][0]["nodes"]
    logger.info(f"Found {len(nodes)} nodes in EFO data")

    # Extract terms and filter out those without labels
    efo_terms: List[EfoTermRecord] = (
        py_.chain(nodes)
        .map(extract_term)
        .filter(lambda term: term is not None)
        .value()
    )

    logger.info(f"Extracted {len(efo_terms)} EFO terms with labels")

    # ==== Export results ====
    output_path = DATA_DIR / "processed" / "efo" / "efo_terms.json"
    logger.info(f"Creating output directory: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(efo_terms)} EFO terms to: {output_path}")
    with output_path.open("w") as f:
        json.dump(efo_terms, f, indent=2)
    logger.info("EFO terms have been written successfully")


if __name__ == "__main__":
    main()
