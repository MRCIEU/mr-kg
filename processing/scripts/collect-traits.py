"Process trait labels from model results"

import argparse
import json

from common_funcs.schema import raw_data_schema
from loguru import logger
from pydash import py_
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RAW_RESULTS_DIR = DATA_DIR / "raw" / "llm-results-aggregated"

MODELS = ["llama3", "llama3-2", "deepseek-r1-distilled", "gpt-4-1", "o4-mini"]


# make args
def make_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually processing",
    )
    return parser.parse_args()


def get_trait_labels(data: list[raw_data_schema.Rawdata]) -> list[str]:
    # exposures: for each item, get trait field
    # outcomes: for each item, get trait field

    def get_all_traits(item: raw_data_schema.Rawdata) -> list[str]:
        """Extract all traits from a single Rawdata item."""
        traits = []
        pmid = item.get("pmid", None)
        logger.info(f"Processing item with PMID: {pmid}")
        metadata: raw_data_schema.Metadata = item.get("metadata", {})

        # Extract traits from exposures
        exposures = metadata.get("exposures", [])
        for exposure in exposures:
            if isinstance(exposure, dict) and "trait" in exposure:
                trait = exposure["trait"]
                if trait:  # Only add non-empty traits
                    traits.append(trait)
            elif isinstance(exposure, str):
                # When exposure is a string, use it as the trait
                logger.info(f"Exposure is a string: {exposure}")
                traits.append(exposure)

        # Extract traits from outcomes
        outcomes = metadata.get("outcomes", [])
        for outcome in outcomes:
            if isinstance(outcome, dict) and "trait" in outcome:
                trait = outcome["trait"]
                if trait:  # Only add non-empty traits
                    traits.append(trait)
            elif isinstance(outcome, str):
                # When outcome is a string, use it as the trait
                logger.info(f"Outcome is a string: {outcome}")
                traits.append(outcome)

        logger.info(f"PMID {pmid}: number of traits {len(traits)}")
        return traits

    unique_traits = (
        py_.chain(data).map(get_all_traits).flatten().uniq().value()
    )
    return unique_traits


# main
def main():
    # Parse command line arguments
    args = make_args()

    # ===== init ====

    # If dry run, stop here
    if args.dry_run:
        logger.info("Dry run mode: Checking file paths and basic setup...")
        for model in MODELS:
            model_data_path = (
                RAW_RESULTS_DIR / model / "processed_results_valid.json"
            )
            if model_data_path.exists():
                logger.info(f"✓ Model data path exists: {model_data_path}")
            else:
                logger.error(f"✗ Model data path missing: {model_data_path}")
        logger.info("Dry run completed. Exiting without processing.")
        return

    # ==== get results ====
    # collect all results and unique them
    def process_model(model: str) -> list[str]:
        """Process a single model and return its trait labels."""
        logger.info(f"Processing model: {model}")
        model_data_path = (
            RAW_RESULTS_DIR / model / "processed_results_valid.json"
        )
        assert model_data_path.exists(), (
            f"Model data path {model_data_path} does not exist."
        )

        with model_data_path.open("r") as f:
            data = json.load(f)
            trait_labels = get_trait_labels(data)
            logger.info(
                f"Model {model} has {len(trait_labels)} unique trait labels."
            )
        return trait_labels

    results: dict[str, list[str]] = (
        py_.chain(MODELS)
        .map(lambda model: (model, process_model(model)))
        .from_pairs()
        .value()
    )
    unique_trait_labels = py_.chain(results).values().flatten().uniq().value()
    logger.info(
        f"Total unique trait labels across all models: {len(unique_trait_labels)}"
    )

    # write output
    output_dir = DATA_DIR / "processed" / "traits.txt"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with output_dir.open("w") as f:
        for trait in unique_trait_labels:
            f.write(f"{trait}\n")

    logger.info(f"Unique trait labels have been written to {output_dir}")


if __name__ == "__main__":
    main()
