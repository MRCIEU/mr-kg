"Process trait labels from model results"

import argparse
import copy
import json
from typing import List, Optional, TypedDict, Union

import pandas as pd
from common_funcs.schema import processed_data_schema, raw_data_schema
from loguru import logger
from pydash import py_
from yiutils.project_utils import find_project_root

PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RAW_RESULTS_DIR = DATA_DIR / "raw" / "llm-results-aggregated"

MODELS = ["llama3", "llama3-2", "deepseek-r1-distilled", "gpt-4-1", "o4-mini"]


class TraitLabelResults(TypedDict):
    """Model results processed by extracting trait_labels"""

    model: str
    data: List[raw_data_schema.Rawdata]
    trait_labels: List[str]


# make args
def make_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Perform a dry run without actually processing",
    )
    res = parser.parse_args()
    return res


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

        # Extract traits from outcomes
        outcomes = metadata.get("outcomes", [])
        for outcome in outcomes:
            if isinstance(outcome, dict) and "trait" in outcome:
                trait = outcome["trait"]
                if trait:  # Only add non-empty traits
                    traits.append(trait)

        logger.info(f"PMID {pmid}: number of traits {len(traits)}")
        res = traits
        return res

    unique_traits = (
        py_.chain(data).map(get_all_traits).flatten().uniq().value()
    )
    res = unique_traits
    return res


# collect all results and unique them
def process_model(model: str) -> TraitLabelResults:
    """Process a single model and return its data and trait labels."""
    logger.info(f"Processing model: {model}")
    model_data_path = RAW_RESULTS_DIR / model / "processed_results_valid.json"
    assert model_data_path.exists(), (
        f"Model data path {model_data_path} does not exist."
    )

    with model_data_path.open("r") as f:
        data = json.load(f)
        trait_labels = get_trait_labels(data)
        logger.info(
            f"Model {model} has {len(trait_labels)} unique trait labels."
        )
    res = {"model": model, "data": data, "trait_labels": trait_labels}
    return res


def process_trait_item(
    trait_item: Union[dict, str], trait_to_index: dict
) -> Optional[dict]:
    """Add linked_index to a single trait item (exposure or outcome)."""
    if not isinstance(trait_item, dict):
        return None

    if "trait" in trait_item:
        trait = trait_item["trait"]
        if trait and trait in trait_to_index:
            trait_item["linked_index"] = trait_to_index[trait]

        # Ensure "id" is int if it exists
        if "id" in trait_item:
            try:
                trait_item["id"] = int(trait_item["id"])
            except (ValueError, TypeError):
                pass

        res = trait_item
        return res

    res = None
    return res


def process_metadata_item(
    item: raw_data_schema.Rawdata, trait_to_index: dict
) -> raw_data_schema.Rawdata:
    """Process a single metadata item and add linked indices to its exposures and outcomes."""
    metadata = item.get("metadata", {})

    # Process exposures
    exposures = metadata.get("exposures", [])
    if exposures:
        metadata["exposures"] = (
            py_.chain(exposures)
            .map(
                lambda trait_item: process_trait_item(
                    trait_item, trait_to_index
                )
            )
            .compact()
            .value()
        )

    # Process outcomes
    outcomes = metadata.get("outcomes", [])
    if outcomes:
        metadata["outcomes"] = (
            py_.chain(outcomes)
            .map(
                lambda trait_item: process_trait_item(
                    trait_item, trait_to_index
                )
            )
            .compact()
            .value()
        )

    res = item
    return res


def process_model_data(
    model_name: str,
    model_data: List[raw_data_schema.Rawdata],
    trait_to_index: dict,
) -> List[raw_data_schema.Rawdata]:
    """Process all data items for a single model."""
    logger.info(f"Adding linked indices for model: {model_name}")
    copied_model_data = copy.deepcopy(model_data)
    res = (
        py_.chain(copied_model_data)
        .map(lambda item: process_metadata_item(item, trait_to_index))
        .value()
    )
    return res


def add_linked_indices(
    model_results: List[TraitLabelResults], trait_to_index: dict
) -> List[processed_data_schema.ProcessModelResults]:
    """Add linked_index property to exposures and outcomes in model results and return a list of ProcessModelResults."""
    res = (
        py_.chain(model_results)
        .map(
            lambda model_result: {
                "model": model_result["model"],
                "data": process_model_data(
                    model_result["model"], model_result["data"], trait_to_index
                ),
            }
        )
        .value()
    )
    return res


def main():
    # Parse command line arguments
    args = make_args()

    # ==== init ====

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

    # [{"model": "llama3", "data": [...], "trait_labels": [...]}, ...]
    model_results: List[TraitLabelResults] = (
        py_.chain(MODELS).map(process_model).value()
    )
    unique_trait_labels = (
        py_.chain(model_results).map("trait_labels").flatten().uniq().value()
    )
    logger.info(
        f"Total unique trait labels across all models: {len(unique_trait_labels)}"
    )
    # Create a DataFrame of unique trait labels with an index column
    traits_df = pd.DataFrame(
        {
            "index": range(len(unique_trait_labels)),
            "trait": unique_trait_labels,
        }
    )

    # ==== Add linked indices to model results ====
    # Create trait to index mapping
    trait_to_index = dict(
        zip(unique_trait_labels, range(len(unique_trait_labels)))
    )
    linked_model_data: List[processed_data_schema.ProcessModelResults] = (
        add_linked_indices(model_results, trait_to_index)
    )

    # ==== write output ====
    output_dir = DATA_DIR / "processed"

    output_path_traits = output_dir / "traits" / "unique_traits.csv"
    output_path_traits.parent.mkdir(parents=True, exist_ok=True)
    traits_df.to_csv(output_path_traits, index=False)
    logger.info(
        f"Unique trait labels have been written to {output_path_traits}"
    )

    output_path_model_results = (
        output_dir / "model_results" / "processed_model_results.json"
    )
    output_path_model_results.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_model_results, "w") as f:
        json.dump(linked_model_data, f, indent=2)
    logger.info(
        f"Processed model results have been written to {output_path_model_results}"
    )


if __name__ == "__main__":
    main()
