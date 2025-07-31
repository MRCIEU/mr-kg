"""Process trait labels from model results and create linked indices.

This script processes LLM model results to:
1. Extract unique trait labels from all models
2. Create a mapping between traits and indices
3. Add linked indices to exposure and outcome traits in the data
4. Output both the trait mapping and processed model results
"""

import argparse
import copy
import json
from typing import List, Optional, TypedDict, Union

import pandas as pd
from common_funcs.schema import processed_data_schema, raw_data_schema
from loguru import logger
from pydash import py_
from yiutils.project_utils import find_project_root

# Project configuration
PROJECT_ROOT = find_project_root("docker-compose.yml")
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RAW_RESULTS_DIR = DATA_DIR / "raw" / "llm-results-aggregated"

# List of models to process
MODELS = ["llama3", "llama3-2", "deepseek-r1-distilled", "gpt-4-1", "o4-mini"]


class TraitLabelResults(TypedDict):
    """Structure for model results with extracted trait labels.

    Attributes:
        model: Name of the model
        data: Raw data items from the model
        trait_labels: Unique trait labels extracted from this model's data
    """
    model: str
    data: List[raw_data_schema.Rawdata]
    trait_labels: List[str]


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments with dry_run option
    """
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
    """Extract unique trait labels from model data.

    Processes both exposures and outcomes in the metadata to find all unique
    trait values across all data items.

    Args:
        data: List of raw data items from a model

    Returns:
        List of unique trait labels found in the data
    """

    def get_all_traits(item: raw_data_schema.Rawdata) -> list[str]:
        """Extract all traits from a single raw data item.

        Args:
            item: Single raw data item containing metadata with exposures/outcomes

        Returns:
            List of trait strings found in this item
        """
        traits = []
        metadata: raw_data_schema.Metadata = item.get("metadata", {})

        # Extract traits from exposures - only process dict items with 'trait' key
        exposures = metadata.get("exposures", [])
        for exposure in exposures:
            if isinstance(exposure, dict) and "trait" in exposure:
                trait = exposure["trait"]
                if trait:  # Only add non-empty traits
                    traits.append(trait)

        # Extract traits from outcomes - only process dict items with 'trait' key
        outcomes = metadata.get("outcomes", [])
        for outcome in outcomes:
            if isinstance(outcome, dict) and "trait" in outcome:
                trait = outcome["trait"]
                if trait:  # Only add non-empty traits
                    traits.append(trait)

        res = traits
        return res

    # Use functional programming to extract and deduplicate traits across all items
    unique_traits = (
        py_.chain(data).map(get_all_traits).flatten().uniq().value()
    )
    res = unique_traits
    return res


def process_model(model: str) -> TraitLabelResults:
    """Process a single model and extract its data and trait labels.

    Loads the processed results JSON file for the specified model and extracts
    unique trait labels from the data.

    Args:
        model: Name of the model to process

    Returns:
        Dictionary containing model name, raw data, and extracted trait labels

    Raises:
        AssertionError: If the model data file doesn't exist
    """
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
    """Process a single trait item and add linked index.

    Adds a 'linked_index' field to trait items that correspond to their position
    in the global trait index. Also ensures 'id' fields are integers.

    Args:
        trait_item: Exposure or outcome item (dict or string)
        trait_to_index: Mapping from trait names to their indices

    Returns:
        Processed trait item with linked_index added, or None if not processable
    """
    # Only process dictionary items - filter out non-dict items
    if not isinstance(trait_item, dict):
        logger.info(f"Trait item is not a dict: {trait_item}")
        return None

    # Process items that have a 'trait' field
    if "trait" in trait_item:
        trait = trait_item["trait"]
        # Add linked index if trait exists in our mapping
        if trait and trait in trait_to_index:
            trait_item["linked_index"] = trait_to_index[trait]

        # Ensure "id" field is integer type if it exists
        if "id" in trait_item:
            try:
                trait_item["id"] = int(trait_item["id"])
            except (ValueError, TypeError):
                # Keep original value if conversion fails
                pass

        res = trait_item
        return res

    # Return None for items without 'trait' field (will be filtered out)
    res = None
    return res


def process_metadata_item(
    item: raw_data_schema.Rawdata, trait_to_index: dict
) -> raw_data_schema.Rawdata:
    """Process metadata for a single data item, adding linked indices.

    Processes both exposures and outcomes lists, adding linked indices to valid
    trait items and filtering out invalid ones.

    Args:
        item: Single raw data item
        trait_to_index: Mapping from trait names to their indices

    Returns:
        Modified data item with linked indices added to traits
    """
    metadata = item.get("metadata", {})

    # Process exposures list - add linked indices and filter out None results
    exposures = metadata.get("exposures", [])
    if exposures:
        metadata["exposures"] = (
            py_.chain(exposures)
            .map(
                lambda trait_item: process_trait_item(
                    trait_item, trait_to_index
                )
            )
            .compact()  # Remove None values
            .value()
        )

    # Process outcomes list - add linked indices and filter out None results
    outcomes = metadata.get("outcomes", [])
    if outcomes:
        metadata["outcomes"] = (
            py_.chain(outcomes)
            .map(
                lambda trait_item: process_trait_item(
                    trait_item, trait_to_index
                )
            )
            .compact()  # Remove None values
            .value()
        )

    res = item
    return res


def process_model_data(
    model_name: str,
    model_data: List[raw_data_schema.Rawdata],
    trait_to_index: dict,
) -> List[raw_data_schema.Rawdata]:
    """Process all data items for a single model, adding linked indices.

    Creates a deep copy of the model data to avoid modifying the original,
    then processes each item to add linked indices to trait references.

    Args:
        model_name: Name of the model being processed
        model_data: List of raw data items from the model
        trait_to_index: Mapping from trait names to their indices

    Returns:
        List of processed data items with linked indices added
    """
    logger.info(f"Adding linked indices for model: {model_name}")
    # Create deep copy to avoid modifying original data
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
    """Add linked indices to all model results.

    Processes each model's data to add linked indices that reference the global
    trait index, creating the final processed model results structure.

    Args:
        model_results: List of raw model results with trait labels
        trait_to_index: Mapping from trait names to their indices

    Returns:
        List of processed model results with linked indices added
    """
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
    """Main function to process trait labels and create linked model data.

    This function:
    1. Processes all model results to extract unique traits
    2. Creates a global trait index mapping
    3. Adds linked indices to all trait references in the data
    4. Outputs both the trait index and processed model results
    """
    # Parse command line arguments
    args = make_args()

    # ==== Initialisation and validation ====

    # Check if this is a dry run - validate file paths without processing
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

    # ==== Process model results and extract traits ====

    # Load and process each model's data to extract trait labels
    model_results: List[TraitLabelResults] = (
        py_.chain(MODELS).map(process_model).value()
    )

    # Combine trait labels from all models and create unique list
    unique_trait_labels = (
        py_.chain(model_results).map("trait_labels").flatten().uniq().value()
    )
    logger.info(
        f"Total unique trait labels across all models: {len(unique_trait_labels)}"
    )

    # Create DataFrame with indexed trait labels for output
    traits_df = pd.DataFrame(
        {
            "index": range(len(unique_trait_labels)),
            "trait": unique_trait_labels,
        }
    )

    # ==== Create linked indices and process model data ====

    # Create mapping from trait names to their indices
    trait_to_index = dict(
        zip(unique_trait_labels, range(len(unique_trait_labels)))
    )

    # Process all model data to add linked indices
    linked_model_data: List[processed_data_schema.ProcessModelResults] = (
        add_linked_indices(model_results, trait_to_index)
    )

    # ==== Write output files ====

    output_dir = DATA_DIR / "processed"

    # Write trait index CSV file
    output_path_traits = output_dir / "traits" / "unique_traits.csv"
    output_path_traits.parent.mkdir(parents=True, exist_ok=True)
    traits_df.to_csv(output_path_traits, index=False)
    logger.info(
        f"Unique trait labels have been written to {output_path_traits}"
    )

    # Write processed model results JSON file
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
