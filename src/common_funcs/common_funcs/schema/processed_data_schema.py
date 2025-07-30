from typing import Any, Dict, List, TypedDict, Union

from common_funcs.schema import raw_data_schema


class LinkedExposureOutcomeItem(TypedDict, total=False):
    """Individual exposure or outcome item with detailed information."""

    id: int
    trait: str
    category: str  # Required field
    linked_index: int


class LinkedMetadata(TypedDict):
    exposures: List[Union[str, LinkedExposureOutcomeItem]]
    outcomes: List[Union[str, LinkedExposureOutcomeItem]]
    methods: List[Union[raw_data_schema.MethodItem, List[str], str]]
    population: Union[
        str, List[str], List[Dict[str, Any]], raw_data_schema.PopulationError
    ]


class LinkedModelData(TypedDict):
    pmid: str
    metadata: LinkedMetadata
    results: raw_data_schema.Results


class ProcessModelResults(TypedDict):
    """Model results processed with linking trait indices"""

    model: str
    data: List[LinkedModelData]
    trait_labels: List[str]
