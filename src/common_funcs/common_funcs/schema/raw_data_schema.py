from typing import TypedDict, Union, List, Any, Dict


class ExposureOutcomeItem(TypedDict, total=False):
    """Individual exposure or outcome item with detailed information."""

    id: Union[str, int]
    name: str
    trait: str
    category: str  # Required field


class MethodItem(TypedDict, total=False):
    """Individual method item with detailed information."""

    id: Union[str, int]
    method: str
    name: str
    category: str
    description: str


class PopulationError(TypedDict, total=False):
    """Population error structure."""

    error: str
    explanation: str


class Metadata(TypedDict, total=False):
    """Metadata schema for processed results."""

    exposures: List[Union[str, ExposureOutcomeItem]]
    outcomes: List[Union[str, ExposureOutcomeItem]]
    methods: List[Union[MethodItem, List[str], str]]
    population: Union[str, List[str], List[Dict[str, Any]], PopulationError]


class ResultItem(TypedDict):
    """Individual result item with statistical measures."""

    exposure: Union[str, None]
    outcome: Union[str, None]
    beta: Union[float, int, str, None]
    units: Union[str, None]
    # Note: Field names with spaces and special characters need to be accessed using dict notation
    # e.g., result_item["odds ratio"] or result_item["95% CI"]


class ResultItemWithSpecialFields(TypedDict):
    """Result item with all fields including those with special characters."""

    exposure: Union[str, None]
    outcome: Union[str, None]
    beta: Union[float, int, str, None]
    units: Union[str, None]
    SE: Union[float, int, str, None]
    direction: Union[str, None]


# Type alias for Results array
Results = List[
    Dict[str, Union[str, float, int, None, List[Union[str, float, int, None]]]]
]


class Rawdata(TypedDict):
    pmid: str
    metadata: Metadata
    results: Results
