"""
types related to postprocessing and data base building
"""

from typing import TypedDict


class RawTraitRecord(TypedDict):
    """Raw trait record"""

    index: int
    trait: str
    vector: list[float]  # dim 200


class RawEfoRecord(TypedDict):
    """Raw trait record"""

    id: str
    label: str
    vector: list[float]  # dim 200


class EmbeddingRecord(TypedDict):
    """Processed record"""

    id: str
    label: str
    vector: list[float]  # dim 200
