"""Shared UI components for the webapp."""

from components.model_selector import model_selector
from components.similarity_display import (
    evidence_similarity_table,
    trait_similarity_table,
)
from components.study_table import study_table
from components.theme import apply_theme

__all__ = [
    "model_selector",
    "study_table",
    "trait_similarity_table",
    "evidence_similarity_table",
    "apply_theme",
]
