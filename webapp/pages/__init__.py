"""Pages module for the MR-KG web app."""

from .model_analysis import show_model_analysis
from .trait_similarities import show_trait_similarities
from .about import show_about

__all__ = ["show_model_analysis", "show_trait_similarities", "show_about"]
