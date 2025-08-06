"""Utility functions for the MR-KG web app."""

import pandas as pd
from typing import List, Dict


def format_traits_display(traits_list: List[Dict]) -> str:
    """Format traits list for display."""
    if not traits_list or traits_list is None:
        return "No traits"

    try:
        trait_labels = []
        for trait in traits_list:
            if isinstance(trait, dict):
                label = trait.get("trait_label", "Unknown")
            elif isinstance(trait, str):
                label = trait
            else:
                label = str(trait)
            trait_labels.append(label)
        return ", ".join(trait_labels) if trait_labels else "No traits"
    except Exception as e:
        return f"Error formatting traits: {str(e)}"


def safe_format_traits(traits):
    """Safely format traits for display in DataFrames."""
    if pd.isna(traits) or traits is None:
        return "No traits"
    try:
        return format_traits_display(traits)
    except Exception:
        return "Error formatting traits"
