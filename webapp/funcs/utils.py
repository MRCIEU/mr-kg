"""Utility functions for the MR-KG web app."""

import pandas as pd
from typing import List, Dict

from pathlib import Path
from yiutils.project_utils import find_project_root


def get_database_paths(profile: str = "local") -> tuple[Path, Path]:
    if profile == "local":
        project_root = find_project_root("docker-compose.yml")
    else:  # docker
        # TODO: config this for docker
        project_root = Path("/app")

    vector_store_db_path = project_root / "data" / "db" / "vector_store.db"
    trait_profile_db_path = (
        project_root / "data" / "db" / "trait_profile_db.db"
    )

    if not vector_store_db_path.exists():
        raise FileNotFoundError(
            f"Vector store database not found at: {vector_store_db_path}"
        )

    if not trait_profile_db_path.exists():
        raise FileNotFoundError(
            f"Trait profile database not found at: {trait_profile_db_path}"
        )
    res = (vector_store_db_path, trait_profile_db_path)
    return res


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
