"""Streamlit web app for demonstrating MR-KG structural PubMed literature data.

This application provides an interactive interface to explore:
1. Extracted data from the pmid_model_analysis view
2. Trait-based filtering using the model_result_traits table
3. Trait profile similarities from the trait_similarity_analysis view
"""

import argparse
from pathlib import Path
import streamlit as st

from yiutils.project_utils import find_project_root


def make_args():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=str,
        choices=["local", "docker"],
        default="local",
        help="Deployment profile: 'local' for local development, 'docker' for containerised deployment",
    )
    return parser.parse_args()


def setup_database_paths(profile: str):
    """Set up database paths based on deployment profile."""
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

    st.session_state.vector_store_db = vector_store_db_path
    st.session_state.trait_profile_db = trait_profile_db_path


def main():
    """Main application."""
    args = make_args()
    print(f"Running with profile: {args.profile}")

    # Set up database paths in session state
    if (
        "vector_store_db" not in st.session_state
        or "trait_profile_db" not in st.session_state
    ):
        setup_database_paths(args.profile)

    st.set_page_config(
        page_title="MR-KG Literature Explorer",
        page_icon="book",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("MR-KG Literature Explorer")
    st.markdown("""
    Explore structural PubMed literature data extracted by large language models.
    This tool provides insights into trait relationships and similar studies.
    """)

    st.markdown("""
    **Available Views:**
    - **About**: Further details about the resource
    - **Model Analysis**: Browse extracted data and filter by traits
    - **Trait Similarities**: Explore studies with similar trait profiles
    """)


if __name__ == "__main__":
    main()
