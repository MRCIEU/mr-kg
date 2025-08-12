"""Streamlit web app for demonstrating MR-KG structural PubMed literature data.

This application provides an interactive interface to explore:
1. Extracted data from the pmid_model_analysis view
2. Trait-based filtering using the model_result_traits table
3. Trait profile similarities from the trait_similarity_analysis view
"""

import argparse
from common_funcs.database_utils.utils import get_database_paths
import streamlit as st


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


def init(args):
    MODELS = [
        "llama3",
        "llama3-2",
        "deepseek-r1-distilled",
        "gpt-4-1",
        "o4-mini",
    ]
    if "models" not in st.session_state:
        st.session_state.models = MODELS
    # Set up database paths in session state
    vector_store_db_path, trait_profile_db_path = get_database_paths(
        args.profile
    )
    print(vector_store_db_path, trait_profile_db_path)
    if "vector_store_db" not in st.session_state:
        st.session_state.vector_store_db = vector_store_db_path
    if "trait_profile_db" not in st.session_state:
        st.session_state.trait_profile_db = trait_profile_db_path


def main():
    """Main application."""
    args = make_args()
    print(f"Running with profile: {args.profile}")

    init(args)

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
    - `explore_traits`: start from here to choose traits of interest
    """)


if __name__ == "__main__":
    main()
