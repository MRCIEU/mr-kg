"""Streamlit web app for demonstrating MR-KG structural PubMed literature data.

This application provides an interactive interface to explore:
1. Extracted data from the pmid_model_analysis view
2. Trait-based filtering using the model_result_traits table
3. Trait profile similarities from the trait_similarity_analysis view
"""

import argparse
import streamlit as st
from pathlib import Path
from app.pages import show_model_analysis, show_trait_similarities, show_about


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
        project_root = Path("..")
    else:  # docker
        project_root = Path("/app")

    st.session_state.vector_store_db = (
        project_root / "data" / "db" / "vector_store.db"
    )
    st.session_state.trait_profile_db = (
        project_root / "data" / "db" / "trait_profile_db.db"
    )


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

    **Current Database Status:**
    - Main analysis data: 26,165 model results
    - Trait relationships: 114,276 trait links
    - Similarity data: 248,560 trait profile comparisons
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    **Available Views:**
    - **Model Analysis**: Browse extracted data and filter by traits
    - **Trait Similarities**: Explore studies with similar trait profiles
    """)

    page = st.sidebar.selectbox(
        "Choose a view:", ["Model Analysis", "Trait Similarities", "About"]
    )

    if page == "Model Analysis":
        show_model_analysis()
    elif page == "Trait Similarities":
        show_trait_similarities()
    elif page == "About":
        show_about()


if __name__ == "__main__":
    main()
