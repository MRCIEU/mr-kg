"""Streamlit web app for demonstrating MR-KG structural PubMed literature data.

This application provides an interactive interface to explore:
1. Extracted data from the pmid_model_analysis view
2. Trait-based filtering using the model_result_traits table
3. Trait profile similarities from the trait_similarity_analysis view
"""

import argparse
import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
PROJECT_ROOT = Path("..")
VECTOR_STORE_DB = PROJECT_ROOT / "data" / "db" / "vector_store.db"
TRAIT_PROFILE_DB = PROJECT_ROOT / "data" / "db" / "trait_profile_db.db"


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


@st.cache_resource
def get_database_connections():
    """Get database connections with caching."""
    vector_conn = duckdb.connect(str(VECTOR_STORE_DB), read_only=True)
    trait_conn = duckdb.connect(str(TRAIT_PROFILE_DB), read_only=True)
    return vector_conn, trait_conn


@st.cache_data
def get_available_traits() -> List[str]:
    """Get list of available traits for filtering."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT DISTINCT trait_label
    FROM model_result_traits
    ORDER BY trait_label
    """
    result = vector_conn.execute(query).fetchall()
    return [row[0] for row in result]


@st.cache_data
def get_pmid_model_data(
    trait_filters: Optional[List[str]] = None, limit: int = 100
) -> pd.DataFrame:
    """Get data from pmid_model_analysis view with optional trait filtering."""
    vector_conn, _ = get_database_connections()

    base_query = """
    SELECT
        pmid,
        model,
        model_result_id,
        title,
        abstract,
        pub_date,
        journal,
        journal_issn,
        author_affil,
        traits
    FROM pmid_model_analysis
    """

    if trait_filters:
        # Create a subquery to filter by traits first
        trait_filter_conditions = []
        for trait in trait_filters:
            escaped_trait = trait.replace("'", "''")
            trait_filter_conditions.append(f"trait_label = '{escaped_trait}'")

        trait_filter_query = f"""
        SELECT DISTINCT model_result_id
        FROM model_result_traits
        WHERE {" OR ".join(trait_filter_conditions)}
        """

        query = f"""
        {base_query}
        WHERE model_result_id IN ({trait_filter_query})
        """
    else:
        query = base_query

    query += f" ORDER BY pmid, model LIMIT {limit}"

    result = vector_conn.execute(query).fetchdf()
    return result


@st.cache_data
def get_trait_similarity_data(
    limit: int = 100, min_similarity: float = 0.0
) -> pd.DataFrame:
    """Get data from trait_similarity_analysis view."""
    _, trait_conn = get_database_connections()

    query = f"""
    SELECT
        query_pmid,
        query_model,
        query_title,
        query_trait_count,
        similar_pmid,
        similar_model,
        similar_title,
        similar_trait_count,
        trait_profile_similarity,
        trait_jaccard_similarity,
        similarity_rank
    FROM trait_similarity_analysis
    WHERE trait_profile_similarity >= {min_similarity}
    ORDER BY trait_profile_similarity DESC
    LIMIT {limit}
    """

    result = trait_conn.execute(query).fetchdf()
    return result


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


def main():
    """Main application."""
    args = make_args()
    print(f"Running with profile: {args.profile}")

    st.set_page_config(
        page_title="MR-KG Literature Explorer",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📚 MR-KG Literature Explorer")
    st.markdown("""
    Explore structural PubMed literature data extracted by large language models.
    This tool provides insights into trait relationships and similar studies.

    **Current Database Status:**
    - Main analysis data: 26,165 model results
    - Trait relationships: 114,276 trait links
    - Similarity data: 248,560 trait profile comparisons
    """)

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
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


def show_model_analysis():
    """Show the model analysis view."""
    st.header("📊 Model Analysis")
    st.markdown("Browse extracted data from PubMed literature analysis.")

    # Get available traits for filtering
    available_traits = get_available_traits()

    # Filters
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_traits = st.multiselect(
            "Filter by traits (optional):",
            options=available_traits,
            help="Select one or more traits to filter the results",
        )

    with col2:
        limit = st.number_input(
            "Max results:", min_value=10, max_value=1000, value=100, step=10
        )

    # Get and display data
    try:
        data = get_pmid_model_data(
            trait_filters=selected_traits if selected_traits else None,
            limit=limit,
        )

        if data.empty:
            st.warning("No data found with the current filters.")
            return

        st.success(f"Found {len(data)} records")

        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Unique PMIDs", data["pmid"].nunique())
        with col3:
            st.metric("Unique Models", data["model"].nunique())
        with col4:
            st.metric("Unique Journals", data["journal"].nunique())

        # Model distribution
        st.subheader("Model Distribution")
        model_counts = data["model"].value_counts()
        st.bar_chart(model_counts)

        # Data table
        st.subheader("Detailed Results")

        # Prepare display data
        display_data = data.copy()

        # Truncate long text fields for table display
        display_data["title_short"] = (
            display_data["title"].str[:100].fillna("") + "..."
        )
        display_data["abstract_short"] = (
            display_data["abstract"].str[:200].fillna("") + "..."
        )

        # Handle traits display safely
        def safe_format_traits(traits):
            if pd.isna(traits) or traits is None:
                return "No traits"
            try:
                return format_traits_display(traits)
            except Exception:
                return "Error formatting traits"

        display_data["traits_display"] = display_data["traits"].apply(
            safe_format_traits
        )

        # Select columns for display
        display_columns = [
            "pmid",
            "model",
            "title_short",
            "journal",
            "pub_date",
            "traits_display",
        ]

        # Show table
        st.dataframe(
            display_data[display_columns],
            column_config={
                "pmid": "PMID",
                "model": "Model",
                "title_short": "Title",
                "journal": "Journal",
                "pub_date": "Publication Date",
                "traits_display": "Traits",
            },
            use_container_width=True,
        )

        # Detailed view for selected row
        if st.checkbox("Show detailed view for selected records"):
            selected_pmid = st.selectbox(
                "Select PMID for detailed view:", options=data["pmid"].unique()
            )

            selected_record = data[data["pmid"] == selected_pmid].iloc[0]

            st.subheader(f"Details for PMID: {selected_pmid}")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**Title:**")
                st.write(selected_record["title"])

                st.write("**Journal:**")
                st.write(
                    f"{selected_record['journal']} ({selected_record['pub_date']})"
                )

                st.write("**Model:**")
                st.write(selected_record["model"])

            with col2:
                st.write("**Traits:**")
                traits_text = format_traits_display(selected_record["traits"])
                st.write(traits_text)

                if selected_record["author_affil"]:
                    st.write("**Author Affiliation:**")
                    st.write(selected_record["author_affil"])

            st.write("**Abstract:**")
            st.write(selected_record["abstract"])

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


def show_trait_similarities():
    """Show the trait similarities view."""
    st.header("🔗 Trait Profile Similarities")
    st.markdown("Explore studies with similar trait profiles.")

    # Filters
    col1, col2 = st.columns([1, 1])

    with col1:
        min_similarity = st.slider(
            "Minimum similarity threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Filter similarities above this threshold",
        )

    with col2:
        limit = st.number_input(
            "Max results:", min_value=10, max_value=1000, value=100, step=10
        )

    # Get and display data
    try:
        data = get_trait_similarity_data(
            limit=limit, min_similarity=min_similarity
        )

        if data.empty:
            st.warning("No similarities found above the threshold.")
            return

        st.success(f"Found {len(data)} similarity relationships")

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Similarity Pairs", len(data))
        with col2:
            st.metric(
                "Avg Similarity",
                f"{data['trait_profile_similarity'].mean():.3f}",
            )
        with col3:
            st.metric(
                "Max Similarity",
                f"{data['trait_profile_similarity'].max():.3f}",
            )
        with col4:
            st.metric("Unique Studies", data["query_pmid"].nunique())

        # Similarity distribution
        st.subheader("Similarity Distribution")
        st.bar_chart(
            data["trait_profile_similarity"].value_counts().sort_index(),
            use_container_width=True,
        )

        # Data table
        st.subheader("Similarity Results")

        # Prepare display data
        display_data = data.copy()
        display_data["query_title_short"] = (
            display_data["query_title"].str[:80] + "..."
        )
        display_data["similar_title_short"] = (
            display_data["similar_title"].str[:80] + "..."
        )

        # Select columns for display
        display_columns = [
            "query_pmid",
            "query_model",
            "similar_pmid",
            "similar_model",
            "trait_profile_similarity",
            "trait_jaccard_similarity",
            "query_trait_count",
            "similar_trait_count",
            "query_title_short",
            "similar_title_short",
        ]

        st.dataframe(
            display_data[display_columns],
            column_config={
                "query_pmid": "Query PMID",
                "query_model": "Query Model",
                "similar_pmid": "Similar PMID",
                "similar_model": "Similar Model",
                "trait_profile_similarity": st.column_config.NumberColumn(
                    "Profile Similarity", format="%.3f"
                ),
                "trait_jaccard_similarity": st.column_config.NumberColumn(
                    "Jaccard Similarity", format="%.3f"
                ),
                "query_trait_count": "Query Traits",
                "similar_trait_count": "Similar Traits",
                "query_title_short": "Query Title",
                "similar_title_short": "Similar Title",
            },
            use_container_width=True,
        )

        # Model comparison
        st.subheader("Model Comparison")

        # Group by model pairs
        model_comparison = (
            data.groupby(["query_model", "similar_model"])
            .agg(
                {
                    "trait_profile_similarity": ["count", "mean", "std"],
                    "trait_jaccard_similarity": "mean",
                }
            )
            .round(3)
        )

        model_comparison.columns = [
            "Count",
            "Mean Profile Sim",
            "Std Profile Sim",
            "Mean Jaccard Sim",
        ]

        st.dataframe(model_comparison, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading similarity data: {str(e)}")


def show_about():
    """Show the about page."""
    st.header("ℹ️ About MR-KG Literature Explorer")

    st.markdown("""
    ## Overview

    The MR-KG Literature Explorer is a web application for demonstrating structural
    PubMed literature data extracted by large language models. This tool enables
    researchers to explore trait relationships and discover similar studies based
    on their trait profiles.

    ## Features

    ### 📊 Model Analysis
    - Browse extracted data from PubMed literature analysis
    - Filter results by specific traits of interest
    - View detailed information about each study including:
      - Publication metadata (title, journal, publication date)
      - Abstract content
      - Extracted traits from multiple LLM models
      - Author affiliations

    ### 🔗 Trait Profile Similarities
    - Explore studies with similar trait profiles
    - Compare trait profile and Jaccard similarities
    - Analyze patterns across different LLM models
    - Filter by similarity thresholds

    ## Data Sources

    This application uses two main DuckDB databases:

    - **vector_store.db**: Contains the main extracted data including:
      - Model results from multiple LLMs
      - Trait embeddings and relationships
      - PubMed metadata
      - EFO (Experimental Factor Ontology) mappings

    - **trait_profile_db.db**: Contains trait profile similarity data including:
      - Pairwise similarity calculations between studies
      - Query combinations and rankings
      - Model-specific similarity statistics

    ## Technology Stack

    - **Frontend**: Streamlit
    - **Database**: DuckDB
    - **Data Processing**: Pandas
    - **Deployment**: UV package manager

    ## Usage Tips

    1. **Model Analysis**: Start by exploring the model analysis view to understand
       the scope of available data. Use trait filters to narrow down to specific
       areas of interest.

    2. **Trait Similarities**: Use the similarity view to discover related studies.
       Adjust the similarity threshold to control the sensitivity of matches.

    3. **Performance**: The application uses caching to improve performance.
       Large queries may take a moment to load initially.

    ## Contact

    For questions about this application or the underlying research, please refer
    to the project documentation or contact the development team.
    """)


if __name__ == "__main__":
    main()
