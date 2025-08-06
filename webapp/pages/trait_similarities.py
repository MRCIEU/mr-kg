"""Trait similarities page for the MR-KG web app."""

import streamlit as st
from resources.database import get_trait_similarity_data


def show_trait_similarities():
    """Show the trait similarities view."""
    st.header("Trait Profile Similarities")
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


show_trait_similarities()
