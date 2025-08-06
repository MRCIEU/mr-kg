"""Model analysis page for the MR-KG web app."""

import streamlit as st
from resources.database import get_available_traits, get_pmid_model_data
from funcs.utils import format_traits_display, safe_format_traits


def show_model_analysis():
    """Show the model analysis view."""
    st.header("Model Analysis")
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
