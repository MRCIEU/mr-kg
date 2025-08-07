"""Explore traits page for the MR-KG web app.

This page provides an interactive interface to explore trait labels from the model_result_traits table.
Features a two-column layout with trait filtering on the left and reserved space for future features on the right.
"""

import streamlit as st
import pandas as pd

from resources.database import get_database_connections


@st.cache_data
def get_top_trait_labels(limit: int = 50) -> pd.DataFrame:
    """Get the top N trait labels ordered by appearance count."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT trait_label, appearance_count
    FROM trait_stats
    ORDER BY appearance_count DESC
    LIMIT ?
    """
    result = vector_conn.execute(query, [limit]).fetchall()
    return pd.DataFrame(result, columns=["trait_label", "appearance_count"])


@st.cache_data
def search_traits(filter_text: str, limit: int = 100) -> pd.DataFrame:
    """Search for traits matching the filter text using database query."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT trait_label, appearance_count
    FROM trait_stats
    WHERE trait_label ILIKE ?
    ORDER BY appearance_count DESC
    LIMIT ?
    """
    # Use SQL ILIKE for case-insensitive pattern matching
    search_pattern = f"%{filter_text}%"
    result = vector_conn.execute(query, [search_pattern, limit]).fetchall()
    return pd.DataFrame(result, columns=["trait_label", "appearance_count"])


def get_trait_data(filter_text: str = "") -> pd.DataFrame:
    """Get trait data based on filter text - either top traits or filtered results."""
    if not filter_text:
        return get_top_trait_labels()
    else:
        return search_traits(filter_text)


@st.cache_data
def get_studies_for_trait_and_model(
    trait_label: str, model: str
) -> pd.DataFrame:
    """Get studies that contain the specified trait from the specified model."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT DISTINCT
        mr.id as model_result_id,
        mr.pmid,
        pubmed.title,
        pubmed.journal,
        pubmed.pub_date,
        mr.metadata
    FROM model_results mr
    JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
    LEFT JOIN mr_pubmed_data pubmed ON mr.pmid = pubmed.pmid
    WHERE mrt.trait_label = ? 
    AND mr.model = ?
    ORDER BY pubmed.pub_date DESC, mr.pmid
    """
    result = vector_conn.execute(query, [trait_label, model]).fetchall()
    return pd.DataFrame(
        result,
        columns=[
            "model_result_id",
            "pmid",
            "title",
            "journal",
            "pub_date",
            "metadata",
        ],
    )


def render_trait_item(
    trait_label: str, appearance_count: int, idx: int
) -> None:
    """Render a single trait item with select button."""
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            # Check if this is the selected trait and highlight it
            if st.session_state.selected_trait == trait_label:
                st.markdown(f"🔹 **{trait_label}** _(selected)_")
                st.markdown(
                    f"<small style='color: #666;'>Appears {appearance_count:,} times</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"• {trait_label}")
                st.markdown(
                    f"<small style='color: #666;'>Appears {appearance_count:,} times</small>",
                    unsafe_allow_html=True,
                )

        with col2:
            button_label = (
                "✓"
                if st.session_state.selected_trait == trait_label
                else "Select"
            )
            if st.button(
                button_label,
                key=f"btn_{idx}_{trait_label}",
                use_container_width=True,
            ):
                if st.session_state.selected_trait == trait_label:
                    # If already selected, deselect
                    st.session_state.selected_trait = None
                    st.session_state.show_results = False
                    st.session_state.expand_panels = True
                else:
                    # Select this trait
                    st.session_state.selected_trait = trait_label
                    st.session_state.show_results = (
                        False  # Reset results when selecting new trait
                    )
                    # Keep panels open when selecting a trait
                st.rerun()

        # Add a subtle separator
        st.markdown("---", unsafe_allow_html=True)


def render_trait_list(trait_df: pd.DataFrame) -> None:
    """Render the trait list section with selection."""
    # Display trait table with buttons
    if not trait_df.empty:
        # Create a container for the trait list with scrolling
        with st.container(height=600):
            st.markdown("**Click 'Select' to view trait details →**")

            for idx, (_, row) in enumerate(trait_df.iterrows()):
                trait_label = row["trait_label"]
                appearance_count = row["appearance_count"]
                render_trait_item(trait_label, appearance_count, idx)
    else:
        st.warning("No traits found matching the filter criteria.")
        st.markdown("**Try:**")
        st.markdown("- Checking your spelling")
        st.markdown("- Using fewer or different keywords")
        st.markdown("- Clearing the filter to see top traits")


def render_left_column() -> None:
    """Render the left column with trait filtering and selection."""
    st.header("Trait Labels")

    # Filter input
    filter_text = st.text_input(
        "Filter traits:",
        placeholder="Type to filter trait labels...",
        help="Search is case-insensitive and searches within trait labels",
    )

    # Get trait data based on filter
    try:
        trait_df = get_trait_data(filter_text)
    except Exception as e:
        st.error(f"Error loading trait data: {str(e)}")
        return

    # Display results summary
    if filter_text:
        st.info(
            f"Found {len(trait_df)} traits matching '{filter_text}' (showing up to 100)"
        )
    else:
        st.info("Showing top 50 most frequent trait labels")

    # Render trait list
    render_trait_list(trait_df)


def render_selected_trait_details() -> None:
    """Render details for the currently selected trait."""
    st.success(f"**Selected trait:** {st.session_state.selected_trait}")

    # Add a clear button
    if st.button("Clear Selection"):
        st.session_state.selected_trait = None
        st.session_state.show_results = False
        st.session_state.expand_panels = (
            True  # Reopen expansion panels when clearing
        )
        st.rerun()

    # Model selection dropdown
    st.markdown("---")
    st.subheader("Model Selection")

    selected_model = st.selectbox(
        "Select a model:",
        options=st.session_state.models,
        index=st.session_state.models.index(st.session_state.selected_model)
        if st.session_state.selected_model in st.session_state.models
        else 0,
        key="model_selector",
    )

    # Update session state when model changes
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model

    # Show extracted results button
    if st.button("Show Extracted Results", type="primary"):
        st.session_state.show_results = True
        st.session_state.expand_panels = False  # Close expansion panels
        st.rerun()

    # Don't display results here anymore - they'll be shown outside the expansion panels


def render_trait_results(trait_label: str, model: str) -> None:
    """Render the list of studies containing the selected trait from the selected model."""
    st.markdown("---")
    st.subheader("Extracted Results")
    st.markdown(
        f"Studies where **{model}** extracted the trait: **{trait_label}**"
    )

    try:
        # Get the studies data
        studies_df = get_studies_for_trait_and_model(trait_label, model)

        if studies_df.empty:
            st.warning(
                f"No studies found where {model} extracted the trait '{trait_label}'"
            )
            return

        st.info(f"Found {len(studies_df)} studies")

        # Display studies in a scrollable container
        with st.container(height=400):
            for idx, (_, row) in enumerate(studies_df.iterrows()):
                with st.expander(
                    f"Study {idx + 1}: {row['pmid']} {row['title']}", expanded=False
                ):
                    st.markdown(f"**PMID:** {row['pmid']}")

                    if pd.notna(row["title"]):
                        st.markdown(f"**Title:** {row['title']}")

                    if pd.notna(row["journal"]):
                        st.markdown(f"**Journal:** {row['journal']}")

                    if pd.notna(row["pub_date"]):
                        st.markdown(f"**Publication Date:** {row['pub_date']}")

                    # Add a link to PubMed if PMID is available
                    if pd.notna(row["pmid"]):
                        pubmed_url = (
                            f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/"
                        )
                        st.markdown(
                            f"**PubMed Link:** [View on PubMed]({pubmed_url})"
                        )

                    # Show metadata if available
                    if pd.notna(row["metadata"]) and row["metadata"]:
                        with st.expander("Model Metadata", expanded=False):
                            st.json(row["metadata"])

    except Exception as e:
        st.error(f"Error loading study results: {str(e)}")

    # Add button to hide results
    if st.button("Hide Results"):
        st.session_state.show_results = False
        st.session_state.expand_panels = True  # Reopen expansion panels
        st.rerun()


def render_right_column() -> None:
    """Render the right column with selected trait details and future features."""
    st.header("Selected Trait")

    if st.session_state.selected_trait:
        render_selected_trait_details()
    else:
        st.info(
            "Click 'Select' next to any trait in the left column to view details here."
        )

    # Future features section
    st.markdown("---")
    st.subheader("Future Features")
    st.markdown("""
    **Coming soon:**
    - Trait similarity analysis
    - Related trait suggestions
    - Trait frequency statistics
    - Export functionality
    """)


def main():
    """Main function for the explore traits page."""
    st.set_page_config(page_title="Explore Traits - MR-KG", layout="wide")

    st.title("Explore Traits")
    st.markdown("""
    Explore the trait labels extracted from the model results. Use the filter box to search for specific traits.
    """)

    # Initialize session state for selected trait
    if "selected_trait" not in st.session_state:
        st.session_state.selected_trait = None

    # Initialize session state for models
    if "models" not in st.session_state:
        st.session_state.models = [
            "gpt-4-1",
            "gpt-3.5-turbo",
            "claude-3",
            "gemini-pro",
        ]

    # Initialize session state for selected model
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4-1"

    # Initialize session state for showing results
    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    # Initialize session state for expansion panels
    if "expand_panels" not in st.session_state:
        st.session_state.expand_panels = True

    # Create expansion panels for the main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        with st.expander(
            "🔍 Trait Selection", expanded=st.session_state.expand_panels
        ):
            render_left_column()

    with col2:
        with st.expander(
            "⚙️ Trait Analysis", expanded=st.session_state.expand_panels
        ):
            render_right_column()

    # Display extracted results outside of expansion panels
    if (
        st.session_state.get("show_results", False)
        and st.session_state.selected_trait
        and st.session_state.selected_model
    ):
        render_trait_results(
            st.session_state.selected_trait, st.session_state.selected_model
        )


if __name__ == "__main__":
    main()
