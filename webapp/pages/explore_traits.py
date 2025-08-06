"""Explore traits page for the MR-KG web app.

This page provides an interactive interface to explore trait labels from the model_result_traits table.
Features a two-column layout with trait filtering on the left and reserved space for future features on the right.
"""

import streamlit as st
import pandas as pd

from resources.database import get_database_connections


@st.cache_data
def get_all_trait_labels() -> pd.DataFrame:
    """Get all unique trait labels from the trait_stats view."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT trait_label, appearance_count
    FROM trait_stats
    ORDER BY appearance_count DESC
    """
    result = vector_conn.execute(query).fetchall()
    return pd.DataFrame(result, columns=["trait_label", "appearance_count"])


@st.cache_data
def filter_traits(trait_df: pd.DataFrame, filter_text: str) -> pd.DataFrame:
    """Filter traits based on user input."""
    if not filter_text:
        return trait_df

    # Case-insensitive filtering
    filtered_df = trait_df[
        trait_df["trait_label"].str.contains(filter_text, case=False, na=False)
    ]
    return filtered_df


def render_trait_item(trait_label: str, appearance_count: int, idx: int) -> None:
    """Render a single trait item with select button."""
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            # Check if this is the selected trait and highlight it
            if st.session_state.selected_trait == trait_label:
                st.markdown(f"🔹 **{trait_label}** _(selected)_")
                st.markdown(f"<small style='color: #666;'>Appears {appearance_count:,} times</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"• {trait_label}")
                st.markdown(f"<small style='color: #666;'>Appears {appearance_count:,} times</small>", unsafe_allow_html=True)

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
                else:
                    # Select this trait
                    st.session_state.selected_trait = trait_label
                st.rerun()

        # Add a subtle separator
        st.markdown("---", unsafe_allow_html=True)


def render_trait_list(
    filtered_traits: pd.DataFrame, total_traits: int, filter_text: str
) -> None:
    """Render the trait list section with filtering and selection."""
    filtered_count = len(filtered_traits)

    # Display results summary
    if filter_text:
        st.info(
            f"Showing {filtered_count} of {total_traits} traits matching '{filter_text}'"
        )
    else:
        st.info(f"Showing all {total_traits} unique trait labels")

    # Display trait table with buttons
    if not filtered_traits.empty:
        # Create a container for the trait list with scrolling
        with st.container(height=600):
            st.markdown("**Click 'Select' to view trait details →**")

            for idx, (_, row) in enumerate(filtered_traits.iterrows()):
                trait_label = row["trait_label"]
                appearance_count = row["appearance_count"]
                render_trait_item(trait_label, appearance_count, idx)
    else:
        render_no_results_message(filter_text)


def render_no_results_message(filter_text: str) -> None:
    """Render message when no traits match the filter."""
    st.warning("No traits found matching the filter criteria.")
    if filter_text:
        st.markdown("**Try:**")
        st.markdown("- Checking your spelling")
        st.markdown("- Using fewer or different keywords")
        st.markdown("- Clearing the filter to see all traits")


def render_left_column(trait_df: pd.DataFrame) -> None:
    """Render the left column with trait filtering and selection."""
    st.header("Trait Labels")

    # Filter input
    filter_text = st.text_input(
        "Filter traits:",
        placeholder="Type to filter trait labels...",
        help="Search is case-insensitive and searches within trait labels",
    )

    # Apply filtering
    filtered_traits = filter_traits(trait_df, filter_text)
    total_traits = len(trait_df)

    # Render trait list
    render_trait_list(filtered_traits, total_traits, filter_text)


def render_selected_trait_details() -> None:
    """Render details for the currently selected trait."""
    st.success(f"**Selected trait:** {st.session_state.selected_trait}")

    # Add a clear button
    if st.button("Clear Selection"):
        st.session_state.selected_trait = None
        st.rerun()

    # Placeholder for future trait analysis features
    st.markdown("---")
    st.subheader("Trait Analysis")
    st.info("Analysis features for this trait will be added here.")


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

    # Load trait data
    try:
        trait_df = get_all_trait_labels()
    except Exception as e:
        st.error(f"Error loading trait data: {str(e)}")
        return

    # Create two columns layout
    left_col, right_col = st.columns([1, 1])

    # Render left column (trait list and filtering)
    with left_col:
        render_left_column(trait_df)

    # Render right column (selected trait details)
    with right_col:
        render_right_column()


if __name__ == "__main__":
    main()
