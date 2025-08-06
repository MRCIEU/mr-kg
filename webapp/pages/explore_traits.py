"""Explore traits page for the MR-KG web app.

This page provides an interactive interface to explore trait labels from the model_result_traits table.
Features a two-column layout with trait filtering on the left and reserved space for future features on the right.
"""

import streamlit as st
import pandas as pd

from resources.database import get_database_connections


@st.cache_data
def get_all_trait_labels() -> pd.DataFrame:
    """Get all unique trait labels from the model_result_traits table."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT DISTINCT trait_label
    FROM model_result_traits
    ORDER BY trait_label
    """
    result = vector_conn.execute(query).fetchall()
    return pd.DataFrame(result, columns=['trait_label'])


@st.cache_data
def filter_traits(trait_df: pd.DataFrame, filter_text: str) -> pd.DataFrame:
    """Filter traits based on user input."""
    if not filter_text:
        return trait_df

    # Case-insensitive filtering
    filtered_df = trait_df[
        trait_df['trait_label'].str.contains(filter_text, case=False, na=False)
    ]
    return filtered_df


def main():
    """Main function for the explore traits page."""
    st.set_page_config(
        page_title="Explore Traits - MR-KG",
        layout="wide"
    )

    st.title("Explore Traits")
    st.markdown("""
    Explore the trait labels extracted from the model results. Use the filter box to search for specific traits.
    """)

    # Load trait data
    try:
        trait_df = get_all_trait_labels()
    except Exception as e:
        st.error(f"Error loading trait data: {str(e)}")
        return

    # Create two columns
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.header("Trait Labels")

        # Filter input
        filter_text = st.text_input(
            "Filter traits:",
            placeholder="Type to filter trait labels...",
            help="Search is case-insensitive and searches within trait labels"
        )

        # Apply filtering
        filtered_traits = filter_traits(trait_df, filter_text)

        # Display results summary
        total_traits = len(trait_df)
        filtered_count = len(filtered_traits)

        if filter_text:
            st.info(f"Showing {filtered_count} of {total_traits} traits matching '{filter_text}'")
        else:
            st.info(f"Showing all {total_traits} unique trait labels")

        # Display trait table
        if not filtered_traits.empty:
            # Configure table display
            st.dataframe(
                filtered_traits,
                column_config={
                    "trait_label": st.column_config.TextColumn(
                        "Trait Label",
                        help="Unique trait labels extracted from model results",
                        width="large"
                    )
                },
                use_container_width=True,
                height=600,
                hide_index=True
            )
        else:
            st.warning("No traits found matching the filter criteria.")
            if filter_text:
                st.markdown("**Try:**")
                st.markdown("- Checking your spelling")
                st.markdown("- Using fewer or different keywords")
                st.markdown("- Clearing the filter to see all traits")

    with right_col:
        st.header("Additional Features")
        st.info("This space is reserved for future features")
        st.markdown("""
        **Coming soon:**
        - Trait similarity analysis
        - Related trait suggestions
        - Trait frequency statistics
        - Export functionality
        """)


if __name__ == "__main__":
    main()
