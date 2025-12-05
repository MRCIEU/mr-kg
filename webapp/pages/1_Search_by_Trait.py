"""Search by Trait page.

Find studies investigating a specific trait (exposure or outcome).
"""

import streamlit as st

from components.model_selector import model_selector
from components.study_table import study_table
from services.db_client import autocomplete_traits, search_studies

st.set_page_config(
    page_title="Search by Trait - MR-KG",
    page_icon=None,
    layout="wide",
)


def main() -> None:
    """Render the search by trait page."""
    st.title("Search by Trait")

    st.markdown(
        "Find studies investigating a specific trait. "
        "A trait can appear as either an exposure or outcome in MR studies."
    )

    # ---- Search controls ----
    col1, col2 = st.columns([3, 1])

    with col2:
        selected_model = model_selector(key="trait_search_model")

    with col1:
        # ---- Trait search with autocomplete ----
        search_term = st.text_input(
            "Enter trait name",
            placeholder="e.g., body mass index, blood pressure",
            key="trait_search_input",
            help="Type at least 2 characters to see suggestions",
        )

        # Get autocomplete suggestions
        selected_trait = None
        if search_term and len(search_term) >= 2:
            suggestions = autocomplete_traits(
                search_term, model=selected_model, limit=20
            )
            if suggestions:
                selected_trait = st.selectbox(
                    "Select trait",
                    options=[""] + suggestions,
                    key="trait_select",
                    help="Select a trait from the suggestions",
                )
            else:
                st.info(
                    f"No matching traits found for model '{selected_model}'."
                )

    # ---- Search and display results ----
    if selected_trait:
        st.divider()
        st.subheader(f"Studies for: {selected_trait}")

        with st.spinner("Searching studies..."):
            results = search_studies(
                trait=selected_trait,
                model=selected_model,
                limit=50,
            )

        if results and results.get("studies"):
            total = results.get("total", len(results["studies"]))
            st.write(f"Found {total} studies")

            # Display study table and handle selection
            selected_pmid = study_table(results["studies"])

            if selected_pmid:
                # Store in session state and navigate to study info page
                st.session_state["selected_pmid"] = selected_pmid
                st.session_state["selected_model"] = selected_model
                st.switch_page("pages/3_Study_Info.py")
        else:
            st.info(
                f"No studies found for trait '{selected_trait}' "
                f"with model '{selected_model}'."
            )


if __name__ == "__main__":
    main()
