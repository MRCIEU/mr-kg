"""Search by Trait page.

Find studies investigating a specific trait (exposure or outcome).
"""

import streamlit as st

from components.model_selector import model_selector
from components.study_table import study_table
from components.theme import apply_theme
from services.db_client import (
    autocomplete_traits,
    filter_studies_by_similarity,
    search_studies,
)

st.set_page_config(
    page_title="Search by Trait - MR-KG",
    page_icon=None,
    layout="wide",
)

# ---- Apply theme ----
apply_theme()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("[API Documentation](/mr-kg/api/docs)")


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

        # ---- Similarity filters ----
        st.markdown("**Filters**")
        require_trait_sim = st.checkbox(
            "Has related studies by trait",
            value=False,
            key="trait_search_require_trait_sim",
            help="Only show studies with related studies by trait similarity",
        )
        require_evidence_sim = st.checkbox(
            "Has related studies by evidence",
            value=False,
            key="trait_search_require_evidence_sim",
            help="Only show studies with related studies by evidence similarity",
        )

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
            studies = results["studies"]

            # ---- Apply similarity filters ----
            if require_trait_sim or require_evidence_sim:
                with st.spinner("Filtering by similarity..."):
                    studies = filter_studies_by_similarity(
                        studies=studies,
                        model=selected_model,
                        require_trait_similarity=require_trait_sim,
                        require_evidence_similarity=require_evidence_sim,
                    )

            total = len(studies)
            original_total = results.get("total", len(results["studies"]))

            if require_trait_sim or require_evidence_sim:
                st.write(
                    f"Found {total} studies (filtered from {original_total})"
                )
            else:
                st.write(f"Found {total} studies")

            if studies:
                # Display study table and handle selection
                selected_pmid = study_table(studies)

                if selected_pmid:
                    # Store in session state and navigate to study info page
                    st.session_state["selected_pmid"] = selected_pmid
                    st.session_state["selected_model"] = selected_model
                    st.switch_page("pages/3_Study_Info.py")
            else:
                st.info("No studies match the selected filters.")
        else:
            st.info(
                f"No studies found for trait '{selected_trait}' "
                f"with model '{selected_model}'."
            )


if __name__ == "__main__":
    main()
