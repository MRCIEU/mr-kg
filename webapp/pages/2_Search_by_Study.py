"""Search by Study page.

Find studies by title text search.
"""

import streamlit as st

from components.model_selector import model_selector
from components.theme import apply_theme
from services.db_client import (
    autocomplete_studies,
    filter_studies_by_similarity,
    search_study_by_pmid,
)

st.set_page_config(
    page_title="Search by Study - MR-KG",
    page_icon=None,
    layout="wide",
)

# ---- Apply theme ----
apply_theme()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("[API](/mr-kg/api)")


def main() -> None:
    """Render the search by study page."""
    st.title("Search by Study")

    st.markdown(
        "Find studies by title or PMID. "
        "Type to search and select a study to view its details."
    )

    # ---- Search controls ----
    col1, col2 = st.columns([3, 1])

    with col2:
        selected_model = model_selector(key="study_search_model")

        # ---- Similarity filters ----
        st.markdown("**Filters**")
        require_trait_sim = st.checkbox(
            "Has related studies by trait",
            value=False,
            key="study_search_require_trait_sim",
            help="Only show studies with related studies by trait similarity",
        )
        require_evidence_sim = st.checkbox(
            "Has related studies by evidence",
            value=False,
            key="study_search_require_evidence_sim",
            help="Only show studies with related studies by evidence similarity",
        )

    with col1:
        # ---- Search mode toggle ----
        search_mode = st.radio(
            "Search by",
            options=["Study Title", "PMID"],
            horizontal=True,
            key="study_search_mode",
            help="Toggle between fuzzy title search or exact PMID lookup",
        )

        # ---- Study search input ----
        if search_mode == "Study Title":
            search_term = st.text_input(
                "Enter study title",
                placeholder="e.g., Mendelian randomization, body mass index",
                key="study_search_input",
                help="Type at least 2 characters to see suggestions",
            )
        else:
            search_term = st.text_input(
                "Enter PMID",
                placeholder="e.g., 12345678",
                key="study_search_pmid_input",
                help="Enter the exact PubMed ID",
            )

    # ---- Display search results ----
    if search_mode == "Study Title":
        # Fuzzy search mode - requires at least 2 characters
        if search_term and len(search_term) >= 2:
            with st.spinner("Searching..."):
                suggestions = autocomplete_studies(
                    search_term, model=selected_model, limit=20
                )
            _display_results(
                suggestions,
                selected_model,
                require_trait_sim,
                require_evidence_sim,
            )
        elif search_term:
            st.info("Please enter at least 2 characters to search.")
    else:
        # PMID exact match mode
        if search_term:
            with st.spinner("Searching..."):
                suggestions = search_study_by_pmid(
                    search_term, model=selected_model
                )
            _display_results(
                suggestions,
                selected_model,
                require_trait_sim,
                require_evidence_sim,
            )


def _display_results(
    suggestions: list[dict],
    selected_model: str,
    require_trait_sim: bool,
    require_evidence_sim: bool,
) -> None:
    """Display search results with optional filtering.

    Args:
        suggestions: List of study dicts from search
        selected_model: Selected extraction model
        require_trait_sim: Filter by trait similarity
        require_evidence_sim: Filter by evidence similarity
    """
    if suggestions:
        # ---- Apply similarity filters ----
        studies = suggestions
        if require_trait_sim or require_evidence_sim:
            with st.spinner("Filtering by similarity..."):
                studies = filter_studies_by_similarity(
                    studies=studies,
                    model=selected_model,
                    require_trait_similarity=require_trait_sim,
                    require_evidence_similarity=require_evidence_sim,
                )

        st.divider()
        st.subheader("Search Results")

        original_total = len(suggestions)
        total = len(studies)

        if require_trait_sim or require_evidence_sim:
            st.write(
                f"Found {total} matching studies (filtered from {original_total})"
            )
        else:
            st.write(f"Found {total} matching studies")

        if studies:
            for i, study in enumerate(studies):
                pmid = study.get("pmid", "")
                title = study.get("title", "")

                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{pmid}**")
                    st.write(_truncate_text(title, 100))
                with col2:
                    if st.button("View", key=f"study_btn_{i}_{pmid}"):
                        # Store in session state and navigate to study info page
                        st.session_state["selected_pmid"] = pmid
                        st.session_state["selected_model"] = selected_model
                        st.switch_page("pages/3_Study_Info.py")

                st.divider()
        else:
            st.info("No studies match the selected filters.")
    else:
        st.info(f"No matching studies found for model '{selected_model}'.")


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


if __name__ == "__main__":
    main()
