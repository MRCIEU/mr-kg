"""Search by Study page.

Find studies by title text search.
"""

import streamlit as st

from components.model_selector import model_selector
from components.theme import apply_theme, theme_toggle
from services.db_client import (
    autocomplete_studies,
    filter_studies_by_similarity,
)

st.set_page_config(
    page_title="Search by Study - MR-KG",
    page_icon=None,
    layout="wide",
)

# ---- Apply theme ----
apply_theme()
theme_toggle()


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
        # ---- Study search with autocomplete ----
        search_term = st.text_input(
            "Enter study title or PMID",
            placeholder="e.g., Mendelian randomization, body mass index",
            key="study_search_input",
            help="Type at least 2 characters to see suggestions",
        )

    # ---- Display autocomplete results ----
    if search_term and len(search_term) >= 2:
        with st.spinner("Searching..."):
            suggestions = autocomplete_studies(
                search_term, model=selected_model, limit=20
            )

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
    elif search_term:
        st.info("Please enter at least 2 characters to search.")


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
