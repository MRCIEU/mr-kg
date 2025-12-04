"""Study table component for displaying search results."""

import streamlit as st


def study_table(studies: list[dict]) -> str | None:
    """Display a table of studies with selection capability.

    Args:
        studies: List of study dicts with pmid, title, pub_date, journal

    Returns:
        Selected PMID if a study is clicked, None otherwise
    """
    if not studies:
        st.info("No studies found.")
        return None

    # ---- Build table data ----
    table_data = []
    for study in studies:
        pub_date = study.get("pub_date", "")
        year = pub_date[:4] if pub_date else "N/A"
        table_data.append(
            {
                "PMID": study.get("pmid", ""),
                "Title": _truncate_text(study.get("title", ""), 80),
                "Year": year,
                "Journal": _truncate_text(
                    study.get("journal", "") or "N/A", 30
                ),
            }
        )

    # ---- Display count ----
    st.write(f"Showing {len(studies)} studies")

    # ---- Create selection interface ----
    selected_pmid = None

    for i, study in enumerate(studies):
        pmid = study.get("pmid", "")
        title = study.get("title", "")
        pub_date = study.get("pub_date", "")
        year = pub_date[:4] if pub_date else "N/A"
        journal = study.get("journal", "") or "N/A"

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{pmid}** - {_truncate_text(title, 70)}")
            st.caption(f"{year} | {journal}")
        with col2:
            if st.button("View", key=f"study_btn_{i}_{pmid}"):
                selected_pmid = pmid

        st.divider()

    return selected_pmid


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
