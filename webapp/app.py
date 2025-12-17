"""MR-KG webapp landing page.

This is the main entry point for the Streamlit webapp, providing navigation
to search pages and resource information.
"""

import streamlit as st

from components.theme import apply_theme
from services.db_client import get_statistics

# ---- Page configuration ----
st.set_page_config(
    page_title="MR-KG: A knowledge graph of Mendelian randomization evidence powered by large language models",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Apply theme ----
apply_theme()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("[API Documentation](/mr-kg/api/docs)")

# ---- Page content ----


def main() -> None:
    """Render the landing page."""
    st.title(
        "MR-KG: A knowledge graph of Mendelian randomization evidence powered by large language models"
    )

    # ---- Resource description ----
    st.markdown("""
MR-KG is a resource for exploring Mendelian Randomization studies through
LLM-extracted trait information and vector similarity search.
The database contains extraction results from multiple large language models,
enabling comparison of trait profiles and evidence patterns across studies.
    """)

    # ---- Key statistics ----
    st.subheader("Key Statistics")

    stats = get_statistics()
    if stats and "overall" in stats:
        overall = stats["overall"]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Studies",
                f"{overall.get('total_papers', 'N/A'):,}"
                if isinstance(overall.get("total_papers"), int)
                else overall.get("total_papers", "N/A"),
            )
        with col2:
            st.metric(
                "Unique Traits",
                f"{overall.get('total_traits', 'N/A'):,}"
                if isinstance(overall.get("total_traits"), int)
                else overall.get("total_traits", "N/A"),
            )
        with col3:
            st.metric(
                "Extraction Models",
                overall.get("total_models", "N/A"),
            )
        with col4:
            st.metric(
                "Total Extractions",
                f"{overall.get('total_extractions', 'N/A'):,}"
                if isinstance(overall.get("total_extractions"), int)
                else overall.get("total_extractions", "N/A"),
            )
    else:
        st.info("Statistics unavailable. Please ensure the API is running.")

    st.divider()

    # ---- Navigation buttons ----
    st.subheader("Explore the Resource")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Search by Trait")
        st.markdown(
            "Find studies investigating a specific trait "
            "(exposure or outcome)."
        )
        if st.button("Search by Trait", key="nav_search_trait"):
            st.switch_page("pages/1_Search_by_Trait.py")

    with col2:
        st.markdown("#### Search by Study")
        st.markdown("Find studies by title or PMID.")
        if st.button("Search by Study", key="nav_search_study"):
            st.switch_page("pages/2_Search_by_Study.py")

    st.divider()

    col3, _ = st.columns(2)
    with col3:
        st.markdown("#### Resource Information")
        st.markdown("View statistics, available models, and documentation.")
        if st.button("Resource Info", key="nav_info"):
            st.switch_page("pages/4_Info.py")

    st.divider()

    # ---- API access ----
    st.subheader("API Access")

    col1, _ = st.columns(2)
    with col1:
        st.markdown("#### REST API")
        st.markdown(
            "Access the data programmatically via the REST API."
        )
        st.markdown("[View API Documentation](/mr-kg/api)")

    st.divider()

    # ---- External links ----
    st.subheader("Links")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[GitHub Repository](https://github.com/MRCIEU/mr-kg)")
    with col2:
        st.markdown(
            "[Documentation](https://github.com/MRCIEU/mr-kg/blob/main/README.md)"
        )


if __name__ == "__main__":
    main()
