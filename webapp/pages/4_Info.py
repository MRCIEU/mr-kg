"""Info page.

Display overarching resource statistics and documentation.
"""

import streamlit as st

from components.model_selector import AVAILABLE_MODELS
from services.db_client import get_statistics

st.set_page_config(
    page_title="Resource Info - MR-KG",
    page_icon=None,
    layout="wide",
)


def main() -> None:
    """Render the info page."""
    st.title("MR-KG Resource Information")

    st.markdown(
        "Overview of the MR-KG resource, including statistics, "
        "available models, and methodology documentation."
    )

    # ---- Fetch statistics ----
    with st.spinner("Fetching resource statistics from database..."):
        stats = get_statistics()

    if stats is None:
        st.error(
            "Unable to load statistics. "
            "Please ensure the databases are accessible."
        )
        return

    # ---- Overall Statistics ----
    st.subheader("Overall Statistics")

    overall = stats.get("overall", {})
    if overall:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            _display_metric(
                "Total Papers",
                overall.get("total_papers"),
            )
        with col2:
            _display_metric(
                "Unique Traits",
                overall.get("total_traits"),
            )
        with col3:
            _display_metric(
                "Extraction Models",
                overall.get("total_models"),
            )
        with col4:
            _display_metric(
                "Total Extractions",
                overall.get("total_extractions"),
            )
    else:
        st.info("Overall statistics not available.")

    st.divider()

    # ---- Model Similarity Statistics ----
    st.subheader("Model Statistics (Trait Similarity)")

    model_sim_stats = stats.get("model_similarity_stats", [])
    if model_sim_stats:
        # Build table data
        table_data = []
        for row in model_sim_stats:
            table_data.append(
                {
                    "Model": row.get("model", "N/A"),
                    "Extractions": _format_number(row.get("extractions")),
                    "Avg Traits": _format_float(row.get("avg_traits")),
                    "Similarities": _format_number(row.get("similarities")),
                }
            )

        # Display as dataframe
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Model similarity statistics not available.")

    st.divider()

    # ---- Model Evidence Statistics ----
    st.subheader("Model Statistics (Evidence Similarity)")

    model_ev_stats = stats.get("model_evidence_stats", [])
    if model_ev_stats:
        # Build table data
        table_data = []
        for row in model_ev_stats:
            table_data.append(
                {
                    "Model": row.get("model", "N/A"),
                    "Extractions": _format_number(row.get("extractions")),
                    "Avg Results": _format_float(row.get("avg_results")),
                    "Similarities": _format_number(row.get("similarities")),
                }
            )

        # Display as dataframe
        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Model evidence statistics not available.")

    st.divider()

    # ---- Available Models ----
    st.subheader("Available Extraction Models")

    st.write("The following LLM models are available for extraction:")

    models_text = ", ".join(AVAILABLE_MODELS)
    st.markdown(f"**Models:** {models_text}")

    st.markdown("**Default model:** gpt-5")

    st.divider()

    # ---- Methodology Documentation ----
    st.subheader("Methodology")

    st.markdown("""
**Trait Profile Similarity**

- Measures research focus overlap between studies
- Based on cosine similarity of trait embedding vectors
- Jaccard similarity measures exact trait overlap
- Range: 0-1 (higher = more similar)

**Evidence Profile Similarity**

- Measures statistical evidence alignment between studies
- Based on agreement in effect direction classifications
- Direction concordance ranges from -1 to +1
- Positive values indicate concordant findings across matched pairs
    """)

    st.divider()

    # ---- External Links ----
    st.subheader("Documentation")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "[Trait Profile Similarity Documentation]"
            "(https://github.com/MRCIEU/mr-kg/blob/main/"
            "docs/processing/trait-profile-similarity.md)"
        )
    with col2:
        st.markdown(
            "[Evidence Profile Similarity Documentation]"
            "(https://github.com/MRCIEU/mr-kg/blob/main/"
            "docs/processing/evidence-profile-similarity.md)"
        )


def _display_metric(label: str, value: int | float | None) -> None:
    """Display a metric value.

    Args:
        label: Metric label
        value: Metric value (can be None)
    """
    if value is not None:
        if isinstance(value, int):
            st.metric(label, f"{value:,}")
        else:
            st.metric(label, f"{value:,.2f}")
    else:
        st.metric(label, "N/A")


def _format_number(value: int | None) -> str:
    """Format a number with thousands separator.

    Args:
        value: Number to format

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    return f"{value:,}"


def _format_float(value: float | None, decimals: int = 2) -> str:
    """Format a float with specified decimals.

    Args:
        value: Float to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


if __name__ == "__main__":
    main()
