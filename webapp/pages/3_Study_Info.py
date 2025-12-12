"""Study Info page.

Display comprehensive information about a selected study with collapsible
panels for different data views.

This page is hidden from the sidebar and only accessible via navigation
from search pages.
"""

import streamlit as st

from components.model_selector import model_selector
from components.similarity_display import (
    evidence_similarity_table,
    trait_similarity_table,
)
from components.theme import apply_theme, theme_toggle
from services.db_client import (
    get_extraction,
    get_similar_by_evidence,
    get_similar_by_trait,
)

# Hide this page from the sidebar
st.set_page_config(
    page_title="Study Info - MR-KG",
    page_icon=None,
    layout="wide",
)

# ---- Apply theme ----
apply_theme()
theme_toggle()

# CSS to hide this page from the sidebar navigation
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] li:has(a[href*="3_Study_Info"]) {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Constants ----
ABSTRACT_TRUNCATE_LENGTH = 500


def main() -> None:
    """Render the study info page."""
    # ---- Get study info from session state ----
    pmid = st.session_state.get("selected_pmid", "")
    model = st.session_state.get("selected_model", "gpt-5")

    if not pmid:
        st.warning("No study selected. Please search for a study first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Search by Trait"):
                st.switch_page("pages/1_Search_by_Trait.py")
        with col2:
            if st.button("Search by Study"):
                st.switch_page("pages/2_Search_by_Study.py")
        return

    # ---- Invalidate cache if study changed ----
    _invalidate_cache_on_study_change(pmid, model)

    st.title("Study Information")

    # ---- Model selector for switching models ----
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Viewing study: **{pmid}**")
    with col2:
        new_model = model_selector(key="study_info_model")
        if new_model != model:
            st.session_state["selected_model"] = new_model
            st.rerun()

    # ---- Fetch extraction data ----
    with st.spinner("Loading study data..."):
        extraction_data = get_extraction(pmid, model)

    if extraction_data is None:
        st.error(
            f"Study {pmid} not found for model {model}. "
            "Please try a different model or search for another study."
        )
        return

    # ---- Panel 1: Study Info (always shown) ----
    _render_study_info(extraction_data)

    # ---- Panel 2: Extraction Results (collapsed by default) ----
    with st.expander("Extraction Results", expanded=False):
        _render_extraction_results(extraction_data)

    # ---- Panel 3: Similar by Trait Profile (lazy load) ----
    with st.expander("Similar Studies by Trait Profile", expanded=False):
        _render_trait_similarity(pmid, model)

    # ---- Panel 4: Similar by Evidence Profile (lazy load) ----
    with st.expander("Similar Studies by Evidence Profile", expanded=False):
        _render_evidence_similarity(pmid, model)


def _invalidate_cache_on_study_change(pmid: str, model: str) -> None:
    """Invalidate cached similarity data when study changes.

    Args:
        pmid: Current PubMed ID
        model: Current extraction model
    """
    current_key = f"{pmid}_{model}"
    previous_key = st.session_state.get("_current_study_key", "")

    if current_key != previous_key:
        # Clear old similarity cache entries
        keys_to_remove = [
            key
            for key in st.session_state.keys()
            if isinstance(key, str)
            and (
                key.startswith("trait_sim_") or key.startswith("evidence_sim_")
            )
        ]
        for key in keys_to_remove:
            del st.session_state[key]

        # Update current study key
        st.session_state["_current_study_key"] = current_key


def _render_study_info(data: dict) -> None:
    """Render the study information panel.

    Args:
        data: Extraction data from database
    """
    st.subheader("Study Details")

    # ---- PMID with PubMed link ----
    pmid = data.get("pmid", "")
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    st.markdown(f"**PMID:** [{pmid}]({pubmed_url})")

    # ---- Title ----
    title = data.get("title", "N/A")
    st.markdown(f"**Title:** {title}")

    # ---- Year and Journal ----
    col1, col2 = st.columns(2)
    with col1:
        pub_date = data.get("pub_date", "")
        year = pub_date[:4] if pub_date else "N/A"
        st.markdown(f"**Year:** {year}")
    with col2:
        journal = data.get("journal", "") or "N/A"
        st.markdown(f"**Journal:** {journal}")

    # ---- Abstract ----
    abstract = data.get("abstract", "")
    if abstract:
        st.markdown("**Abstract:**")
        _render_truncated_abstract(abstract)

    st.divider()


def _render_truncated_abstract(abstract: str) -> None:
    """Render abstract with truncation and expand option.

    Args:
        abstract: Full abstract text
    """
    if len(abstract) <= ABSTRACT_TRUNCATE_LENGTH:
        st.write(abstract)
        return

    # Truncate at word boundary
    truncated = abstract[:ABSTRACT_TRUNCATE_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    truncated += "..."

    # Use session state to track expansion
    expand_key = "abstract_expanded"
    if expand_key not in st.session_state:
        st.session_state[expand_key] = False

    if st.session_state[expand_key]:
        st.write(abstract)
        if st.button("Show less", key="collapse_abstract"):
            st.session_state[expand_key] = False
            st.rerun()
    else:
        st.write(truncated)
        if st.button("Show more", key="expand_abstract"):
            st.session_state[expand_key] = True
            st.rerun()


def _render_extraction_results(data: dict) -> None:
    """Render the extraction results panel.

    Args:
        data: Extraction data from database
    """
    results = data.get("results", [])
    traits = data.get("traits", [])

    if not results:
        st.info("No extraction results available for this study.")
        return

    st.markdown(f"**Extraction Model:** {data.get('model', 'N/A')}")
    st.markdown(f"**Exposure-Outcome Pairs:** {len(results)}")
    st.markdown(f"**Traits Identified:** {len(traits)}")

    st.divider()

    # ---- Display each result ----
    for i, result in enumerate(results, 1):
        exposure = result.get("exposure", "N/A")
        outcome = result.get("outcome", "N/A")

        st.markdown(f"**{i}. {exposure} -> {outcome}**")

        # ---- Effect size ----
        effect_parts = []
        if result.get("beta") is not None:
            effect_parts.append(f"Beta: {result['beta']:.4f}")
        if result.get("odds_ratio") is not None:
            effect_parts.append(f"OR: {result['odds_ratio']:.4f}")
        if result.get("hazard_ratio") is not None:
            effect_parts.append(f"HR: {result['hazard_ratio']:.4f}")

        if effect_parts:
            st.write(", ".join(effect_parts))

        # ---- Confidence interval ----
        ci_lower = result.get("ci_lower")
        ci_upper = result.get("ci_upper")
        if ci_lower is not None and ci_upper is not None:
            st.write(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # ---- P-value ----
        p_value = result.get("p_value")
        if p_value is not None:
            st.write(f"P-value: {p_value:.2e}")

        # ---- Direction ----
        direction = result.get("direction")
        if direction:
            st.write(f"Direction: {direction}")

        st.divider()

    # ---- Show raw data (collapsed) ----
    with st.expander("Raw Results (JSON)", expanded=False):
        st.json(results)

    with st.expander("Raw Metadata (JSON)", expanded=False):
        metadata = data.get("metadata", {})
        st.json(metadata)


def _render_trait_similarity(pmid: str, model: str) -> None:
    """Render the trait similarity panel with lazy loading.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model
    """
    # Use session state to cache loaded data
    cache_key = f"trait_sim_{pmid}_{model}"

    if cache_key not in st.session_state:
        with st.spinner("Loading trait similarity data..."):
            data = get_similar_by_trait(pmid, model, limit=10)
            st.session_state[cache_key] = data

    data = st.session_state[cache_key]

    if data is None:
        st.info("Trait similarity data not available for this study.")
        return

    # ---- Query info ----
    st.markdown(
        f"**Query Study Traits:** {data.get('query_trait_count', 'N/A')}"
    )

    st.divider()

    # ---- Display similar studies ----
    similar_studies = data.get("similar_studies", [])
    selected_pmid = trait_similarity_table(similar_studies)

    if selected_pmid:
        st.session_state["selected_pmid"] = selected_pmid
        st.rerun()

    # ---- Show raw data (collapsed) ----
    with st.expander("Raw Data (JSON)", expanded=False):
        st.json(data)


def _render_evidence_similarity(pmid: str, model: str) -> None:
    """Render the evidence similarity panel with lazy loading.

    Args:
        pmid: PubMed ID of the study
        model: Extraction model
    """
    # ---- Toggle for showing matched pairs ----
    show_matched_pairs = st.checkbox(
        "Show matched evidence pair details",
        value=False,
        help=(
            "When enabled, shows the actual exposure-outcome pairs that "
            "matched between studies. This requires additional computation."
        ),
        key=f"show_matched_pairs_{pmid}_{model}",
    )

    # Use session state to cache loaded data
    # Cache key includes show_matched_pairs to reload when toggled
    cache_key = f"evidence_sim_{pmid}_{model}_{show_matched_pairs}"

    if cache_key not in st.session_state:
        with st.spinner("Loading evidence similarity data..."):
            data = get_similar_by_evidence(
                pmid,
                model,
                limit=10,
                compute_matched_pairs=show_matched_pairs,
            )
            st.session_state[cache_key] = data

    data = st.session_state[cache_key]

    if data is None:
        st.info("Evidence similarity data not available for this study.")
        return

    # ---- Query info ----
    st.markdown(
        f"**Query Study Results:** {data.get('query_result_count', 'N/A')}"
    )

    st.divider()

    # ---- Display similar studies ----
    similar_studies = data.get("similar_studies", [])
    selected_pmid = evidence_similarity_table(
        similar_studies,
        show_matched_pairs=show_matched_pairs,
    )

    if selected_pmid:
        st.session_state["selected_pmid"] = selected_pmid
        st.rerun()

    # ---- Show raw data (collapsed) ----
    with st.expander("Raw Data (JSON)", expanded=False):
        st.json(data)


if __name__ == "__main__":
    main()
