"""Similarity display components for trait and evidence similarity results."""

import streamlit as st


def trait_similarity_table(similar_studies: list[dict]) -> str | None:
    """Display trait similarity results in a table format.

    Args:
        similar_studies: List of similar study dicts from API

    Returns:
        Selected PMID if a study is clicked, None otherwise
    """
    if not similar_studies:
        st.info("No similar studies found by trait profile.")
        return None

    selected_pmid = None

    # ---- Header row ----
    cols = st.columns([2, 4, 2, 2, 1])
    with cols[0]:
        st.markdown("**PMID**")
    with cols[1]:
        st.markdown("**Title**")
    with cols[2]:
        st.markdown("**Semantic**")
    with cols[3]:
        st.markdown("**Jaccard**")
    with cols[4]:
        st.markdown("**Traits**")

    st.divider()

    # ---- Data rows ----
    for i, study in enumerate(similar_studies):
        pmid = study.get("pmid", "")
        title = study.get("title", "")
        semantic_sim = study.get("trait_profile_similarity", 0)
        jaccard_sim = study.get("trait_jaccard_similarity", 0)
        trait_count = study.get("trait_count", 0)

        cols = st.columns([2, 4, 2, 2, 1])
        with cols[0]:
            if st.button(pmid, key=f"trait_sim_{i}_{pmid}"):
                selected_pmid = pmid
        with cols[1]:
            st.write(_truncate_text(title, 50))
        with cols[2]:
            st.write(f"{semantic_sim:.2%}")
        with cols[3]:
            st.write(f"{jaccard_sim:.2%}")
        with cols[4]:
            st.write(str(trait_count))

    return selected_pmid


def evidence_similarity_table(
    similar_studies: list[dict],
    show_matched_pairs: bool = False,
) -> str | None:
    """Display evidence similarity results in a table format.

    Args:
        similar_studies: List of similar study dicts from API
        show_matched_pairs: If True, show matched evidence pairs in expandable
            sections for each study (requires matched_evidence_pairs to be
            populated in the data)

    Returns:
        Selected PMID if a study is clicked, None otherwise
    """
    if not similar_studies:
        st.info("No similar studies found by evidence profile.")
        return None

    selected_pmid = None

    # ---- Header row ----
    cols = st.columns([2, 3, 2, 1, 2])
    with cols[0]:
        st.markdown("**PMID**")
    with cols[1]:
        st.markdown("**Title**")
    with cols[2]:
        st.markdown("**Concordance**")
    with cols[3]:
        st.markdown("**Pairs**")
    with cols[4]:
        st.markdown("**Match Type**")

    st.divider()

    # ---- Data rows ----
    for i, study in enumerate(similar_studies):
        pmid = study.get("pmid", "")
        title = study.get("title", "")
        concordance = study.get("direction_concordance", 0)
        matched_pairs = study.get("matched_pairs", 0)
        match_type_exact = study.get("match_type_exact", False)
        match_type_fuzzy = study.get("match_type_fuzzy", False)
        match_type_efo = study.get("match_type_efo", False)

        cols = st.columns([2, 3, 2, 1, 2])
        with cols[0]:
            if st.button(pmid, key=f"evidence_sim_{i}_{pmid}"):
                selected_pmid = pmid
        with cols[1]:
            st.write(_truncate_text(title, 40))
        with cols[2]:
            # Color code concordance: green for positive, red for negative
            color = _concordance_color(concordance)
            st.markdown(f":{color}[{concordance:+.2f}]")
        with cols[3]:
            st.write(str(matched_pairs))
        with cols[4]:
            match_type_str = _format_match_type(
                match_type_exact, match_type_fuzzy, match_type_efo
            )
            st.write(match_type_str)

        # ---- Show matched evidence pairs if available and requested ----
        if show_matched_pairs:
            matched_evidence_pairs = study.get("matched_evidence_pairs")
            if matched_evidence_pairs is not None:
                _render_matched_evidence_pairs(matched_evidence_pairs, pmid, i)

    return selected_pmid


def _render_matched_evidence_pairs(
    pairs: list[dict], pmid: str, study_index: int
) -> None:
    """Render matched evidence pairs in an expander.

    Args:
        pairs: List of matched evidence pair dicts
        pmid: PMID of the similar study (for unique key)
        study_index: Index of the study in the list (for unique key)
    """
    if not pairs:
        st.caption("No matched pairs found")
        return

    with st.expander(f"Matched pairs ({len(pairs)})", expanded=False):
        for j, pair in enumerate(pairs):
            query_exp = pair.get("query_exposure", "")
            query_out = pair.get("query_outcome", "")
            query_dir = pair.get("query_direction", "")
            similar_exp = pair.get("similar_exposure", "")
            similar_out = pair.get("similar_outcome", "")
            similar_dir = pair.get("similar_direction", "")
            match_type = pair.get("match_type", "")

            # ---- Format the match display ----
            st.markdown(f"**Match {j + 1}** ({match_type})")

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Query study:")
                st.write(f"{query_exp} -> {query_out}")
                if query_dir:
                    st.caption(f"Direction: {query_dir}")
            with col2:
                st.caption("Similar study:")
                st.write(f"{similar_exp} -> {similar_out}")
                if similar_dir:
                    st.caption(f"Direction: {similar_dir}")

            if j < len(pairs) - 1:
                st.divider()


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


def _concordance_color(value: float) -> str:
    """Get color name for concordance value.

    Args:
        value: Direction concordance value (-1 to +1)

    Returns:
        Color name for st.markdown
    """
    if value >= 0.5:
        return "green"
    elif value >= 0:
        return "orange"
    else:
        return "red"


def _format_match_type(exact: bool, fuzzy: bool, efo: bool) -> str:
    """Format match type flags into a readable string.

    Args:
        exact: Whether exact matching was used
        fuzzy: Whether fuzzy matching was used
        efo: Whether EFO ontology matching was used

    Returns:
        Formatted string indicating match types
    """
    types = []
    if exact:
        types.append("Exact")
    if fuzzy:
        types.append("Fuzzy")
    if efo:
        types.append("EFO")

    if not types:
        return "N/A"
    return ", ".join(types)
