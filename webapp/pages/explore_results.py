"""Explore Results page for the MR-KG web app.

This page shows detailed analysis of a selected study result and finds studies with similar trait profiles.
Features a two-column layout with study details on the left and similar studies on the right.
"""

import streamlit as st
import pandas as pd

from resources.database import get_database_connections


@st.cache_data
def get_study_results(pmid: str, model: str) -> pd.DataFrame:
    """Get all traits extracted for a specific study and model along with metadata and results."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT
        mrt.trait_label,
        mrt.trait_index,
        mrt.trait_id_in_result,
        mr.metadata,
        mr.results
    FROM model_results mr
    JOIN model_result_traits mrt ON mr.id = mrt.model_result_id
    WHERE mr.pmid = ? AND mr.model = ?
    ORDER BY mrt.trait_label
    """
    result = vector_conn.execute(query, [pmid, model]).fetchall()
    return pd.DataFrame(
        result,
        columns=[
            "trait_label",
            "trait_index",
            "trait_id_in_result",
            "metadata",
            "results",
        ],
    )


@st.cache_data
def get_similar_studies(
    pmid: str, model: str, limit: int = 10
) -> pd.DataFrame:
    """Get studies with similar trait profiles using the top_similarity_pairs view."""
    _, trait_profile_conn = get_database_connections()
    query = """
    SELECT
        tsp.similar_pmid,
        tsp.similar_title,
        tsp.trait_profile_similarity,
        tsp.trait_jaccard_similarity,
        tsp.query_trait_count,
        tsp.similar_trait_count,
        tsp.model
    FROM top_similarity_pairs tsp
    WHERE tsp.query_pmid = ?
    AND tsp.model = ?
    ORDER BY tsp.trait_profile_similarity DESC
    LIMIT ?
    """
    result = trait_profile_conn.execute(query, [pmid, model, limit]).fetchall()
    return pd.DataFrame(
        result,
        columns=[
            "similar_pmid",
            "similar_title",
            "trait_profile_similarity",
            "trait_jaccard_similarity",
            "query_trait_count",
            "similar_trait_count",
            "model",
        ],
    )


@st.cache_data
def get_study_details(pmid: str) -> pd.DataFrame:
    """Get detailed information about a study from the pubmed data."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT
        pmid,
        title,
        abstract,
        pub_date,
        journal,
        journal_issn,
        author_affil
    FROM mr_pubmed_data
    WHERE pmid = ?
    """
    result = vector_conn.execute(query, [pmid]).fetchall()
    return pd.DataFrame(
        result,
        columns=[
            "pmid",
            "title",
            "abstract",
            "pub_date",
            "journal",
            "journal_issn",
            "author_affil",
        ],
    )


def render_study_details() -> None:
    """Render the detailed view of the selected study."""
    st.header("Study Details")

    # Display basic information
    st.success(f"**PMID:** {st.session_state.explore_pmid}")
    st.markdown(f"**Model:** {st.session_state.explore_model}")

    if st.session_state.get("explore_title"):
        st.markdown(f"**Title:** {st.session_state.explore_title}")

    if st.session_state.get("explore_journal"):
        st.markdown(f"**Journal:** {st.session_state.explore_journal}")

    if st.session_state.get("explore_pub_date"):
        st.markdown(
            f"**Publication Date:** {st.session_state.explore_pub_date}"
        )

    # Get additional study details
    try:
        study_details = get_study_details(st.session_state.explore_pmid)
        if not study_details.empty:
            row = study_details.iloc[0]

            if pd.notna(row["abstract"]) and row["abstract"]:
                with st.expander("Abstract", expanded=False):
                    st.markdown(row["abstract"])

            if pd.notna(row["author_affil"]) and row["author_affil"]:
                with st.expander("Author Affiliations", expanded=False):
                    st.markdown(row["author_affil"])
    except Exception as e:
        st.warning(f"Could not load additional study details: {str(e)}")

    # PubMed link
    pubmed_url = (
        f"https://pubmed.ncbi.nlm.nih.gov/{st.session_state.explore_pmid}/"
    )
    st.markdown(f"🔗 [View on PubMed]({pubmed_url})")

    # Display extracted traits
    st.markdown("---")
    st.subheader("Extracted Traits")

    try:
        results_df = get_study_results(
            st.session_state.explore_pmid, st.session_state.explore_model
        )

        if results_df.empty:
            st.warning("No traits found for this study.")
        else:
            st.info(f"Found {len(results_df)} extracted traits")

            # Display traits in a nice format
            for idx, (_, trait_row) in enumerate(results_df.iterrows()):
                st.markdown(f"• **{trait_row['trait_label']}**")

            # Display extracted data section
            st.markdown("---")
            st.subheader("Extracted Data")

            # Get the first row since metadata and results should be the same for all traits from the same study
            if not results_df.empty:
                first_row = results_df.iloc[0]

                # Display metadata
                if pd.notna(first_row["metadata"]) and first_row["metadata"]:
                    with st.expander("Model Metadata", expanded=False):
                        st.json(first_row["metadata"])
                else:
                    st.info("No metadata available")

                # Display results
                if pd.notna(first_row["results"]) and first_row["results"]:
                    with st.expander("Model Results", expanded=False):
                        st.json(first_row["results"])
                else:
                    st.info("No results data available")

    except Exception as e:
        st.error(f"Error loading traits: {str(e)}")

    # Back to traits page button
    st.markdown("---")
    if st.button("← Back to Trait Explorer", type="secondary"):
        st.switch_page("pages/explore_traits.py")


def render_similar_studies() -> None:
    """Render the list of studies with similar trait profiles."""
    st.header("Similar Studies")

    st.markdown(
        f"Studies with similar trait profiles to **{st.session_state.explore_pmid}**"
    )

    try:
        similar_df = get_similar_studies(
            st.session_state.explore_pmid, st.session_state.explore_model
        )

        if similar_df.empty:
            st.warning("No similar studies found.")
            return

        st.info(f"Found {len(similar_df)} similar studies")

        # Display similar studies
        for idx, (_, row) in enumerate(similar_df.iterrows()):
            with st.container():
                # Similarity metrics
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{idx + 1}. PMID: {row['similar_pmid']}**")
                    if pd.notna(row["similar_title"]):
                        title_display = (
                            row["similar_title"][:100] + "..."
                            if len(str(row["similar_title"])) > 100
                            else row["similar_title"]
                        )
                        st.markdown(f"_{title_display}_")

                    # Similarity metrics
                    st.markdown(
                        f"🎯 **Trait Profile Similarity:** {row['trait_profile_similarity']:.3f}"
                    )
                    st.markdown(
                        f"🔗 **Jaccard Similarity:** {row['trait_jaccard_similarity']:.3f}"
                    )
                    st.markdown(
                        f"📊 **Trait Counts:** Query: {row['query_trait_count']}, Similar: {row['similar_trait_count']}"
                    )

                with col2:
                    if st.button(
                        "Explore Results",
                        key=f"similar_{idx}_{row['similar_pmid']}",
                        use_container_width=True,
                        type="primary",
                    ):
                        # Update session state to explore this similar study
                        st.session_state.explore_pmid = row["similar_pmid"]
                        st.session_state.explore_model = row["model"]
                        st.session_state.explore_title = row["similar_title"]
                        # Clear other fields since we don't have them from similarity data
                        st.session_state.explore_journal = None
                        st.session_state.explore_pub_date = None
                        st.session_state.explore_metadata = None
                        st.rerun()

                st.markdown("---")

    except Exception as e:
        st.error(f"Error loading similar studies: {str(e)}")


def main():
    """Main function for the explore results page."""
    st.set_page_config(page_title="Explore Results - MR-KG", layout="wide")

    st.title("Explore Results")
    st.markdown(
        "Detailed analysis of study results and similar trait profiles."
    )

    # Check if we have the required session state
    if not st.session_state.get("explore_pmid") or not st.session_state.get(
        "explore_model"
    ):
        st.error(
            "No study selected. Please go back to the trait explorer and select a study."
        )
        if st.button("← Back to Trait Explorer"):
            st.switch_page("pages/explore_traits.py")
        return

    # Create two columns layout
    left_col, right_col = st.columns([1, 1])

    # Render left column (study details)
    with left_col:
        render_study_details()

    # Render right column (similar studies)
    with right_col:
        render_similar_studies()


if __name__ == "__main__":
    main()
