"""Database connection and query functions for the MR-KG web app."""

import duckdb
import pandas as pd
import streamlit as st
from typing import List, Optional

from funcs.utils import get_database_paths


@st.cache_resource
def setup_database_paths(profile: str = "local") -> None:
    """Set up database paths based on deployment profile."""
    vector_store_db_path, trait_profile_db_path = get_database_paths(profile)
    if "vector_store_db" not in st.session_state:
        st.session_state.vector_store_db = vector_store_db_path
    if "trait_profile_db" not in st.session_state:
        st.session_state.trait_profile_db = trait_profile_db_path


@st.cache_resource
def get_database_connections():
    """Get database connections with caching."""
    vector_store_db = st.session_state.vector_store_db
    trait_profile_db = st.session_state.trait_profile_db

    vector_conn = duckdb.connect(str(vector_store_db), read_only=True)
    trait_conn = duckdb.connect(str(trait_profile_db), read_only=True)
    return vector_conn, trait_conn


@st.cache_data
def get_available_traits() -> List[str]:
    """Get list of available traits for filtering."""
    vector_conn, _ = get_database_connections()
    query = """
    SELECT DISTINCT trait_label
    FROM model_result_traits
    ORDER BY trait_label
    """
    result = vector_conn.execute(query).fetchall()
    return [row[0] for row in result]


@st.cache_data
def get_pmid_model_data(
    trait_filters: Optional[List[str]] = None, limit: int = 100
) -> pd.DataFrame:
    """Get data from pmid_model_analysis view with optional trait filtering."""
    vector_conn, _ = get_database_connections()

    base_query = """
    SELECT
        pmid,
        model,
        model_result_id,
        title,
        abstract,
        pub_date,
        journal,
        journal_issn,
        author_affil,
        traits
    FROM pmid_model_analysis
    """

    if trait_filters:
        # Create a subquery to filter by traits first
        trait_filter_conditions = []
        for trait in trait_filters:
            escaped_trait = trait.replace("'", "''")
            trait_filter_conditions.append(f"trait_label = '{escaped_trait}'")

        trait_filter_query = f"""
        SELECT DISTINCT model_result_id
        FROM model_result_traits
        WHERE {" OR ".join(trait_filter_conditions)}
        """

        query = f"""
        {base_query}
        WHERE model_result_id IN ({trait_filter_query})
        ORDER BY pmid, model LIMIT {limit}
        """
    else:
        query = f"{base_query} ORDER BY pmid, model LIMIT {limit}"

    result = vector_conn.execute(query).fetchdf()
    return result


@st.cache_data
def get_trait_similarity_data(
    limit: int = 100, min_similarity: float = 0.0
) -> pd.DataFrame:
    """Get data from trait_similarity_analysis view."""
    _, trait_conn = get_database_connections()

    query = f"""
    SELECT
        query_pmid,
        query_model,
        query_title,
        query_trait_count,
        similar_pmid,
        similar_model,
        similar_title,
        similar_trait_count,
        trait_profile_similarity,
        trait_jaccard_similarity,
        similarity_rank
    FROM trait_similarity_analysis
    WHERE trait_profile_similarity >= {min_similarity}
    ORDER BY trait_profile_similarity DESC
    LIMIT {limit}
    """

    result = trait_conn.execute(query).fetchdf()
    return result
