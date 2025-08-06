"""About page for the MR-KG web app."""

import streamlit as st


def show_about():
    """Show the about page."""
    st.header("About MR-KG Literature Explorer")

    st.markdown("""
    ## Overview

    The MR-KG Literature Explorer is a web application for demonstrating structural
    PubMed literature data extracted by large language models. This tool enables
    researchers to explore trait relationships and discover similar studies based
    on their trait profiles.

    ## Features

    ### Model Analysis
    - Browse extracted data from PubMed literature analysis
    - Filter results by specific traits of interest
    - View detailed information about each study including:
      - Publication metadata (title, journal, publication date)
      - Abstract content
      - Extracted traits from multiple LLM models
      - Author affiliations

    ### Trait Profile Similarities
    - Explore studies with similar trait profiles
    - Compare trait profile and Jaccard similarities
    - Analyze patterns across different LLM models
    - Filter by similarity thresholds

    ## Data Sources

    This application uses two main DuckDB databases:

    - **vector_store.db**: Contains the main extracted data including:
      - Model results from multiple LLMs
      - Trait embeddings and relationships
      - PubMed metadata
      - EFO (Experimental Factor Ontology) mappings

    - **trait_profile_db.db**: Contains trait profile similarity data including:
      - Pairwise similarity calculations between studies
      - Query combinations and rankings
      - Model-specific similarity statistics

    ## Technology Stack

    - **Frontend**: Streamlit
    - **Database**: DuckDB
    - **Data Processing**: Pandas
    - **Deployment**: UV package manager

    ## Usage Tips

    1. **Model Analysis**: Start by exploring the model analysis view to understand
       the scope of available data. Use trait filters to narrow down to specific
       areas of interest.

    2. **Trait Similarities**: Use the similarity view to discover related studies.
       Adjust the similarity threshold to control the sensitivity of matches.

    3. **Performance**: The application uses caching to improve performance.
       Large queries may take a moment to load initially.

    ## Contact

    For questions about this application or the underlying research, please refer
    to the project documentation or contact the development team.
    """)
