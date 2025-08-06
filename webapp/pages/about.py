"""About page for the MR-KG web app."""

import streamlit as st


def about_page():
    """Show the about page."""
    st.header("About MR-KG Literature Explorer")

    st.markdown("""
    ## Overview

    The MR-KG Literature Explorer is a web application for demonstrating structural
    PubMed literature data extracted by large language models. This tool enables
    researchers to explore trait relationships and discover similar studies based
    on their trait profiles.
    """)


about_page()
