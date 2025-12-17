"""Theme configuration for the webapp.

Provides Gruvbox dark theme support.
"""

from dataclasses import dataclass

import streamlit as st


@dataclass
class ThemeColors:
    """Color scheme for a theme."""

    background: str
    secondary_background: str
    text: str
    primary: str
    # Additional Gruvbox colors for UI elements
    accent_red: str
    accent_green: str
    accent_yellow: str
    accent_blue: str


# ---- Gruvbox color definitions ----
# Reference: docs/misc/gruvbox-dark.toml

GRUVBOX_DARK = ThemeColors(
    background="#282828",
    secondary_background="#3c3836",
    text="#ebdbb2",
    primary="#83a598",
    accent_red="#fb4934",
    accent_green="#b8bb26",
    accent_yellow="#fabd2f",
    accent_blue="#83a598",
)


def _generate_theme_css() -> str:
    """Generate CSS for the Gruvbox dark theme."""
    theme = GRUVBOX_DARK
    return f"""
    <style>
        /* Main app background */
        .stApp {{
            background-color: {theme.background};
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {theme.secondary_background};
        }}

        /* Text colors */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
            color: {theme.text};
        }}

        /* Headers */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {theme.text};
        }}

        /* Markdown text */
        .stMarkdown {{
            color: {theme.text};
        }}

        /* Metric labels and values */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {{
            color: {theme.text};
        }}

        /* Input fields */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div {{
            background-color: {theme.secondary_background};
            color: {theme.text};
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {theme.primary};
            color: {theme.background};
            border: none;
        }}

        .stButton > button:hover {{
            background-color: {theme.accent_blue};
            color: {theme.background};
        }}

        /* Dataframe/Table styling */
        .stDataFrame {{
            background-color: {theme.secondary_background};
        }}

        /* Info/Warning/Error boxes */
        .stAlert {{
            background-color: {theme.secondary_background};
            color: {theme.text};
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {theme.secondary_background};
            color: {theme.text};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme.background};
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {theme.text};
        }}

        /* Checkbox */
        .stCheckbox label span {{
            color: {theme.text};
        }}

        /* Divider */
        hr {{
            border-color: {theme.secondary_background};
        }}

        /* Links */
        a {{
            color: {theme.primary};
        }}

        a:hover {{
            color: {theme.accent_blue};
        }}

        /* Code blocks */
        code {{
            background-color: {theme.secondary_background};
            color: {theme.accent_green};
        }}
    </style>
    """


def apply_theme() -> None:
    """Apply the Gruvbox dark theme CSS to the page.

    This should be called at the start of every page after set_page_config.
    """
    st.markdown(_generate_theme_css(), unsafe_allow_html=True)



