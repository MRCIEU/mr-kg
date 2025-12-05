"""Theme configuration and toggle component for the webapp.

Provides Gruvbox light and dark theme support with a sidebar toggle.
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

GRUVBOX_LIGHT = ThemeColors(
    background="#fbf1c7",
    secondary_background="#ebdbb2",
    text="#3c3836",
    primary="#458588",
    accent_red="#cc241d",
    accent_green="#98971a",
    accent_yellow="#d79921",
    accent_blue="#458588",
)


def get_current_theme() -> ThemeColors:
    """Get the current theme colors based on session state."""
    if st.session_state.get("theme_mode", "dark") == "light":
        return GRUVBOX_LIGHT
    return GRUVBOX_DARK


def _generate_theme_css(theme: ThemeColors) -> str:
    """Generate CSS for the given theme."""
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
    """Apply the current theme CSS to the page.

    This should be called at the start of every page after set_page_config.
    """
    theme = get_current_theme()
    st.markdown(_generate_theme_css(theme), unsafe_allow_html=True)


def theme_toggle() -> None:
    """Render a theme toggle in the sidebar.

    This should be called in the sidebar of every page.
    """
    # Initialize theme in session state if not present
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"

    with st.sidebar:
        st.markdown("---")
        current_mode = st.session_state.theme_mode
        toggle_label = "Light mode" if current_mode == "dark" else "Dark mode"

        if st.button(
            toggle_label,
            key="theme_toggle_btn",
            help="Toggle between light and dark theme",
            use_container_width=True,
        ):
            st.session_state.theme_mode = (
                "light" if current_mode == "dark" else "dark"
            )
            st.rerun()

        # Show current mode indicator
        mode_icon = "Dark" if current_mode == "dark" else "Light"
        st.caption(f"Current: {mode_icon} mode")
