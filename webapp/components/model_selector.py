"""Model selector component for choosing extraction model."""

import streamlit as st

# Available extraction models
AVAILABLE_MODELS = [
    "deepseek-r1-distilled",
    "gpt-4-1",
    "gpt-4o",
    "gpt-5",
    "llama3",
    "llama3-2",
    "o4-mini",
]

DEFAULT_MODEL = "gpt-5"


def model_selector(key: str = "model") -> str:
    """Dropdown for selecting extraction model.

    Args:
        key: Unique key for the selectbox widget

    Returns:
        Selected model name
    """
    default_index = AVAILABLE_MODELS.index(DEFAULT_MODEL)

    selected = st.selectbox(
        "Model",
        options=AVAILABLE_MODELS,
        index=default_index,
        key=key,
        help="Select the LLM extraction model to use",
    )

    return selected
