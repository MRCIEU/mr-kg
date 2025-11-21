"""Color schemes for manuscript figures.

This module provides color palettes for use in Altair visualizations,
with support for different color scheme styles.
"""

from typing import Dict, List


# ==== Gruvbox Dark color scheme ====
# Based on the Gruvbox color palette by morhetz
# https://github.com/morhetz/gruvbox

GRUVBOX_DARK = {
    # Background colors
    "bg0_h": "#1d2021",  # Hard contrast background
    "bg0": "#282828",  # Default background
    "bg1": "#3c3836",  # Lighter background
    "bg2": "#504945",  # Even lighter background
    "bg3": "#665c54",  # Lightest background
    # Foreground colors
    "fg0": "#fbf1c7",  # Light foreground
    "fg1": "#ebdbb2",  # Default foreground
    "fg2": "#d5c4a1",  # Darker foreground
    "fg3": "#bdae93",  # Even darker foreground
    "fg4": "#a89984",  # Darkest foreground
    # Primary colors (bright)
    "red": "#fb4934",
    "green": "#b8bb26",
    "yellow": "#fabd2f",
    "blue": "#83a598",
    "purple": "#d3869b",
    "aqua": "#8ec07c",
    "orange": "#fe8019",
    # Primary colors (neutral/faded)
    "red_neutral": "#cc241d",
    "green_neutral": "#98971a",
    "yellow_neutral": "#d79921",
    "blue_neutral": "#458588",
    "purple_neutral": "#b16286",
    "aqua_neutral": "#689d6a",
    "orange_neutral": "#d65d0e",
    # Grays
    "gray": "#928374",
    "gray_dark": "#7c6f64",
}


def get_gruvbox_sequential(n: int = 5) -> List[str]:
    """Get a sequential color palette from Gruvbox colors.

    Args:
        n: Number of colors needed

    Returns:
        List of hex color codes
    """
    if n <= 4:
        return [
            GRUVBOX_DARK["green"],
            GRUVBOX_DARK["yellow"],
            GRUVBOX_DARK["orange"],
            GRUVBOX_DARK["red"],
        ][:n]
    else:
        return [
            GRUVBOX_DARK["green"],
            GRUVBOX_DARK["aqua"],
            GRUVBOX_DARK["yellow"],
            GRUVBOX_DARK["orange"],
            GRUVBOX_DARK["red"],
        ][:n]


def get_gruvbox_categorical(n: int = 8) -> List[str]:
    """Get a categorical color palette from Gruvbox colors.

    Args:
        n: Number of colors needed

    Returns:
        List of hex color codes
    """
    colors = [
        GRUVBOX_DARK["blue"],
        GRUVBOX_DARK["orange"],
        GRUVBOX_DARK["green"],
        GRUVBOX_DARK["purple"],
        GRUVBOX_DARK["yellow"],
        GRUVBOX_DARK["aqua"],
        GRUVBOX_DARK["red"],
        GRUVBOX_DARK["gray"],
    ]
    return colors[:n]


# ==== Reproducibility tier colors (Gruvbox) ====

REPRODUCIBILITY_COLORS_GRUVBOX = {
    "high": GRUVBOX_DARK["green"],
    "moderate": GRUVBOX_DARK["yellow"],
    "low": GRUVBOX_DARK["orange"],
    "discordant": GRUVBOX_DARK["gray"],
}

REPRODUCIBILITY_COLORS_GRUVBOX_TITLE = {
    "High": GRUVBOX_DARK["green"],
    "Moderate": GRUVBOX_DARK["yellow"],
    "Low": GRUVBOX_DARK["orange"],
    "Discordant": GRUVBOX_DARK["gray"],
}


# ==== Match type colors (Gruvbox) ====

MATCH_TYPE_COLORS_GRUVBOX = {
    "exact": GRUVBOX_DARK["green_neutral"],
    "fuzzy": GRUVBOX_DARK["orange"],
}


# ==== Field type colors (Gruvbox) ====

FIELD_TYPE_COLORS_GRUVBOX = {
    "Confidence Interval": GRUVBOX_DARK["orange"],
    "Direction": GRUVBOX_DARK["green"],
    "Effect Size": GRUVBOX_DARK["red"],
    "Statistical": GRUVBOX_DARK["purple"],
    "P-value": GRUVBOX_DARK["purple"],
    "95% CI": GRUVBOX_DARK["blue"],
    "SE": GRUVBOX_DARK["yellow_neutral"],
}


# ==== Helper functions ====


def get_tier_colors(use_gruvbox: bool = True) -> Dict[str, str]:
    """Get reproducibility tier color mapping.

    Args:
        use_gruvbox: If True, use Gruvbox colors; otherwise use defaults

    Returns:
        Dictionary mapping tier names to hex colors
    """
    if use_gruvbox:
        return REPRODUCIBILITY_COLORS_GRUVBOX_TITLE
    else:
        return {
            "High": "#2ecc71",
            "Moderate": "#f39c12",
            "Low": "#e74c3c",
            "Discordant": "#95a5a6",
        }


def get_match_type_colors(use_gruvbox: bool = True) -> Dict[str, str]:
    """Get match type color mapping.

    Args:
        use_gruvbox: If True, use Gruvbox colors; otherwise use defaults

    Returns:
        Dictionary mapping match types to hex colors
    """
    if use_gruvbox:
        return MATCH_TYPE_COLORS_GRUVBOX
    else:
        return {
            "exact": "#2E7D32",
            "fuzzy": "#F57C00",
        }


def get_field_type_colors(use_gruvbox: bool = True) -> Dict[str, str]:
    """Get field type color mapping.

    Args:
        use_gruvbox: If True, use Gruvbox colors; otherwise use defaults

    Returns:
        Dictionary mapping field types to hex colors
    """
    if use_gruvbox:
        return FIELD_TYPE_COLORS_GRUVBOX
    else:
        # Use default Altair category10 colors
        return {}
