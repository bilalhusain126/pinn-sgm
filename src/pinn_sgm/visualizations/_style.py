"""
Publication-quality plot styling configuration.

Sets up matplotlib to produce LaTeX-compatible, clean, professional plots
with Computer Modern fonts and no grids.
"""

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# Suppress matplotlib mathtext font substitution warnings
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)


def setup_publication_style():
    """
    Configure matplotlib for LaTeX-compatible, publication-ready plots.

    Features:
    - STIX fonts (LaTeX-like, widely available)
    - Matching text and math fonts
    - No grids (clean look)
    - Removes top and right spines (modern aesthetic)
    - Reasonable font sizes (not too large)
    - Thinner line widths for axes
    """
    plt.rcParams.update({
        # --- Font configuration: STIX (LaTeX-like, widely available) ---
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'stix',

        # --- Font sizes ---
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 13,

        # --- Axes styling: clean and minimal ---
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # --- Figure defaults ---
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    logger.debug("Publication style configured successfully")
