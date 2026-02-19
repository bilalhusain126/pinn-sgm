"""
Visualization utilities for PINN solutions and training diagnostics.

This package provides plotting functions organized by purpose:
  - plot_training: Training diagnostics (loss curves, gradient norms)
  - plot_evaluation: Predictions vs analytical solutions
  - plot_analysis: Deep-dive analysis (errors, score magnitudes)

To apply publication-quality styling, call:
    from pinn_sgm.visualizations import setup_publication_style
    setup_publication_style()
"""

from ._style import setup_publication_style
from .plot_training import plot_training_history
from .plot_evaluation import plot_density_evolution, plot_score_field
from .plot_analysis import plot_error_analysis, plot_score_magnitude_analysis

__all__ = [
    'setup_publication_style',
    'plot_training_history',
    'plot_density_evolution',
    'plot_score_field',
    'plot_error_analysis',
    'plot_score_magnitude_analysis',
]
