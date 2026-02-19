"""
Utility functions for PINN-SGM framework.

This module provides utilities for score extraction and
integration with diffusion models.

Note: Visualization functions have been moved to pinn_sgm.visualizations package.
"""

from .score_extraction import ScoreExtractor

__all__ = [
    'ScoreExtractor',
]
