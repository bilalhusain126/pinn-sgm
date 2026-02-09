"""
Neural network architectures for PINN solvers and score-based models.

This module provides neural network architectures used to approximate
PDE solutions in the Physics-Informed Neural Network framework and
score functions for diffusion models.
"""

from .mlp import MLP, DensityMLP, ScoreNetwork

__all__ = ['MLP', 'DensityMLP', 'ScoreNetwork']
