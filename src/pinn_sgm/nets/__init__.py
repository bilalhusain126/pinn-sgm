"""
Neural network architectures for PINN solvers.

This module provides neural network architectures used to approximate
PDE solutions in the Physics-Informed Neural Network framework.
"""

from .mlp import MLP, DensityMLP

__all__ = ['MLP', 'DensityMLP']
