"""
Neural network architectures for PINN solvers and score-based models.

This module provides neural network architectures used to approximate
PDE solutions in the Physics-Informed Neural Network framework and
score functions for diffusion models.

Base Architectures:
  - MLP: Standard multi-layer perceptron
  - DGM: Deep Galerkin Method with LSTM-like gating

Application-Specific Networks:
  - DensityNetwork: Wrapper for probability density approximation (ensures positivity)
  - ScoreNetwork: For score function approximation in generative models
"""

from .mlp import MLP
from .dgm import DGM
from .density_net import DensityNetwork
from .score_net import ScoreNetwork

__all__ = ['MLP', 'DGM', 'DensityNetwork', 'ScoreNetwork']
