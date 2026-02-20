"""
Neural network architectures for PINN solvers.

Base Architectures:
  - MLP: Standard multi-layer perceptron
  - DGM: Deep Galerkin Method with LSTM-like gating

Application-Specific Networks:
  - DensityNetwork: Wrapper for probability density approximation (ensures positivity)
"""

from .mlp import MLP
from .dgm import DGM
from .density_net import DensityNetwork

__all__ = ['MLP', 'DGM', 'DensityNetwork']
