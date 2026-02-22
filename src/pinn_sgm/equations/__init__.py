"""
PDE and SDE equations for PINN solvers.

This module provides abstract base classes and concrete implementations
of partial differential equations and stochastic differential equations.

  - BasePDE / BaseSDE:         abstract interfaces (equations/base.py)
  - FokkerPlanckMertonND:      Merton model FPE + SDE (equations/merton.py)
"""

from .base import BasePDE, BaseSDE
from .merton import FokkerPlanckMertonND

__all__ = [
    'BasePDE',
    'BaseSDE',
    'FokkerPlanckMertonND',
]
