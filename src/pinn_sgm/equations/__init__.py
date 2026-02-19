"""
PDE and SDE equations for PINN solvers.

This module provides abstract base classes and concrete implementations
of partial differential equations and stochastic differential equations,
particularly Fokker-Planck equations for financial applications.
"""

from .base import BasePDE, BaseSDE
from .fokker_planck import FokkerPlanckMertonND

__all__ = [
    'BasePDE',
    'BaseSDE',
    'FokkerPlanckMertonND',
]
