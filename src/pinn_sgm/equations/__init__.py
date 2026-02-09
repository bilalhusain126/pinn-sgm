"""
PDE equations for PINN solvers.

This module provides abstract base classes and concrete implementations
of partial differential equations, particularly the Fokker-Planck equation
for financial applications.
"""

from .base import BasePDE
from .fokker_planck import FokkerPlanckMerton

__all__ = [
    'BasePDE',
    'FokkerPlanckMerton',
]
