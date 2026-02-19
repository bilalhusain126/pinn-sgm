"""
PINN solvers for PDEs.

This module provides Physics-Informed Neural Network solvers for
various partial differential equations.
"""

from .pinn_solver import PINNSolver
from .score_pinn_solver import ScorePINNSolver

__all__ = ['PINNSolver', 'ScorePINNSolver']
