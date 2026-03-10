"""
Calibration tools for structural credit risk models.

  - calibrator: equation-agnostic optimizer fitting model params to observed values
"""

from .calibrator import Calibrator

__all__ = ['Calibrator']
