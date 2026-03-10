"""
Equation-agnostic calibrator for structural models.

Given observed values (default probabilities, CDS spreads, etc.) at multiple
maturities, finds the model parameters θ that minimise the fitting error:

    θ* = argmin_θ  Σᵢ (model(θ)ᵢ − targetᵢ)²

The calibrator is model-independent: it accepts any callable that maps
a parameter vector to model-implied values.

Pipeline:
    params θ  →  model_fn(θ)  →  model values  →  compare to targets
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class Calibrator:
    """
    Calibrate model parameters to observed target values via L-BFGS-B.

    Args:
        model_fn:    Callable mapping params (np.ndarray) → model values (np.ndarray).
        param_names: Names of the parameters being calibrated (for output).

    Example:
        >>> model_fn = lambda p: merton_default_probability(p[0], p[1], K=80, r=0.05, T=T)
        >>> cal = Calibrator(model_fn=model_fn, param_names=['V0', 'sigma'])
        >>> result = cal.calibrate(target_values=observed_PD, x0=[90, 0.3])
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        param_names: List[str],
    ):
        self.model_fn = model_fn
        self.param_names = param_names

    def _objective(self, params: np.ndarray, target_values: np.ndarray) -> float:
        """Sum-of-squared errors."""
        model_values = self.model_fn(params)
        errors = model_values - target_values
        return np.sum(errors ** 2)

    def calibrate(
        self,
        target_values: np.ndarray,
        x0: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        options: Optional[Dict] = None,
    ) -> Dict:
        """
        Calibrate model parameters to observed target values.

        Args:
            target_values: Observed values to fit against [n_points]
            x0:            Initial parameter guess [n_params]
            bounds:        Parameter bounds [(low, high), ...] for each parameter
            options:       Additional options passed to scipy.optimize.minimize

        Returns:
            Dictionary with params, param_dict, model_values, target_values,
            errors, objective, success, and scipy_result.
        """
        target_values = np.atleast_1d(np.asarray(target_values, dtype=np.float64))
        x0 = np.asarray(x0, dtype=np.float64)

        # --- Run optimiser ---
        result = minimize(
            self._objective,
            x0=x0,
            args=(target_values,),
            method='L-BFGS-B',
            bounds=bounds,
            options=options or {'maxiter': 1000, 'ftol': 1e-12},
        )

        # --- Extract results ---
        calibrated_params = result.x
        param_dict = dict(zip(self.param_names, calibrated_params))
        model_values = self.model_fn(calibrated_params)
        errors = model_values - target_values

        # --- Logging ---
        logger.info("Calibration %s", "converged" if result.success else "FAILED")
        for name, val in param_dict.items():
            logger.info("  %s = %.6f", name, val)
        logger.info("  RMSE: %.6f", np.sqrt(np.mean(errors ** 2)))

        return {
            'params': calibrated_params,
            'param_dict': param_dict,
            'model_values': model_values,
            'target_values': target_values,
            'errors': errors,
            'objective': result.fun,
            'success': result.success,
            'scipy_result': result,
        }
