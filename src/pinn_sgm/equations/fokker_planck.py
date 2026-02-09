"""
Fokker-Planck equation implementations for financial models.

This module implements the Fokker-Planck equation (FPE) governing the evolution
of probability density functions for stochastic processes.
"""

from typing import Optional
import torch
import numpy as np

from .base import BasePDE
from ..config import MertonModelConfig


class FokkerPlanckMerton(BasePDE):
    """
    Fokker-Planck equation for Merton structural credit risk model.

    The Merton model assumes firm asset value V_t follows Geometric Brownian Motion:
        dV_t = μ V_t dt + σ V_t dW_t

    In log-space X_t = ln(V_t), the dynamics become:
        dX_t = (μ - σ²/2) dt + σ dW_t = α dt + σ dW_t

    The probability density p(x, t) of X_t satisfies the Fokker-Planck equation:
        ∂p/∂t + α ∂p/∂x - (σ²/2) ∂²p/∂x² = 0

    This is an advection-diffusion equation with:
        - Drift coefficient: α = μ - σ²/2
        - Diffusion coefficient: σ²/2

    Args:
        config: Merton model configuration
        x0: Initial log-asset value
        device: Computation device
        dtype: Tensor data type
    """

    def __init__(
        self,
        config: MertonModelConfig,
        x0: Optional[float] = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(spatial_dim=1, device=device, dtype=dtype)

        self.config = config
        self.mu = config.mu
        self.sigma = config.sigma
        self.alpha = config.alpha  # Effective drift: μ - σ²/2
        self.x0 = x0 if x0 is not None else config.x0

        # Diffusion coefficient (constant)
        self.D = 0.5 * self.sigma ** 2

    def pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fokker-Planck PDE residual.

        The FPE in conservative form:
            ∂p/∂t = -∂/∂x(α p) + (σ²/2) ∂²p/∂x²

        Expanding the drift term:
            ∂p/∂t = -α ∂p/∂x + (σ²/2) ∂²p/∂x²

        Residual form (should be zero):
            R = ∂p/∂t + α ∂p/∂x - (σ²/2) ∂²p/∂x²

        Args:
            x: Spatial coordinates [Batch, 1]
            t: Time coordinates [Batch, 1]
            u: Density p(x, t) [Batch, 1]
            u_t: Time derivative ∂p/∂t [Batch, 1]
            u_x: Spatial derivative ∂p/∂x [Batch, 1]
            u_xx: Second spatial derivative ∂²p/∂x² [Batch, 1]

        Returns:
            PDE residual [Batch, 1]
        """
        # Advection term: α ∂p/∂x
        advection = self.alpha * u_x

        # Diffusion term: -(σ²/2) ∂²p/∂x²
        diffusion = -self.D * u_xx

        # Residual: ∂p/∂t + α ∂p/∂x - (σ²/2) ∂²p/∂x²
        residual = u_t + advection + diffusion

        return residual

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initial condition: Dirac delta at x_0.

        For numerical stability, we approximate δ(x - x_0) with a narrow Gaussian:
            p(x, 0) ≈ N(x; x_0, ε²) = (1/√(2πε²)) exp(-(x - x_0)²/(2ε²))

        where ε is a small regularization parameter.

        Args:
            x: Spatial coordinates [Batch, 1]

        Returns:
            Initial density [Batch, 1]
        """
        epsilon = 0.01  # Regularization parameter
        variance = epsilon ** 2

        # Gaussian approximation of delta function
        coeff = 1.0 / np.sqrt(2 * np.pi * variance)
        exponent = -(x - self.x0) ** 2 / (2 * variance)
        p0 = coeff * torch.exp(exponent)

        return p0

    def analytical_solution(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Analytical solution of Fokker-Planck equation for constant coefficients.

        For the advection-diffusion equation with constant α and D = σ²/2,
        starting from δ(x - x_0), the solution is a Gaussian:

            p(x, t) = N(x; μ_t, σ_t²)

        where:
            μ_t = x_0 + α t           (mean drifts linearly)
            σ_t² = σ² t               (variance grows linearly)

        Args:
            x: Spatial coordinates [Batch, 1] or [Batch]
            t: Time values [Batch, 1] or [Batch]

        Returns:
            Analytical density [Batch, 1]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Avoid t=0 to prevent division by zero
        t = torch.clamp(t, min=1e-6)

        # Time-evolved mean and variance
        mu_t = self.x0 + self.alpha * t
        var_t = self.sigma ** 2 * t

        # Gaussian density
        coeff = 1.0 / torch.sqrt(torch.tensor(2 * np.pi) * var_t)
        exponent = -(x - mu_t) ** 2 / (2 * var_t)
        p = coeff * torch.exp(exponent)

        return p

    def analytical_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Analytical score function for the Merton model.

        For the Gaussian density with mean μ_t = x_0 + α*t and variance σ²*t,
        the score function (gradient of log-density) is:

            s(x, t) = ∇_x log p(x, t) = -(x - μ_t) / (σ² t)

        This provides exact ground truth for validating PINN and SGM scores.

        Args:
            x: Spatial coordinates [Batch, 1] or [Batch]
            t: Time values [Batch, 1] or [Batch]

        Returns:
            Score values [Batch, 1]

        Example:
            >>> equation = FokkerPlanckMerton(config)
            >>> x = torch.tensor([[0.0], [1.0]])
            >>> t = torch.tensor([[0.5], [0.5]])
            >>> score = equation.analytical_score(x, t)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Avoid t=0 to prevent division by zero
        t = torch.clamp(t, min=1e-6)

        # Time-evolved mean and variance
        mu_t = self.x0 + self.alpha * t
        var_t = self.sigma ** 2 * t

        # Score: s(x, t) = -(x - μ_t) / (σ² t)
        score = -(x - mu_t) / var_t

        return score

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FokkerPlanckMerton(μ={self.mu:.4f}, σ={self.sigma:.4f}, "
            f"α={self.alpha:.4f}, x0={self.x0:.4f})"
        )
