"""
Fokker-Planck equation implementations for financial models.

This module implements the Fokker-Planck equation (FPE) governing the evolution
of probability density functions for stochastic processes.

References:
    - PhD Research Document: Section 2.1 (The Fokker-Planck Equation)
    - PhD Research Document: Section 3.2 (Log-Space Transformation and FPE Derivation)
"""

from typing import Optional, Callable
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

    References:
        - PhD Research Document: Equation (3.1), (3.4), (3.6)
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
        coeff = 1.0 / torch.sqrt(2 * np.pi * variance)
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

        References:
            - PhD Research Document: Section 3.2
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
        coeff = 1.0 / torch.sqrt(2 * np.pi * var_t)
        exponent = -(x - mu_t) ** 2 / (2 * var_t)
        p = coeff * torch.exp(exponent)

        return p

    def default_probability(
        self,
        debt_threshold: Optional[float] = None,
        t: Optional[float] = None
    ) -> float:
        """
        Compute probability of default P(X_T < ln(D)) = P(V_T < D).

        Using the analytical solution, this is:
            P(default) = Φ((ln(D) - μ_T) / σ_T)

        where Φ is the standard normal CDF.

        Args:
            debt_threshold: Debt level D (uses config if not provided)
            t: Time horizon (terminal time)

        Returns:
            Default probability

        References:
            - Merton (1974): "On the Pricing of Corporate Debt"
        """
        if debt_threshold is None:
            if self.config.debt_threshold is None:
                raise ValueError("debt_threshold must be provided or set in config")
            debt_threshold = self.config.debt_threshold

        if t is None:
            raise ValueError("Time horizon t must be provided")

        # Log of debt threshold
        ln_D = np.log(debt_threshold)

        # Mean and std at time t
        mu_t = self.x0 + self.alpha * t
        sigma_t = self.sigma * np.sqrt(t)

        # Standardized distance to default
        d = (ln_D - mu_t) / sigma_t

        # Default probability using normal CDF
        from scipy.stats import norm
        prob_default = norm.cdf(d)

        return prob_default

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FokkerPlanckMerton(μ={self.mu:.4f}, σ={self.sigma:.4f}, "
            f"α={self.alpha:.4f}, x0={self.x0:.4f})"
        )


class FokkerPlanckGeneral(BasePDE):
    """
    General Fokker-Planck equation with arbitrary drift and diffusion.

    The general FPE for a stochastic process:
        dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t

    has the form:
        ∂p/∂t = -∂/∂x[μ(x,t) p] + (1/2) ∂²/∂x²[σ²(x,t) p]

    For constant coefficients (μ, σ independent of x, t), this reduces to:
        ∂p/∂t + μ ∂p/∂x - (σ²/2) ∂²p/∂x² = 0

    Args:
        drift_fn: Drift function μ(x, t)
        diffusion_fn: Diffusion function σ(x, t)
        initial_condition_fn: Initial density p(x, 0)
        device: Computation device
        dtype: Tensor data type

    References:
        - PhD Research Document: Theorem 2.1 (Equation 2.2)
    """

    def __init__(
        self,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_condition_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        super().__init__(spatial_dim=1, device=device, dtype=dtype)

        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.initial_condition_fn = initial_condition_fn

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
        Compute general Fokker-Planck PDE residual.

        Conservative form:
            ∂p/∂t = -∂/∂x[μ(x,t) p] + (1/2) ∂²/∂x²[σ²(x,t) p]

        Expanding (using product rule):
            ∂p/∂t = -[∂μ/∂x p + μ ∂p/∂x] + (1/2)[∂²(σ²)/∂x² p + 2 ∂(σ²)/∂x ∂p/∂x + σ² ∂²p/∂x²]

        For constant coefficients (μ_x = 0, σ_x = 0):
            ∂p/∂t = -μ ∂p/∂x + (σ²/2) ∂²p/∂x²

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
        # Evaluate drift and diffusion at (x, t)
        mu = self.drift_fn(x, t)  # [Batch, 1]
        sigma = self.diffusion_fn(x, t)  # [Batch, 1]
        D = 0.5 * sigma ** 2  # Diffusion coefficient

        # Advection term: μ ∂p/∂x
        advection = mu * u_x

        # Diffusion term: -(σ²/2) ∂²p/∂x²
        diffusion = -D * u_xx

        # Residual: ∂p/∂t + μ ∂p/∂x - (σ²/2) ∂²p/∂x²
        residual = u_t + advection + diffusion

        return residual

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition using provided function.

        Args:
            x: Spatial coordinates [Batch, 1]

        Returns:
            Initial density p(x, 0) [Batch, 1]
        """
        return self.initial_condition_fn(x)

    def analytical_solution(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        No general analytical solution available.

        Returns:
            None (no analytical solution)
        """
        return None

    def __repr__(self) -> str:
        """String representation."""
        return "FokkerPlanckGeneral(drift=custom, diffusion=custom)"
