"""
Merton structural model: Fokker-Planck PDE and SDE implementations.

The multivariate Merton model assumes a constant-coefficient linear SDE:
    dX_t = μ dt + Σ dW_t

The probability density p(x, t) of X_t satisfies the Fokker-Planck equation:
    ∂p/∂t + μ·∇p - (1/2) ∑ᵢⱼ Dᵢⱼ ∂²p/∂xᵢ∂xⱼ = 0

where D = ΣΣᵀ is the diffusion matrix.  Starting from a point mass at x₀,
the density is Gaussian for all t > 0:
    p(x, t) = 𝒩(x; x₀ + μt, Dt)

This closed-form solution makes the Merton model an ideal test bed for
PINN and Score-PINN solvers.
"""

from typing import Optional
import torch
import numpy as np

from .base import BasePDE, BaseSDE


class FokkerPlanckMertonND(BasePDE, BaseSDE):
    """
    N-dimensional Fokker-Planck equation for multivariate Merton model.

    The n-dimensional Merton model assumes:
        dX_t = μ dt + Σ dW_t

    where:
        X_t ∈ ℝⁿ: state vector
        μ ∈ ℝⁿ: constant drift vector
        Σ ∈ ℝⁿˣⁿ: constant volatility matrix
        W_t ∈ ℝⁿ: n-dimensional Brownian motion

    The probability density p(x, t) satisfies the Fokker-Planck equation:
        ∂p/∂t + μ·∇p - (1/2)∑ᵢⱼ Dᵢⱼ ∂²p/∂xᵢ∂xⱼ = 0

    where D = ΣΣᵀ is the diffusion matrix.

    For constant coefficients starting from x₀, the analytical solution is:
        p(x, t) = 𝒩(x; x₀ + μt, Dt)

    Args:
        spatial_dim: Dimension of state space (n)
        mu: Drift vector [n] or scalar (broadcasted to all dimensions)
        sigma: Volatility matrix [n, n] or vector [n] (diagonal) or scalar (isotropic)
        x0: Initial state [n] or scalar (broadcasted)
        device: Computation device
        dtype: Tensor data type

    Example:
        >>> # 2D model with diagonal covariance
        >>> equation = FokkerPlanckMertonND(
        ...     spatial_dim=2,
        ...     mu=[0.1, 0.05],
        ...     sigma=[0.2, 0.3],  # diagonal: σ₁=0.2, σ₂=0.3
        ...     x0=[0.0, 0.0]
        ... )
    """

    def __init__(
        self,
        spatial_dim: int,
        mu: Optional[float | list | np.ndarray] = None,
        sigma: Optional[float | list | np.ndarray] = None,
        x0: Optional[float | list | np.ndarray] = None,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        # Explicitly call BasePDE.__init__ to set shared attributes (spatial_dim, device, dtype).
        # BaseSDE has an identical __init__ signature; calling BasePDE is sufficient since
        # both set the same attributes.
        BasePDE.__init__(self, spatial_dim=spatial_dim, device=device, dtype=dtype)

        # --- Drift vector μ ---
        if mu is None:
            mu = 0.0
        if np.isscalar(mu):
            mu = np.full(spatial_dim, mu)
        else:
            mu = np.asarray(mu)
            if mu.shape != (spatial_dim,):
                raise ValueError(f"mu must have shape ({spatial_dim},), got {mu.shape}")

        self.mu = torch.tensor(mu, dtype=dtype, device=device)

        # --- Volatility matrix Σ ---
        if sigma is None:
            sigma = 1.0
        if np.isscalar(sigma):
            # Isotropic: Σ = σI
            sigma = np.eye(spatial_dim) * sigma
        else:
            sigma = np.asarray(sigma)
            if sigma.ndim == 1:
                # Diagonal: Σ = diag(σ)
                if sigma.shape != (spatial_dim,):
                    raise ValueError(f"sigma vector must have shape ({spatial_dim},), got {sigma.shape}")
                sigma = np.diag(sigma)
            elif sigma.ndim == 2:
                # Full matrix
                if sigma.shape != (spatial_dim, spatial_dim):
                    raise ValueError(
                        f"sigma matrix must have shape ({spatial_dim}, {spatial_dim}), got {sigma.shape}"
                    )
            else:
                raise ValueError(f"sigma must be scalar, vector, or matrix, got shape {sigma.shape}")

        self.sigma = torch.tensor(sigma, dtype=dtype, device=device)

        # --- Diffusion matrix D = ΣΣᵀ ---
        self.D = torch.matmul(self.sigma, self.sigma.T)

        # --- Initial state x₀ ---
        if x0 is None:
            x0 = 0.0
        if np.isscalar(x0):
            x0 = np.full(spatial_dim, x0)
        else:
            x0 = np.asarray(x0)
            if x0.shape != (spatial_dim,):
                raise ValueError(f"x0 must have shape ({spatial_dim},), got {x0.shape}")

        self.x0 = torch.tensor(x0, dtype=dtype, device=device)

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
        Compute n-dimensional Fokker-Planck PDE residual.

        The n-D FPE:
            ∂p/∂t + μ·∇p - (1/2)∑ᵢⱼ Dᵢⱼ ∂²p/∂xᵢ∂xⱼ = 0

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Time coordinates [Batch, 1]
            u: Density p(x, t) [Batch, 1]
            u_t: Time derivative ∂p/∂t [Batch, 1]
            u_x: Spatial gradient ∇p [Batch, spatial_dim]
            u_xx: Hessian matrix [Batch, spatial_dim, spatial_dim]

        Returns:
            PDE residual [Batch, 1]
        """
        # --- Drift term: μ·∇p ---
        drift_term = torch.sum(self.mu * u_x, dim=-1, keepdim=True)  # [Batch, 1]

        # --- Diffusion term: (1/2) ∑ᵢⱼ Dᵢⱼ ∂²p/∂xᵢ∂xⱼ ---
        # Contract: ∑ᵢⱼ Dᵢⱼ Hᵢⱼ = Tr(D H) since H is symmetric
        diffusion_term = 0.5 * torch.einsum('ij,bij->b', self.D, u_xx).unsqueeze(-1)  # [Batch, 1]

        # --- PDE residual: ∂p/∂t + μ·∇p - (1/2)∑ᵢⱼ Dᵢⱼ ∂²p/∂xᵢ∂xⱼ ---
        residual = u_t + drift_term - diffusion_term

        return residual

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initial condition: Dirac delta at x₀.

        For numerical stability, we approximate δ(x - x₀) with a narrow Gaussian:
            p(x, 0) ≈ 𝒩(x; x₀, ε²I)

        Args:
            x: Spatial coordinates [Batch, spatial_dim]

        Returns:
            Initial density [Batch, 1]
        """
        epsilon = 0.01  # Regularization parameter
        variance = epsilon ** 2

        # Multivariate Gaussian
        diff = x - self.x0  # [Batch, spatial_dim]
        exponent = -0.5 * torch.sum(diff ** 2, dim=-1, keepdim=True) / variance  # [Batch, 1]
        coeff = 1.0 / ((2 * np.pi * variance) ** (self.spatial_dim / 2))
        p0 = coeff * torch.exp(exponent)

        return p0

    def analytical_solution(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Analytical solution: multivariate Gaussian.

        For constant μ and D, starting from x₀, the solution is:
            p(x, t) = 𝒩(x; x₀ + μt, Dt)

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1] or [Batch]

        Returns:
            Analytical density [Batch, 1]
        """
        # --- Shapes ---
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, spatial_dim]
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [Batch, 1]

        t_clamped = torch.clamp(t, min=1e-6)
        batch_size = x.shape[0]

        # --- Mean and covariance: μ_t = x₀ + μt, Cov_t = Dt ---
        mu_t = self.x0 + self.mu * t_clamped  # [Batch, spatial_dim]
        t_expanded = t_clamped.view(batch_size, 1, 1)
        cov_t = self.D.unsqueeze(0) * t_expanded  # [Batch, spatial_dim, spatial_dim]

        diff = x - mu_t  # [Batch, spatial_dim]

        # --- Covariance inverse and determinant ---
        det_cov = torch.linalg.det(cov_t)  # [Batch]
        inv_cov = torch.linalg.inv(cov_t)  # [Batch, spatial_dim, spatial_dim]

        # --- Mahalanobis distance ---
        mahal = torch.einsum('bi,bij,bj->b', diff, inv_cov, diff)  # [Batch]

        # --- Gaussian PDF ---
        coeff = 1.0 / torch.sqrt((2 * np.pi) ** self.spatial_dim * det_cov)
        p = coeff * torch.exp(-0.5 * mahal)

        return p.unsqueeze(-1)  # [Batch, 1]

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift coefficient f(x, t) = μ (constant drift).

        Args:
            x: State coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]

        Returns:
            Drift vector μ [Batch, spatial_dim]
        """
        batch_size = x.shape[0]
        # Expand constant drift to batch size
        return self.mu.unsqueeze(0).expand(batch_size, -1)

    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient G(x, t) = Σ (constant diffusion).

        Args:
            x: State coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]

        Returns:
            Diffusion matrix Σ [Batch, spatial_dim, spatial_dim]
        """
        batch_size = x.shape[0]
        # Expand constant diffusion to batch size
        return self.sigma.unsqueeze(0).expand(batch_size, -1, -1)

    def is_constant_coefficients(self) -> bool:
        """
        Check if drift and diffusion are constant.

        Returns:
            True (Merton model has constant coefficients)
        """
        return True

    def initial_score(self, x: torch.Tensor, t_epsilon: float = 0.1) -> torch.Tensor:
        """
        Initial score function for n-D Merton model.

        For the Dirac delta initial condition p₀(x) = δ(x - x₀), we evaluate
        the score at a small time t_ε to avoid the singularity at t=0.

        At small time, the distribution is approximately Gaussian with mean x₀
        and covariance D·t_ε, giving:
            s₀(x) ≈ s(x, t_ε) = -(D·t_ε)⁻¹(x - x₀ - μ·t_ε)

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t_epsilon: Regularization time for Dirac delta (default: 0.1)

        Returns:
            Initial score s₀(x) [Batch, spatial_dim]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # --- Evaluate at t_epsilon to regularize Dirac delta ---
        t_small = torch.full((batch_size, 1), t_epsilon, device=self.device, dtype=self.dtype)
        return self.analytical_score(x, t_small)

    def analytical_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Analytical score function for n-D Merton model.

        For the Gaussian density with mean μ_t = x₀ + μt and covariance Dt,
        the score function is:
            s(x, t) = ∇_x log p(x, t) = -(Dt)⁻¹(x - x₀ - μt)

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1] or [Batch]

        Returns:
            Score vectors [Batch, spatial_dim]
        """
        # --- Shapes ---
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t_clamped = torch.clamp(t, min=1e-6)
        batch_size = x.shape[0]

        # --- Mean and covariance: μ_t = x₀ + μt, Cov_t = Dt ---
        mu_t = self.x0 + self.mu * t_clamped  # [Batch, spatial_dim]
        t_expanded = t_clamped.view(batch_size, 1, 1)
        cov_t = self.D.unsqueeze(0) * t_expanded  # [Batch, spatial_dim, spatial_dim]

        # --- Inverse covariance ---
        inv_cov = torch.linalg.inv(cov_t)  # [Batch, spatial_dim, spatial_dim]

        # --- Score computation: s = -(Dt)⁻¹(x - μ_t) ---
        diff = x - mu_t  # [Batch, spatial_dim]
        score = -torch.einsum('bij,bj->bi', inv_cov, diff)  # [Batch, spatial_dim]

        return score

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FokkerPlanckMertonND(spatial_dim={self.spatial_dim}, "
            f"μ={self.mu.cpu().numpy()}, D_shape={self.D.shape})"
        )
