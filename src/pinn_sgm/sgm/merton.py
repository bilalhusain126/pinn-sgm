"""
Merton SDE: concrete ForwardSDE for the multivariate Merton structural model.

The forward SDE is:
    dX_t = μ dt + Σ dW_t

with closed-form Gaussian transition kernel:
    p(x(t) | x(0)) = N(x(0) + μt,  Dt)    where D = ΣΣᵀ

This gives closed-form implementations of all four abstract methods, so no
numerical SDE integration is needed for training or sampling.
"""

import logging
from typing import Optional
import torch

from .base import ForwardSDE

logger = logging.getLogger(__name__)


class MertonSDE(ForwardSDE):
    """
    Forward SDE for the multivariate Merton model.

    Wraps a FokkerPlanckMertonND equation and extends it with the SGM
    interface (marginal_score, sample_marginal). All four abstract methods
    have closed-form implementations; simulate_paths is overridden with an
    exact Gaussian-increment sampler for efficiency.

    Note on marginal_score vs analytical_score:
        - analytical_score(x, t)      from BaseSDE: ∇ log p(x, t) assuming
          p_0 = δ(x − x₀) with x₀ fixed — used by ScorePINNSolver.
        - marginal_score(x, x0, t)    from ForwardSDE: ∇ log p(x(t) | x(0))
          with x0 as an explicit batch argument — used by DSMTrainer.

    Args:
        equation: A configured FokkerPlanckMertonND instance.
    """

    def __init__(self, equation):
        super().__init__(
            spatial_dim=equation.spatial_dim,
            device=equation.device,
            dtype=equation.dtype,
        )
        self.equation = equation

        # --- Cache equation tensors ---
        self._mu    = equation.mu     # [d]
        self._sigma = equation.sigma  # [d, d]
        self._D     = equation.D      # [d, d]  D = ΣΣᵀ

        # --- Pre-compute Cholesky of D for efficient sampling ---
        # D = LLᵀ  ⟹  Dt = (L√t)(L√t)ᵀ
        self._L = torch.linalg.cholesky(self._D)  # [d, d]

    # ------------------------------------------------------------------
    # BaseSDE interface (drift / diffusion / initial_score / analytical_score)
    # ------------------------------------------------------------------

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Constant drift μ broadcast to batch."""
        return self.equation.drift(x, t)

    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Constant diffusion matrix Σ broadcast to batch."""
        return self.equation.diffusion(x, t)

    def initial_score(self, x: torch.Tensor, t_epsilon: float = 0.1) -> torch.Tensor:
        """Delegate to equation (evaluates analytical score at t_epsilon)."""
        return self.equation.initial_score(x, t_epsilon)

    def analytical_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Delegate to equation: ∇ log p(x, t) assuming p_0 = δ(x − x₀)."""
        return self.equation.analytical_score(x, t)

    # ------------------------------------------------------------------
    # ForwardSDE interface (marginal_score / sample_marginal)
    # ------------------------------------------------------------------

    def marginal_score(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DSM target: ∇_{x(t)} log p(x(t) | x(0)) = -(Dt)⁻¹ (x - x0 - μt).

        Args:
            x:  Noisy sample [B, d]
            x0: Clean sample [B, d]
            t:  Time [B, 1]

        Returns:
            Score [B, d]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_clamped = torch.clamp(t, min=1e-6)

        B = x.shape[0]

        # --- Covariance and its inverse ---
        cov_t = self._D.unsqueeze(0) * t_clamped.view(B, 1, 1)  # [B, d, d]
        inv_cov = torch.linalg.inv(cov_t)                        # [B, d, d]

        # --- Conditional mean: x0 + μt ---
        mean_t = x0 + self._mu.unsqueeze(0) * t_clamped          # [B, d]

        diff = x - mean_t                                         # [B, d]
        return -torch.einsum('bij,bj->bi', inv_cov, diff)         # [B, d]

    def sample_marginal(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample x(t) ~ N(x0 + μt, Dt) using the Cholesky factor of D.

        Args:
            x0: Clean sample [B, d]
            t:  Time [B, 1]

        Returns:
            Noisy sample [B, d]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_clamped = torch.clamp(t, min=1e-6)

        B = x0.shape[0]

        # --- Mean: x0 + μt ---
        mean_t = x0 + self._mu.unsqueeze(0) * t_clamped          # [B, d]

        # --- Cholesky std: L_t = L √t  (so L_t L_tᵀ = Dt) ---
        sqrt_t = torch.sqrt(t_clamped)                            # [B, 1]
        L_t = self._L.unsqueeze(0) * sqrt_t.view(B, 1, 1)        # [B, d, d]

        z = torch.randn(B, self.spatial_dim, device=self.device, dtype=self.dtype)
        return mean_t + torch.einsum('bij,bj->bi', L_t, z)        # [B, d]

    # ------------------------------------------------------------------
    # Override simulate_paths with closed-form Gaussian increments
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        x0: torch.Tensor,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Exact Gaussian-increment path simulation.

        Uses x_{k+1} - x_k ~ N(μΔt, DΔt) directly rather than the generic
        E-M loop in ForwardSDE, which is equivalent but faster since drift
        and diffusion are constant.

        Args:
            x0:      Initial state [d] or [1, d]
            T:       Terminal time
            n_steps: Number of time steps
            n_paths: Number of independent paths
            seed:    Optional random seed

        Returns:
            Paths [n_paths, n_steps+1, d]
        """
        if seed is not None:
            torch.manual_seed(seed)

        d = self.spatial_dim
        dt = T / n_steps
        sqrt_dt = dt ** 0.5

        x0_flat = x0.view(-1).to(device=self.device, dtype=self.dtype)
        paths = torch.zeros(n_paths, n_steps + 1, d, device=self.device, dtype=self.dtype)
        paths[:, 0, :] = x0_flat.unsqueeze(0).expand(n_paths, -1)

        # --- Constant increment statistics ---
        mu_dt = self._mu * dt        # [d]
        L_dt  = self._L  * sqrt_dt   # [d, d]

        x = paths[:, 0, :].clone()
        for k in range(n_steps):
            z = torch.randn(n_paths, d, device=self.device, dtype=self.dtype)
            x = x + mu_dt + torch.einsum('ij,bj->bi', L_dt, z)
            paths[:, k + 1, :] = x

        logger.info(
            "Simulated %d Merton paths (d=%d, T=%.2f, n_steps=%d)",
            n_paths, d, T, n_steps,
        )
        return paths
