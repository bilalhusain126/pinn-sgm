"""
Abstract base class for forward SDEs in the SGM framework.

ForwardSDE extends BaseSDE with the two additional methods required for
denoising score matching and reverse-SDE sampling:
  - marginal_score:  DSM training target ∇_{x(t)} log p(x(t) | x(0))
  - sample_marginal: forward noising x(0) → x(t)

It also provides a concrete Euler-Maruyama path simulator that any subclass
inherits for free (override for closed-form efficiency, as MertonSDE does).

Inheritance chain:
    BaseSDE  (equations/base.py)  ← drift, diffusion, initial_score, analytical_score
        └── ForwardSDE            ← + marginal_score, sample_marginal, simulate_paths
                └── MertonSDE    (sgm/merton.py)
"""

import logging
from abc import abstractmethod
from typing import Optional
import torch

from pinn_sgm.equations.base import BaseSDE

logger = logging.getLogger(__name__)


class ForwardSDE(BaseSDE):
    """
    Abstract forward SDE for score-based generative modelling.

    Extends BaseSDE with the SGM-specific interface needed by DSMTrainer
    and ReverseDiffusionSampler. Concretely:

      - DSMTrainer calls sample_marginal to apply forward noise, then
        marginal_score to compute the denoising target.

      - ReverseDiffusionSampler calls drift and diffusion (inherited from
        BaseSDE) to run the reverse Euler-Maruyama predictor step.

    Subclasses must implement all four abstract methods. simulate_paths is
    provided as a generic Euler-Maruyama utility and can be overridden with
    a closed-form sampler for efficiency.
    """

    # ------------------------------------------------------------------
    # SGM-specific abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def marginal_score(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DSM training target: ∇_{x(t)} log p(x(t) | x(0)).

        Depends on both the noisy sample x and the clean starting point x0.
        This is distinct from BaseSDE.analytical_score(x, t), which computes
        ∇ log p(x, t) marginalised over the initial condition.

        Args:
            x:  Noisy state x(t) [B, d]
            x0: Clean starting state x(0) [B, d]
            t:  Time [B, 1]

        Returns:
            DSM target [B, d]
        """

    @abstractmethod
    def sample_marginal(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample x(t) ~ p(x(t) | x(0)) from the forward transition kernel.

        Used by DSMTrainer to apply forward noise to clean data.

        Args:
            x0: Clean starting state [B, d]
            t:  Time [B, 1]

        Returns:
            Noisy sample x(t) [B, d]
        """

    # ------------------------------------------------------------------
    # Concrete utility — generic Euler-Maruyama path simulator
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
        Simulate forward SDE paths using Euler-Maruyama.

        x_{k+1} = x_k + f(x_k, t_k) Δt + G(x_k, t_k) √Δt z_k,  z_k ~ N(0, I)

        Subclasses with constant or affine coefficients should override this
        with a closed-form sampler for efficiency.

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

        # --- Initialise paths ---
        x0_flat = x0.view(-1).to(device=self.device, dtype=self.dtype)
        paths = torch.zeros(n_paths, n_steps + 1, d, device=self.device, dtype=self.dtype)
        paths[:, 0, :] = x0_flat.unsqueeze(0).expand(n_paths, -1)

        x = paths[:, 0, :].clone()  # [n_paths, d]

        for k in range(n_steps):
            t_val = k * dt
            t_batch = torch.full((n_paths, 1), t_val, device=self.device, dtype=self.dtype)

            # --- Euler-Maruyama step ---
            f = self.drift(x, t_batch)                    # [n_paths, d]
            G = self.diffusion(x, t_batch)                # [n_paths, d, d]
            z = torch.randn(n_paths, d, device=self.device, dtype=self.dtype)
            GdW = torch.einsum('bij,bj->bi', G, sqrt_dt * z)  # [n_paths, d]
            x = x + f * dt + GdW

            paths[:, k + 1, :] = x

        logger.info(
            "Simulated %d paths (d=%d, T=%.2f, n_steps=%d, dt=%.4f)",
            n_paths, d, T, n_steps, dt,
        )
        return paths
