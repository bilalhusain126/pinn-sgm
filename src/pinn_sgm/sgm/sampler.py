"""
Reverse-SDE sampler for score-based generative models.

Implements the Predictor-Corrector (PC) sampler from Song et al.:
    - Predictor: Euler-Maruyama discretisation of the reverse-time SDE
    - Corrector: Langevin MCMC steps at each time point

The sampler is fully equation-agnostic. It accepts any ForwardSDE (for drift
and diffusion coefficients) and any score_fn callable. score_fn can be:
    - equation.analytical_score          (ground-truth, for validation)
    - ScoreExtractor(network)            (PINN-derived score)
    - trainer.predict_score              (empirically trained s_θ)
    - lambda x, t: (1-γ)*s_θ(x,t) + γ*s_φ(x,t)   (Method B blending)
"""

import logging
from typing import Callable, Optional
import torch

from .base import ForwardSDE

logger = logging.getLogger(__name__)


class ReverseDiffusionSampler:
    """
    Predictor-Corrector reverse-SDE sampler.

    Discretises [T, t_eps] into n_steps intervals and at each step applies:

    Predictor (reverse Euler-Maruyama, Eq. 6):
        x ← x + [f(x,t) - G(x,t)Gᵀ(x,t) s(x,t)] |Δt| + G(x,t) √|Δt| z

    Corrector (Langevin MCMC, n_corrector_steps times at fixed t):
        x ← x + ε · s(x, t) + √(2ε) · z

    Set n_corrector_steps=0 to run predictor-only (plain reverse E-M).

    Args:
        sde:                  Forward SDE — supplies drift and diffusion
        score_fn:             Callable (x: [B,d], t: [B,1]) → score [B,d]
        n_steps:              Reverse-time discretisation steps
        n_corrector_steps:    Langevin steps per predictor step (0 = EM only)
        corrector_step_size:  Langevin step size ε
        t_eps:                Minimum time to integrate to (matches DSMConfig)
        device:               Computation device
    """

    def __init__(
        self,
        sde: ForwardSDE,
        score_fn: Callable,
        n_steps: int = 500,
        n_corrector_steps: int = 1,
        corrector_step_size: float = 0.01,
        t_eps: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        self.sde = sde
        self.score_fn = score_fn
        self.n_steps = n_steps
        self.n_corrector_steps = n_corrector_steps
        self.corrector_step_size = corrector_step_size
        self.t_eps = t_eps
        self.device = device if device is not None else sde.device

    @torch.no_grad()
    def sample(
        self,
        x_T: torch.Tensor,
        T: float,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples by running the reverse SDE from t=T to t=t_eps.

        Args:
            x_T:               Samples from the prior p_T  [B, d]
            T:                 Starting time (must match forward SDE terminal time)
            return_trajectory: If True, return all intermediate states [B, n_steps+1, d]

        Returns:
            Final samples [B, d], or full trajectory [B, n_steps+1, d]
        """
        x = x_T.to(device=self.device, dtype=self.sde.dtype)
        B, d = x.shape

        # Time grid: T → t_eps  (n_steps+1 points, n_steps intervals)
        times = torch.linspace(T, self.t_eps, self.n_steps + 1,
                               device=self.device, dtype=self.sde.dtype)
        dt = (self.t_eps - T) / self.n_steps  # negative

        if return_trajectory:
            trajectory = torch.zeros(B, self.n_steps + 1, d,
                                     device=self.device, dtype=self.sde.dtype)
            trajectory[:, 0, :] = x

        for i in range(self.n_steps):
            t_i = times[i]
            t_next = times[i + 1]

            t_batch = torch.full((B, 1), t_i.item(),
                                 device=self.device, dtype=self.sde.dtype)

            # ---- Predictor: reverse Euler-Maruyama ----
            x = self._predictor_step(x, t_batch, dt)

            # ---- Corrector: Langevin MCMC at t_next ----
            if self.n_corrector_steps > 0:
                t_next_batch = torch.full((B, 1), t_next.item(),
                                         device=self.device, dtype=self.sde.dtype)
                for _ in range(self.n_corrector_steps):
                    x = self._corrector_step(x, t_next_batch)

            if return_trajectory:
                trajectory[:, i + 1, :] = x

        logger.info(
            "Reverse SDE: generated %d samples (%dD) with %d steps, "
            "%d corrector steps per step",
            B, d, self.n_steps, self.n_corrector_steps,
        )

        if return_trajectory:
            return trajectory
        return x

    def _predictor_step(
        self,
        x: torch.Tensor,
        t_batch: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        One reverse Euler-Maruyama step.

        dx = [f(x,t) - G(x,t)Gᵀ(x,t) s(x,t)] |Δt| + G(x,t) √|Δt| z
        """
        B = x.shape[0]
        abs_dt = abs(dt)
        sqrt_abs_dt = abs_dt ** 0.5

        f = self.sde.drift(x, t_batch)       # [B, d]
        G = self.sde.diffusion(x, t_batch)   # [B, d, d]
        s = self.score_fn(x, t_batch)        # [B, d]

        # GGᵀ s: [B, d]
        Gt = G.transpose(-1, -2)                          # [B, d, d]
        GGt = torch.bmm(G, Gt)                            # [B, d, d]
        GGt_s = torch.einsum('bij,bj->bi', GGt, s)        # [B, d]

        # Noise term: G √|Δt| z
        z = torch.randn_like(x)
        dW = sqrt_abs_dt * z                              # [B, d]
        G_dW = torch.einsum('bij,bj->bi', G, dW)          # [B, d]

        x_new = x + (f - GGt_s) * abs_dt + G_dW
        return x_new

    def _corrector_step(
        self,
        x: torch.Tensor,
        t_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        One Langevin MCMC corrector step.

        x ← x + ε · s(x, t) + √(2ε) · z
        """
        eps = self.corrector_step_size
        s = self.score_fn(x, t_batch)   # [B, d]
        z = torch.randn_like(x)
        x_new = x + eps * s + (2 * eps) ** 0.5 * z
        return x_new

    def sample_euler_maruyama(
        self,
        x_T: torch.Tensor,
        T: float,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Predictor-only reverse E-M sampling (no Langevin corrector).

        Equivalent to sample(...) with n_corrector_steps=0 but does not
        temporarily modify the sampler state.
        """
        orig = self.n_corrector_steps
        self.n_corrector_steps = 0
        result = self.sample(x_T, T, return_trajectory=return_trajectory)
        self.n_corrector_steps = orig
        return result

    def sample_pc(
        self,
        x_T: torch.Tensor,
        T: float,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Full Predictor-Corrector sampling.

        Equivalent to sample(...) but explicit alias for clarity.
        Uses n_corrector_steps and corrector_step_size set at construction.
        """
        return self.sample(x_T, T, return_trajectory=return_trajectory)
