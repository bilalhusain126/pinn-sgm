"""
Denoising Score Matching (DSM) trainer for score-based generative models.

Implements the continuous-time DSM objective from Song et al. (ICLR 2021),
Eq. 7. The trainer is equation-agnostic: it accepts any ForwardSDE and any
score network, and takes external data (x0_data) rather than simulating paths
internally. This makes swapping synthetic data for real market data trivial.

Optional Method A hook: add a physics regularisation penalty
    L_total = L_DSM + λ · E[||s_θ(x,t) - s_φ(x,t)||²]
by setting theory_score_fn and lambda_physics in DSMConfig.
"""

import logging
from typing import Callable, Dict, List, Optional
import torch
import torch.nn as nn

from ..config import DSMConfig
from .base import ForwardSDE

logger = logging.getLogger(__name__)


class DSMTrainer:
    """
    Trains a score network s_θ(x, t) ≈ ∇_x log p_t(x) using denoising score
    matching (Song et al. 2021, Eq. 7).

    Training loop (per epoch):
        1. Sample batch of x0 from x0_data
        2. Sample t ~ Uniform(t_eps, T)
        3. Sample xt ~ sde.sample_marginal(x0, t)   [forward noise]
        4. Compute DSM target: sde.marginal_score(xt, x0, t)
        5. L_DSM = mean(||s_θ(xt, t) - target||²)
        6. [Optional Method A] L_phys = mean(||s_θ(xt, t) - s_φ(xt, t)||²)
           L_total = L_DSM + lambda_physics * L_phys

    The trainer never simulates paths internally. x0_data is always supplied
    externally (E-M paths, real market data, etc.).

    Args:
        sde:              Forward SDE providing noise and DSM targets
        score_network:    Network mapping [B, d+1] → [B, d]
        config:           Training hyperparameters
        device:           Computation device
        theory_score_fn:  Method A hook — s_φ(x, t) callable (default: None)
    """

    def __init__(
        self,
        sde: ForwardSDE,
        score_network: nn.Module,
        config: DSMConfig,
        device: torch.device,
        theory_score_fn: Optional[Callable] = None,
    ):
        self.sde = sde
        self.score_network = score_network.to(device)
        self.config = config
        self.device = device
        self.theory_score_fn = theory_score_fn

        self._use_physics = (
            theory_score_fn is not None and config.lambda_physics > 0.0
        )

        self.optimizer = torch.optim.AdamW(
            self.score_network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay_gamma,
        )

    def train(self, x0_data: torch.Tensor) -> Dict[str, List[float]]:
        """
        Train the score network on the provided data.

        Args:
            x0_data: Clean data samples [N, d]. For Interpretation 1 (point
                     mass initial condition), this is x₀ repeated N times.
                     For real market data, pass the actual observations.

        Returns:
            Dictionary with loss history:
                'dsm_loss':     DSM loss per epoch
                'physics_loss': Physics penalty per epoch (empty if disabled)
                'total_loss':   Total loss per epoch
        """
        cfg = self.config
        N = x0_data.shape[0]
        x0_data = x0_data.to(device=self.device, dtype=self.sde.dtype)

        history: Dict[str, List[float]] = {
            'dsm_loss': [],
            'physics_loss': [],
            'total_loss': [],
        }

        self.score_network.train()

        for epoch in range(1, cfg.n_epochs + 1):
            # --- Sample batch ---
            idx = torch.randint(0, N, (cfg.batch_size,), device=self.device)
            x0_batch = x0_data[idx]  # [B, d]

            # --- Sample diffusion time t ~ Uniform(t_eps, T) ---
            t_batch = (
                torch.rand(cfg.batch_size, 1, device=self.device, dtype=self.sde.dtype)
                * (cfg.T - cfg.t_eps)
                + cfg.t_eps
            )  # [B, 1]

            # --- Forward noise: sample xt ~ p(x(t) | x(0)) ---
            with torch.no_grad():
                xt = self.sde.sample_marginal(x0_batch, t_batch)  # [B, d]

            # --- Score network prediction ---
            inputs = torch.cat([xt, t_batch], dim=-1)  # [B, d+1]
            s_pred = self.score_network(inputs)          # [B, d]

            # --- DSM target ---
            target = self.sde.marginal_score(xt, x0_batch, t_batch)  # [B, d]

            # --- DSM loss ---
            dsm_loss = ((s_pred - target) ** 2).sum(dim=-1).mean()

            # --- Method A: physics penalty ---
            if self._use_physics:
                with torch.no_grad():
                    s_theory = self.theory_score_fn(xt, t_batch)  # [B, d]
                phys_loss = ((s_pred - s_theory) ** 2).sum(dim=-1).mean()
                total_loss = dsm_loss + cfg.lambda_physics * phys_loss
            else:
                phys_loss = torch.tensor(0.0, device=self.device)
                total_loss = dsm_loss

            # --- Optimise ---
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            history['dsm_loss'].append(dsm_loss.item())
            history['physics_loss'].append(phys_loss.item())
            history['total_loss'].append(total_loss.item())

            if epoch % cfg.log_every == 0:
                msg = f"Epoch {epoch}/{cfg.n_epochs}  DSM={dsm_loss.item():.4e}"
                if self._use_physics:
                    msg += f"  Phys={phys_loss.item():.4e}"
                msg += f"  LR={self.scheduler.get_last_lr()[0]:.2e}"
                logger.info(msg)

        self.score_network.eval()
        logger.info(
            "DSM training complete. Final DSM loss: %.4e", history['dsm_loss'][-1]
        )
        return history

    def predict_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate trained score network s_θ(x, t).

        Convenience wrapper so the trainer itself can be passed as a score_fn
        to ReverseDiffusionSampler.

        Args:
            x: State [B, d]
            t: Time  [B, 1]

        Returns:
            Score [B, d]
        """
        self.score_network.eval()
        with torch.no_grad():
            inputs = torch.cat([x, t], dim=-1)
            return self.score_network(inputs)
