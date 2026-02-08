"""
Score function extraction from PINN-solved density fields.

This module implements the extraction of theoretical score functions from
trained PINN solutions of the Fokker-Planck equation, enabling integration
with score-based generative models.

References:
    - PhD Research Document: Section 3.4 (Score Extraction for Diffusion Models)
    - PhD Research Document: Equation (3.8)
    - PhD Research Document: Equation (2.20) (Hybrid Score Field)
"""

import logging
from typing import Optional, Callable, Union
import torch
import torch.nn as nn
import numpy as np

from ..config import ScoreModelConfig

logger = logging.getLogger(__name__)


class ScoreExtractor:
    """
    Extract score function ∇_x log p(x, t) from trained PINN density network.

    The score function is the gradient of the log-density with respect to
    the spatial variable. Given a trained network p_θ(x, t), we compute:

        s_theory(x, t) = ∇_x log p(x, t) ≈ ∇_x p_θ(x, t) / p_θ(x, t)

    This is computed using automatic differentiation, providing exact gradients.

    Args:
        network: Trained PINN network approximating p(x, t)
        device: Computation device
        dtype: Tensor data type
        epsilon: Small constant to prevent division by zero

    References:
        - PhD Research Document: Equation (3.8)
        - PhD Research Document: Section 2.3.1 (The Score Function)
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        epsilon: float = 1e-8
    ):
        self.network = network
        self.device = device
        self.dtype = dtype
        self.epsilon = epsilon

        # Set network to evaluation mode
        self.network.eval()
        self.network.to(device=device, dtype=dtype)

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute theoretical score s_theory(x, t) = ∇_x log p(x, t).

        Args:
            x: Spatial coordinates [Batch, 1], requires_grad=True
            t: Temporal coordinates [Batch, 1]

        Returns:
            Score values [Batch, 1]
        """
        return self.compute_score(x, t)

    def compute_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score function using automatic differentiation.

        The score is computed as:
            s(x, t) = ∂/∂x log p(x, t) = (1/p) ∂p/∂x

        Args:
            x: Spatial coordinates [Batch, 1]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Score [Batch, 1]
        """
        # Ensure gradients are enabled
        x = x.requires_grad_(True)

        # Forward pass: compute density
        p = self.network(x, t)  # [Batch, 1]

        # Compute gradient ∇_x p(x, t)
        grad_p = torch.autograd.grad(
            outputs=p,
            inputs=x,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True
        )[0]  # [Batch, 1]

        # Score: s = (∇p) / p
        # Add epsilon to denominator for numerical stability
        score = grad_p / (p + self.epsilon)

        return score

    def compute_score_safe(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        min_density: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute score with additional safety checks for numerical stability.

        In regions where p(x, t) is very small, the score can become numerically
        unstable. This method clips the density to a minimum value.

        Args:
            x: Spatial coordinates [Batch, 1]
            t: Temporal coordinates [Batch, 1]
            min_density: Minimum density value for clipping

        Returns:
            Score [Batch, 1]
        """
        # Ensure gradients are enabled
        x = x.requires_grad_(True)

        # Forward pass
        p = self.network(x, t)

        # Clip density to minimum value
        p_clipped = torch.clamp(p, min=min_density)

        # Compute gradient
        grad_p = torch.autograd.grad(
            outputs=p_clipped,
            inputs=x,
            grad_outputs=torch.ones_like(p_clipped),
            create_graph=True,
            retain_graph=True
        )[0]

        # Score
        score = grad_p / p_clipped

        return score

    def compute_score_log_derivative(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative score computation using log-derivative trick.

        Computes s = ∇_x log p(x, t) directly by taking gradient of log(p).
        This can be more numerically stable in some cases.

        Args:
            x: Spatial coordinates [Batch, 1]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Score [Batch, 1]
        """
        # Ensure gradients are enabled
        x = x.requires_grad_(True)

        # Forward pass
        p = self.network(x, t)

        # Compute log-density (add epsilon for stability)
        log_p = torch.log(p + self.epsilon)

        # Gradient of log-density
        score = torch.autograd.grad(
            outputs=log_p,
            inputs=x,
            grad_outputs=torch.ones_like(log_p),
            create_graph=True,
            retain_graph=True
        )[0]

        return score

    def evaluate_on_grid(
        self,
        x_min: float = -5.0,
        x_max: float = 5.0,
        t_value: float = 1.0,
        num_points: int = 100
    ) -> tuple:
        """
        Evaluate score function on a spatial grid at fixed time.

        Useful for visualization and analysis.

        Args:
            x_min: Minimum spatial coordinate
            x_max: Maximum spatial coordinate
            t_value: Time value
            num_points: Number of grid points

        Returns:
            x_grid: Spatial coordinates [num_points]
            scores: Score values [num_points]
        """
        # Create grid
        x_grid = torch.linspace(x_min, x_max, num_points, device=self.device, dtype=self.dtype)
        t_grid = torch.full((num_points,), t_value, device=self.device, dtype=self.dtype)

        # Reshape for network
        x_input = x_grid.unsqueeze(-1)
        t_input = t_grid.unsqueeze(-1)

        # Compute scores
        with torch.no_grad():
            scores = self.compute_score(x_input, t_input).squeeze()

        return x_grid.cpu().numpy(), scores.cpu().numpy()


def hybrid_score(
    x: torch.Tensor,
    t: torch.Tensor,
    score_empirical: Union[torch.Tensor, Callable],
    score_theoretical: Union[torch.Tensor, Callable],
    config: ScoreModelConfig,
    T: float = 1.0
) -> torch.Tensor:
    """
    Compute hybrid score field combining empirical and theoretical scores.

    The hybrid score is defined as:
        ŝ(x, t) = (1 - φ_t) s_θ(x, t) + φ_t s_theory(x, t)

    where:
        - s_θ(x, t) is the empirical score learned from data
        - s_theory(x, t) is the theoretical score from PINN
        - φ_t ∈ [0, 1] is a time-dependent weight

    The weight φ_t controls the trade-off between:
        - Empirical flexibility (captures market idiosyncrasies)
        - Theoretical consistency (enforces structural constraints)

    Args:
        x: Spatial coordinates [Batch, 1]
        t: Time values [Batch, 1] or [Batch]
        score_empirical: Empirical score s_θ(x, t) or function computing it
        score_theoretical: Theoretical score s_theory(x, t) or function computing it
        config: Score model configuration
        T: Terminal time for normalization

    Returns:
        Hybrid score ŝ(x, t) [Batch, 1]

    References:
        - PhD Research Document: Equation (2.20)
        - PhD Research Document: Section 2.3.4 (Modified Langevin Corrector)

    Examples:
        >>> config = ScoreModelConfig(phi_start=0.0, phi_end=1.0, interpolation='linear')
        >>> # At t=0 (high data density): φ_0 = 0, use empirical score
        >>> # At t=T (low data density): φ_T = 1, use theoretical score
        >>> hybrid_s = hybrid_score(x, t, s_empirical, s_theoretical, config, T=1.0)
    """
    # Ensure proper tensor shape
    if t.dim() == 2:
        t = t.squeeze(-1)

    # Compute time-dependent weights
    t_normalized = t / T
    phi_t = torch.zeros_like(t)

    for i, t_val in enumerate(t_normalized):
        phi_t[i] = config.get_phi(t_val.item(), T=1.0)

    # Reshape phi_t for broadcasting
    phi_t = phi_t.unsqueeze(-1)  # [Batch, 1]

    # Compute scores (handle both tensor and callable inputs)
    if callable(score_empirical):
        s_emp = score_empirical(x, t)
    else:
        s_emp = score_empirical

    if callable(score_theoretical):
        s_th = score_theoretical(x, t)
    else:
        s_th = score_theoretical

    # Hybrid score: (1 - φ_t) s_empirical + φ_t s_theoretical
    s_hybrid = (1 - phi_t) * s_emp + phi_t * s_th

    return s_hybrid


class LangevinCorrector:
    """
    Modified Langevin dynamics corrector using hybrid scores.

    Implements the corrector step in the Predictor-Corrector framework
    for sampling from diffusion models, enhanced with theoretical scores.

    The Langevin corrector step is:
        x_{i+1} = x_i + ε ŝ(x_i, t) + √(2ε) z_i,    z_i ~ N(0, I)

    where ŝ is the hybrid score combining empirical and theoretical components.

    Args:
        score_network: Empirical score network s_θ(x, t)
        score_extractor: Theoretical score extractor s_theory(x, t)
        config: Score model configuration
        step_size: Langevin step size ε

    References:
        - PhD Research Document: Equation (2.19), (2.20)
        - Song et al. (2021): "Score-Based Generative Modeling through SDEs"
    """

    def __init__(
        self,
        score_network: nn.Module,
        score_extractor: ScoreExtractor,
        config: ScoreModelConfig,
        step_size: float = 1e-3
    ):
        self.score_network = score_network
        self.score_extractor = score_extractor
        self.config = config
        self.step_size = step_size

    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        T: float = 1.0
    ) -> torch.Tensor:
        """
        Perform one Langevin corrector step.

        Args:
            x: Current samples [Batch, dim]
            t: Current time [Batch, 1] or scalar
            T: Terminal time

        Returns:
            Updated samples [Batch, dim]
        """
        # Ensure proper shape
        if isinstance(t, (int, float)):
            t = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        # Compute empirical score
        s_empirical = self.score_network(x, t)

        # Compute theoretical score
        s_theoretical = self.score_extractor(x, t)

        # Compute hybrid score
        s_hybrid = hybrid_score(
            x, t,
            s_empirical,
            s_theoretical,
            self.config,
            T
        )

        # Langevin dynamics step
        noise = torch.randn_like(x)
        x_next = x + self.step_size * s_hybrid + np.sqrt(2 * self.step_size) * noise

        return x_next

    def __call__(self, x: torch.Tensor, t: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """Alias for step method."""
        return self.step(x, t, T)
