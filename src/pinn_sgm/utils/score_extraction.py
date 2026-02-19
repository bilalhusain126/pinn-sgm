"""
Score function extraction from PINN-solved density fields.

This module implements the extraction of theoretical score functions from
trained PINN solutions of the Fokker-Planck equation, enabling integration
with score-based generative models.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import numpy as np

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
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        epsilon: float = 1e-8,
        spatial_dim: int = None
    ):
        self.network = network
        self.device = device
        self.dtype = dtype
        self.epsilon = epsilon

        # Infer spatial_dim from network if not provided
        if spatial_dim is None:
            if hasattr(network, 'spatial_dim'):
                self.spatial_dim = network.spatial_dim
            else:
                raise ValueError(
                    "spatial_dim must be provided if the network does not expose a spatial_dim attribute."
                )
        else:
            self.spatial_dim = spatial_dim

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
            x: Spatial coordinates [Batch, spatial_dim]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Score values [Batch, spatial_dim]
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
            s(x, t) = ∇_x log p(x, t) = (1/p) ∇_x p

        Supports both 1D and multi-dimensional spatial inputs.

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Score [Batch, spatial_dim]
        """
        # Enable gradients even if called within torch.no_grad() context
        with torch.enable_grad():
            # Clone and detach first to handle tensors created with no_grad()
            x = x.detach().clone().requires_grad_(True)
            t = t.detach().clone()  # t doesn't need gradients for score

            # Forward pass: compute density (concatenated input)
            inputs = torch.cat([x, t], dim=-1)
            p = self.network(inputs)  # [Batch, 1]

            # Compute gradient ∇_x p(x, t)
            grad_p = torch.autograd.grad(
                outputs=p,
                inputs=x,
                grad_outputs=torch.ones_like(p),
                create_graph=True,
                retain_graph=True
            )[0]  # [Batch, spatial_dim]

            # Score: s = (∇p) / p
            # Add epsilon to denominator for numerical stability
            score = grad_p / (p + self.epsilon)

        return score

    def predict_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Alias for compute_score() for compatibility with visualization functions.

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Score [Batch, spatial_dim]
        """
        return self.compute_score(x, t)

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

        # Compute scores (requires gradients, so no torch.no_grad())
        scores = self.compute_score(x_input, t_input).squeeze()

        return x_grid.cpu().numpy(), scores.detach().cpu().numpy()

