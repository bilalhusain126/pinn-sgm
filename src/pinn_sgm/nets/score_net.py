"""
Score function network for score-based generative modeling.

This module provides ScoreNetwork, a specialized architecture for approximating
score functions in diffusion models and score matching objectives.
"""

from typing import List
import torch
import torch.nn as nn
from .mlp import MLP


class ScoreNetwork(nn.Module):
    """
    Neural network for score function approximation.

    This network directly approximates the score function:
        s(x, t) = ∇_x log p(x, t)

    Unlike DensityMLP which learns p(x,t), this network directly outputs
    the score vector, which can be more efficient for score matching objectives.

    Args:
        spatial_dim: Dimension of spatial variable x
        hidden_dims: List of hidden layer dimensions
        time_embedding_dim: Dimension for time encoding (0 to disable)
        activation: Activation function name
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        hidden_dims: List[int] = [64, 64, 64],
        time_embedding_dim: int = 16,
        activation: str = 'silu'
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.time_embedding_dim = time_embedding_dim

        # Input dimension: spatial + time embedding
        input_dim = spatial_dim
        if time_embedding_dim > 0:
            input_dim += time_embedding_dim

        # Build network
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=spatial_dim,  # Output score vector
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None  # No activation on score output
        )

        # Time embedding (sinusoidal, like in Transformers)
        if time_embedding_dim > 0:
            self.register_buffer(
                'time_frequencies',
                torch.exp(
                    torch.linspace(0, 4, time_embedding_dim // 2)
                ) * torch.pi
            )

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding.

        Args:
            t: Time values of shape [Batch] or [Batch, 1]

        Returns:
            Embedded time of shape [Batch, time_embedding_dim]
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        # Compute sinusoidal features
        t_expanded = t.unsqueeze(-1) * self.time_frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(t_expanded), torch.cos(t_expanded)], dim=-1)

        return embedding

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute score function s(x, t) = ∇_x log p(x, t).

        Args:
            x: Spatial coordinates of shape [Batch, spatial_dim]
            t: Time values of shape [Batch] or [Batch, 1]

        Returns:
            Score vectors of shape [Batch, spatial_dim]
        """
        # Embed time if needed
        if self.time_embedding_dim > 0:
            t_embed = self._embed_time(t)
            inputs = torch.cat([x, t_embed], dim=-1)
        else:
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            inputs = torch.cat([x, t], dim=-1)

        # Forward pass
        score = self.mlp(inputs)

        return score

    def __repr__(self) -> str:
        """String representation."""
        return f"ScoreNetwork(spatial_dim={self.spatial_dim}, time_embedding_dim={self.time_embedding_dim})"
