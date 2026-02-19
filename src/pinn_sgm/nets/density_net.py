"""
Probability density network for vanilla PINN solvers.

This module provides DensityNetwork, a wrapper that ensures any base network
outputs positive probability densities by applying a positivity constraint.
"""

from typing import Optional
import torch
import torch.nn as nn


class DensityNetwork(nn.Module):
    """
    Wrapper that ensures any base network outputs positive probability densities.

    This network enforces non-negativity by applying a Softplus activation
    at the output layer:
        p(x, t) = log(1 + exp(z))
    where z is the output from the base network. This ensures p(x, t) > 0
    everywhere, a critical requirement for probability densities.

    The base network should:
    - Accept concatenated input [x_1, ..., x_spatial_dim, t] of shape [Batch, spatial_dim + 1]
    - Output scalar values of shape [Batch, 1]

    Args:
        base_network: The underlying neural network (MLP, DGM, or custom)
        use_softplus: If True, apply Softplus for positivity; if False, use exp (default: True)
        spatial_dim: Dimension of spatial variable x (optional, for metadata)

    Example:
        >>> from pinn_sgm.nets import MLP, DGM, DensityNetwork
        >>>
        >>> # With MLP
        >>> base = MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64])
        >>> network = DensityNetwork(base, spatial_dim=2)
        >>>
        >>> # With DGM for high dimensions
        >>> base = DGM(input_dim=10, output_dim=1, hidden_dims=[50, 50, 50])
        >>> network = DensityNetwork(base, spatial_dim=9)
    """

    def __init__(
        self,
        base_network: nn.Module,
        use_softplus: bool = True,
        spatial_dim: Optional[int] = None
    ):
        super().__init__()

        self.base_network = base_network
        self.use_softplus = use_softplus
        self.spatial_dim = spatial_dim

        # Output activation for positivity
        if use_softplus:
            self.output_activation = nn.Softplus()
        else:
            # Use exp as a function, not a module
            self.output_activation = lambda x: torch.exp(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute probability density p(x, t).

        Args:
            inputs: Concatenated spatial and temporal coordinates [Batch, spatial_dim + 1]
                   Format: [x_1, ..., x_spatial_dim, t]

        Returns:
            Density values of shape [Batch, 1], guaranteed positive
        """
        # Forward through base network
        z = self.base_network(inputs)  # [Batch, 1]

        # Apply output activation for positivity
        p = self.output_activation(z)  # [Batch, 1]

        return p

    def __repr__(self) -> str:
        """String representation showing architecture."""
        output_str = "Softplus" if self.use_softplus else "Exp"
        spatial_str = f", spatial_dim={self.spatial_dim}" if self.spatial_dim is not None else ""
        return f"DensityNetwork(base={self.base_network.__class__.__name__}, output={output_str}{spatial_str})"
