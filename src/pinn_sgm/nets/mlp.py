"""
Base Multi-Layer Perceptron architecture for PINN solvers.

Implements a standard MLP with flexible configuration of layers and activations.
For the density-specific wrapper, see density_net.py.
"""

from typing import List, Literal, Optional
import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    """
    Factory function for activation functions.

    Args:
        name: Activation function name (case-insensitive)

    Returns:
        PyTorch activation module

    Raises:
        ValueError: If activation name is not recognized
    """
    name = name.lower()
    activations = {
        'silu': nn.SiLU(),
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
        'softplus': nn.Softplus(),
    }

    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(activations.keys())}"
        )

    return activations[name]


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for PINN applications.

    The network structure is:
        Input → [Linear → Activation] × N → Linear → Output

    The activation function must be smooth and sufficiently differentiable
    to support computation of high-order derivatives via automatic differentiation.

    Args:
        input_dim: Dimension of input (e.g., 2 for (x, t))
        output_dim: Dimension of output (e.g., 1 for scalar density p(x,t))
        hidden_dims: List of hidden layer dimensions (e.g., [64, 64, 64])
        activation: Activation function name (default: 'SiLU')
        output_activation: Optional activation for output layer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = 'silu',
        output_activation: Optional[str] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.output_activation_name = output_activation

        # Build network layers
        layers = []
        in_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(activation))
            in_dim = hidden_dim

        # Output layer (no activation by default)
        layers.append(nn.Linear(in_dim, output_dim))

        # Optional output activation
        if output_activation is not None:
            layers.append(_get_activation(output_activation))

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [Batch, input_dim]

        Returns:
            Output tensor of shape [Batch, output_dim]
        """
        return self.model(x)

    def __repr__(self) -> str:
        """String representation showing architecture."""
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        dims_str = " → ".join(map(str, dims))
        activation_str = f", activation={self.activation_name}"
        if self.output_activation_name:
            activation_str += f", output_activation={self.output_activation_name}"
        return f"MLP({dims_str}{activation_str})"
