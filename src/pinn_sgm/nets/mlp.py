"""
Multi-Layer Perceptron architectures for PINN solvers.

Implements standard MLP and specialized DensityMLP for probability density approximation.
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

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")

        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"All hidden dimensions must be positive, got {hidden_dims}")

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


class DensityMLP(nn.Module):
    """
    Specialized MLP for probability density approximation in Fokker-Planck PINNs.

    This network enforces non-negativity by applying a Softplus activation
    at the output layer:
        p(x, t) = log(1 + exp(z))
    where z is the pre-activation output. This ensures p(x, t) > 0 everywhere,
    a critical requirement for probability densities.

    Supports multi-dimensional spatial inputs for multi-factor models:
        - 1D: Merton model (single asset)
        - 2D: Heston model (asset + volatility), two-asset portfolios
        - ND: Multi-asset portfolios, multi-factor interest rate models

    Args:
        spatial_dim: Dimension of spatial variable x (default: 1)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function for hidden layers (default: 'tanh')
        use_softplus: If True, apply Softplus; if False, use exp (default: True)
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        hidden_dims: List[int] = [64, 64, 64],
        activation: str = 'tanh',
        use_softplus: bool = True
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.input_dim = spatial_dim + 1  # x (d-dim) + t (1-dim)
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.use_softplus = use_softplus

        # Build hidden layers
        layers = []
        in_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(activation))
            in_dim = hidden_dim

        # Output layer: scalar density value (pre-activation)
        layers.append(nn.Linear(in_dim, 1))

        self.hidden_network = nn.Sequential(*layers)

        # Output activation for positivity
        if use_softplus:
            self.output_activation = nn.Softplus()
        else:
            self.output_activation = torch.exp

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute probability density p(x, t).

        Args:
            x: Spatial coordinates of shape [Batch, spatial_dim] or [Batch] (if spatial_dim=1)
            t: Temporal coordinates of shape [Batch, 1] or [Batch]

        Returns:
            Density values of shape [Batch, 1], guaranteed positive
        """
        # Ensure proper shapes
        if self.spatial_dim == 1:
            # For 1D, accept both [Batch] and [Batch, 1]
            if x.dim() == 1:
                x = x.unsqueeze(-1)
        else:
            # For multi-D, x should already be [Batch, spatial_dim]
            if x.dim() == 1:
                raise ValueError(
                    f"For spatial_dim={self.spatial_dim}, x must be [Batch, {self.spatial_dim}], "
                    f"got shape {x.shape}"
                )

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Concatenate inputs: [x, t]
        inputs = torch.cat([x, t], dim=-1)  # [Batch, spatial_dim + 1]

        # Forward through hidden network
        z = self.hidden_network(inputs)  # [Batch, 1]

        # Apply output activation for positivity
        p = self.output_activation(z)  # [Batch, 1]

        return p

    def __repr__(self) -> str:
        """String representation showing architecture."""
        dims = [self.input_dim] + self.hidden_dims + [1]
        dims_str = " → ".join(map(str, dims))
        activation_str = f", activation={self.activation_name}"
        output_str = "Softplus" if self.use_softplus else "Exp"
        return f"DensityMLP({dims_str}{activation_str}, output={output_str}, spatial_dim={self.spatial_dim})"


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
