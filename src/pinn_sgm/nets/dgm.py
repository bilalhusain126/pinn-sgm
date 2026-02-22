"""
Deep Galerkin Method (DGM) neural network architecture.

DGM uses LSTM-like gating mechanisms to improve gradient flow and expressiveness
for high-dimensional PDE problems. This architecture is particularly effective
for problems in 10+ dimensions.

Reference:
    Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm
    for solving partial differential equations. Journal of Computational Physics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLayer(nn.Module):
    """
    LSTM-like layer with gating mechanism for DGM.

    Uses update gate (Z), forget gate (G), reset gate (R), and candidate state (H)
    to control information flow through the network.

    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of hidden state
        trans1: Activation for gates ('tanh', 'relu', 'sigmoid')
        trans2: Activation for candidate state ('tanh', 'relu', 'sigmoid')
    """

    def __init__(self, input_dim: int, output_dim: int, trans1: str = "tanh", trans2: str = "tanh"):
        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # --- Activation functions ---
        self.trans1 = self._get_activation(trans1)
        self.trans2 = self._get_activation(trans2)

        # --- Input transformation weights (U matrices) ---
        self.Uz = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.Ug = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.Ur = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.Uh = nn.Parameter(torch.Tensor(input_dim, output_dim))

        # --- State transformation weights (W matrices) ---
        self.Wz = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.Wg = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.Wr = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.Wh = nn.Parameter(torch.Tensor(output_dim, output_dim))

        # --- Biases ---
        self.bz = nn.Parameter(torch.zeros(output_dim))
        self.bg = nn.Parameter(torch.zeros(output_dim))
        self.br = nn.Parameter(torch.zeros(output_dim))
        self.bh = nn.Parameter(torch.zeros(output_dim))

        # --- Initialize weights ---
        self.reset_parameters()

    def _get_activation(self, activation_type: str):
        """Get activation function by name."""
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
        }
        if activation_type not in activations:
            raise ValueError(f"Unknown activation: {activation_type}. Choose from {list(activations.keys())}")
        return activations[activation_type]

    def reset_parameters(self):
        """Initialize weights using Xavier uniform initialization."""
        for weight in [self.Uz, self.Ug, self.Ur, self.Uh, self.Wz, self.Wg, self.Wr, self.Wh]:
            nn.init.xavier_uniform_(weight)

    def forward(self, S: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM layer.

        Args:
            S: Hidden state from previous layer [Batch, output_dim]
            X: Input features [Batch, input_dim]

        Returns:
            Updated hidden state [Batch, output_dim]
        """
        # --- Gate computations ---
        Z = self.trans1(X @ self.Uz + S @ self.Wz + self.bz)  # Update gate
        G = self.trans1(X @ self.Ug + S @ self.Wg + self.bg)  # Forget gate
        R = self.trans1(X @ self.Ur + S @ self.Wr + self.br)  # Reset gate
        H = self.trans2(X @ self.Uh + (S * R) @ self.Wh + self.bh)  # Candidate

        # --- Update state: S_new = (1-G)⊙H + Z⊙S ---
        S_new = (1 - G) * H + Z * S
        return S_new


class DenseLayer(nn.Module):
    """
    Dense (fully-connected) layer with optional activation.

    Args:
        input_dim: Dimension of input features
        output_dim: Dimension of output features
        activation: Activation function ('tanh', 'relu', or None)
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = None):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # --- Linear transformation ---
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

        # --- Activation function ---
        self.activation = self._get_activation(activation)

        # --- Initialize weights ---
        self.reset_parameters()

    def _get_activation(self, activation_type: str):
        """Get activation function by name."""
        if activation_type is None:
            return None
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
        }
        if activation_type not in activations:
            raise ValueError(f"Unknown activation: {activation_type}. Choose from {list(activations.keys())}")
        return activations[activation_type]

    def reset_parameters(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dense layer.

        Args:
            X: Input tensor [Batch, input_dim]

        Returns:
            Output tensor [Batch, output_dim]
        """
        out = X @ self.W + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out


class DGM(nn.Module):
    """
    Deep Galerkin Method (DGM) neural network.

    DGM uses LSTM-like layers with gating mechanisms to solve high-dimensional PDEs.
    The architecture provides better gradient flow and expressiveness compared to
    standard MLPs, particularly effective for dimensions > 10.

    Architecture:
        Input → Dense(activation) → LSTM → LSTM → ... → Dense(output)

    Args:
        input_dim: Dimension of input features (e.g., spatial_dim + 1 for (x,t))
        output_dim: Dimension of output (e.g., spatial_dim for score vector)
        hidden_dims: List of hidden layer dimensions for LSTM layers (e.g., [50, 50, 50])
        activation: Activation function for initial layer ('tanh', 'relu')
        final_activation: Activation for output layer (None, 'tanh', etc.)

    Example:
        >>> # For 2D score function: input (x1, x2, t), output (s1, s2)
        >>> net = DGM(input_dim=3, output_dim=2, hidden_dims=[50, 50, 50])
        >>> x = torch.randn(100, 3)  # Batch of 100 samples
        >>> s = net(x)  # Output: [100, 2]
        >>>
        >>> # Variable width layers for better expressiveness
        >>> net = DGM(input_dim=3, output_dim=2, hidden_dims=[64, 128, 64])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        activation: str = "tanh",
        final_activation: str = None
    ):
        super(DGM, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)

        # --- Initial transformation: input → first hidden state ---
        self.initial_layer = DenseLayer(input_dim, hidden_dims[0], activation=activation)

        # --- LSTM layers with gating (possibly varying widths) ---
        self.lstm_layers = nn.ModuleList([
            LSTMLayer(input_dim, hidden_dims[i], trans1="tanh", trans2=activation)
            for i in range(self.n_layers)
        ])

        # --- Projection layers for dimension changes between LSTM layers ---
        self.projection_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) if hidden_dims[i] != hidden_dims[i + 1] else None
            for i in range(self.n_layers - 1)
        ])

        # --- Final transformation: last hidden state → output ---
        self.final_layer = DenseLayer(hidden_dims[-1], output_dim, activation=final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DGM network.

        Args:
            x: Input tensor [Batch, input_dim]
               For Score-PINN: concatenated [x_spatial, t]

        Returns:
            Output tensor [Batch, output_dim]
            For Score-PINN: score vector
        """
        # --- Initial transformation ---
        S = self.initial_layer(x)

        # --- LSTM forward pass with optional projections ---
        for i, lstm_layer in enumerate(self.lstm_layers):
            S = lstm_layer(S, x)
            if i < len(self.projection_layers) and self.projection_layers[i] is not None:
                S = self.projection_layers[i](S)

        # --- Output ---
        out = self.final_layer(S)
        return out

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DGM(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dims={self.hidden_dims})"
        )
