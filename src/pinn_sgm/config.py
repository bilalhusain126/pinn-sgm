"""
Configuration dataclasses for PINN solvers and training.

This module defines configuration objects using Python dataclasses with
post-initialization validation to ensure parameter consistency.

References:
"""

from dataclasses import dataclass
from typing import Literal, Optional
import torch


@dataclass
class TrainingConfig:
    """
    Configuration for neural network training.

    Attributes:
        batch_size: Number of collocation points per training batch
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate for optimizer
        lr_scheduler: LR scheduler type ('step' or 'plateau')
        lr_decay_step: Number of epochs between LR decay (for 'step' scheduler)
        lr_decay_rate: Multiplicative factor for LR decay (gamma)
        lr_patience: Epochs to wait before reducing LR (for 'plateau' scheduler)
        optimizer: Optimizer type ('adam' or 'lbfgs')
        gradient_clip_val: Maximum gradient norm (None to disable clipping)
        verbose: If True, print training progress
        log_interval: Number of epochs between progress logs
    """
    batch_size: int = 1024
    epochs: int = 10000
    learning_rate: float = 1e-3
    lr_scheduler: Literal['step', 'plateau'] = 'step'
    lr_decay_step: int = 1000
    lr_decay_rate: float = 0.9
    lr_patience: int = 500
    optimizer: Literal['adam', 'lbfgs'] = 'adam'
    gradient_clip_val: Optional[float] = 1.0
    verbose: bool = True
    log_interval: int = 100

    def __post_init__(self):
        """Validate training configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.lr_decay_step <= 0:
            raise ValueError(f"lr_decay_step must be positive, got {self.lr_decay_step}")

        if not 0 < self.lr_decay_rate <= 1:
            raise ValueError(f"lr_decay_rate must be in (0, 1], got {self.lr_decay_rate}")

        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError(f"gradient_clip_val must be positive or None, got {self.gradient_clip_val}")

        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")


@dataclass
class PINNConfig:
    """
    Configuration for Physics-Informed Neural Network PDE solver.

    Attributes:
        T: Terminal time for PDE solution
        x_min: Minimum spatial domain boundary
        x_max: Maximum spatial domain boundary
        num_collocation: Number of interior collocation points
        num_boundary: Number of boundary collocation points (if applicable)
        num_initial: Number of initial condition points
        device: Computation device ('cpu' or 'cuda')
        dtype: Tensor data type
        enforce_normalization: If True, enforce ∫p(x,t)dx = 1
        normalization_weight: Weight for normalization loss term
    """
    T: float = 1.0
    x_min: float = -5.0
    x_max: float = 5.0
    num_collocation: int = 10000
    num_boundary: int = 1000
    num_initial: int = 1000
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    enforce_normalization: bool = True
    normalization_weight: float = 1.0

    def __post_init__(self):
        """Validate PINN configuration parameters."""
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")

        if self.x_min >= self.x_max:
            raise ValueError(f"x_min must be < x_max, got x_min={self.x_min}, x_max={self.x_max}")

        if self.num_collocation <= 0:
            raise ValueError(f"num_collocation must be positive, got {self.num_collocation}")

        if self.num_boundary < 0:
            raise ValueError(f"num_boundary must be non-negative, got {self.num_boundary}")

        if self.num_initial <= 0:
            raise ValueError(f"num_initial must be positive, got {self.num_initial}")

        if self.device not in ['cpu', 'cuda', 'mps']:
            # Also allow explicit device indices like 'cuda:0'
            if not (self.device.startswith('cuda:') or self.device.startswith('mps:')):
                raise ValueError(f"device must be 'cpu', 'cuda', 'mps', or explicit device, got {self.device}")

        if self.normalization_weight < 0:
            raise ValueError(f"normalization_weight must be non-negative, got {self.normalization_weight}")

    @property
    def torch_device(self) -> torch.device:
        """Convert device string to torch.device object."""
        return torch.device(self.device)

    @property
    def domain_size(self) -> float:
        """Compute spatial domain size."""
        return self.x_max - self.x_min


@dataclass
class MertonModelConfig:
    """
    Configuration for Merton structural credit risk model.

    The Merton model assumes firm asset value V_t follows Geometric Brownian Motion:
        dV_t = μ V_t dt + σ V_t dW_t

    In log-space X_t = ln(V_t), this becomes:
        dX_t = (μ - σ²/2) dt + σ dW_t

    Attributes:
        mu: Asset drift (expected return)
        sigma: Asset volatility
        x0: Initial log-asset value X_0 = ln(V_0)
    """
    mu: float = 0.05
    sigma: float = 0.2
    x0: float = 0.0

    def __post_init__(self):
        """Validate Merton model parameters."""
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    @property
    def alpha(self) -> float:
        """
        Effective drift in log-space: α = μ - σ²/2

        This is the drift term in the Fokker-Planck equation after Itô's lemma.
        """
        return self.mu - 0.5 * self.sigma ** 2

    @property
    def diffusion_coeff(self) -> float:
        """Diffusion coefficient (same in both V and X space)."""
        return self.sigma


@dataclass
class ScoreModelConfig:
    """
    Configuration for score-based generative model integration.

    This controls how the theoretical score (from PINN) is blended with
    the empirical score (from data) during the diffusion generation process.

    The hybrid score is: ŝ(x,t) = (1 - φ_t) s_θ(x,t) + φ_t s_theory(x,t)

    Attributes:
        phi_start: Initial weight φ_0 for theoretical score at t=0
        phi_end: Final weight φ_T for theoretical score at t=T
        interpolation: Method for φ_t interpolation ('linear', 'exponential', 'sigmoid')
    """
    phi_start: float = 0.0
    phi_end: float = 1.0
    interpolation: Literal['linear', 'exponential', 'sigmoid'] = 'linear'

    def __post_init__(self):
        """Validate score model configuration."""
        if not 0 <= self.phi_start <= 1:
            raise ValueError(f"phi_start must be in [0, 1], got {self.phi_start}")

        if not 0 <= self.phi_end <= 1:
            raise ValueError(f"phi_end must be in [0, 1], got {self.phi_end}")

    def get_phi(self, t: float, T: float) -> float:
        """
        Compute time-dependent weight φ_t ∈ [0, 1].

        Args:
            t: Current time
            T: Terminal time

        Returns:
            Weight for theoretical score at time t
        """
        normalized_t = t / T

        if self.interpolation == 'linear':
            return self.phi_start + (self.phi_end - self.phi_start) * normalized_t

        elif self.interpolation == 'exponential':
            import numpy as np
            alpha = np.log(self.phi_end / max(self.phi_start, 1e-8))
            return self.phi_start * np.exp(alpha * normalized_t)

        elif self.interpolation == 'sigmoid':
            import numpy as np
            # Sigmoid centered at t/T = 0.5
            z = 10 * (normalized_t - 0.5)  # Steepness parameter = 10
            sigmoid = 1 / (1 + np.exp(-z))
            return self.phi_start + (self.phi_end - self.phi_start) * sigmoid

        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")
