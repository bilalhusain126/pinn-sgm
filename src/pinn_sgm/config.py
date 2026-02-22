"""
Configuration dataclasses for PINN solvers, score models, and training.

This module defines all configuration objects using Python dataclasses with
post-initialization validation to ensure parameter consistency.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
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
        weight_decay: L2 penalty coefficient for AdamW optimizer (0.0 disables)
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
    weight_decay: float = 1e-4
    gradient_clip_val: Optional[float] = 1.0
    verbose: bool = True
    log_interval: int = 100

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if not 0 < self.lr_decay_rate <= 1:
            raise ValueError(f"lr_decay_rate must be in (0, 1], got {self.lr_decay_rate}")

        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")

        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError(f"gradient_clip_val must be positive or None, got {self.gradient_clip_val}")


@dataclass
class PINNConfig:
    """
    Configuration for Physics-Informed Neural Network PDE solver.

    Attributes:
        x_range: Spatial domain (x_min, x_max)
        t_range: Time domain (t_min, t_max)
        num_collocation: Number of interior collocation points
        num_boundary: Number of boundary collocation points (if applicable)
        num_initial: Number of initial condition points
        num_integration: Number of quadrature points for 1D normalization integral
        num_mc_samples: Number of Monte Carlo samples for multi-D normalization integral
        device: Computation device ('cpu' or 'cuda')
        dtype: Tensor data type
        enforce_normalization: If True, enforce ∫p(x,t)dx = 1
        pde_weight: Weight λ_PDE for PDE residual loss term
        ic_weight: Weight λ_IC for initial condition loss term
        bc_weight: Weight λ_BC for boundary condition loss term
        normalization_weight: Weight λ_norm for normalization loss term
    """
    x_range: Tuple[float, float] = (-5.0, 5.0)
    t_range: Tuple[float, float] = (0.0, 1.0)
    num_collocation: int = 10000
    num_boundary: int = 1000
    num_initial: int = 1000
    num_integration: int = 500
    num_mc_samples: int = 5000
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    enforce_normalization: bool = True
    pde_weight: float = 1.0
    ic_weight: float = 1.0
    bc_weight: float = 1.0
    normalization_weight: float = 1.0

    @property
    def x_min(self) -> float:
        """Minimum spatial coordinate."""
        return self.x_range[0]

    @property
    def x_max(self) -> float:
        """Maximum spatial coordinate."""
        return self.x_range[1]

    @property
    def t_min(self) -> float:
        """Minimum time."""
        return self.t_range[0]

    @property
    def t_max(self) -> float:
        """Maximum time."""
        return self.t_range[1]

    @property
    def T(self) -> float:
        """Terminal time (alias for t_max for backward compatibility)."""
        return self.t_range[1]

    def __post_init__(self):
        if self.x_range[0] >= self.x_range[1]:
            raise ValueError(f"x_min must be < x_max, got x_range={self.x_range}")

        if self.t_range[0] < 0:
            raise ValueError(f"t_min must be non-negative, got t_range={self.t_range}")

        if self.t_range[0] >= self.t_range[1]:
            raise ValueError(f"t_min must be < t_max, got t_range={self.t_range}")

        if self.pde_weight < 0:
            raise ValueError(f"pde_weight must be non-negative, got {self.pde_weight}")

        if self.ic_weight < 0:
            raise ValueError(f"ic_weight must be non-negative, got {self.ic_weight}")

        if self.bc_weight < 0:
            raise ValueError(f"bc_weight must be non-negative, got {self.bc_weight}")

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
class ScorePINNConfig:
    """
    Configuration for Score-PINN solver.

    Score-PINN learns the score function s(x,t) = ∇ log p(x,t) by solving
    the Score PDE using physics-informed neural networks.

    Attributes:
        n_collocation: Number of collocation points for PDE residual
        n_initial: Number of points for initial condition
        batch_size: Batch size for residual computation
        lambda_initial: Weight for initial condition loss
        lambda_residual: Weight for PDE residual loss
        x_range: Spatial domain range (x_min, x_max)
        t_range: Time domain range (t_min, t_max)
        method: Score learning method ('score_pinn', 'score_matching', 'sliced_score_matching')
        n_projections: Number of projection directions for sliced score matching
        use_hte: Whether to use Hutchinson Trace Estimation for divergence computation
        n_hte_samples: Number of random samples for HTE (1 is usually sufficient)
        t_epsilon: Regularization time for singular initial conditions (e.g., Dirac delta)
    """
    n_collocation: int = 10000
    n_initial: int = 1000
    batch_size: int = 256
    lambda_initial: float = 1.0
    lambda_residual: float = 1.0
    x_range: Tuple[float, float] = (-5.0, 5.0)
    t_range: Tuple[float, float] = (0.0, 1.0)
    method: Literal['score_pinn', 'score_matching', 'sliced_score_matching'] = 'score_pinn'
    n_projections: int = 1
    use_hte: bool = True
    n_hte_samples: int = 1
    t_epsilon: float = 0.1

    def __post_init__(self):
        if self.lambda_initial < 0:
            raise ValueError(f"lambda_initial must be non-negative, got {self.lambda_initial}")

        if self.lambda_residual < 0:
            raise ValueError(f"lambda_residual must be non-negative, got {self.lambda_residual}")

        if self.x_range[0] >= self.x_range[1]:
            raise ValueError(f"x_range[0] must be < x_range[1], got x_range={self.x_range}")

        if self.t_range[0] < 0:
            raise ValueError(f"t_min must be non-negative, got t_range={self.t_range}")

        if self.t_range[0] >= self.t_range[1]:
            raise ValueError(f"t_range[0] must be < t_range[1], got t_range={self.t_range}")

        if self.t_epsilon <= 0:
            raise ValueError(f"t_epsilon must be positive, got {self.t_epsilon}")

    @property
    def x_min(self) -> float:
        """Minimum spatial domain boundary."""
        return self.x_range[0]

    @property
    def x_max(self) -> float:
        """Maximum spatial domain boundary."""
        return self.x_range[1]

    @property
    def t_min(self) -> float:
        """Minimum time."""
        return self.t_range[0]

    @property
    def t_max(self) -> float:
        """Maximum time."""
        return self.t_range[1]


@dataclass
class DSMConfig:
    """
    Configuration for Denoising Score Matching (DSM) training.

    Implements the continuous-time DSM objective (Song et al. 2021, Eq. 7).
    The trainer is equation-agnostic: it accepts any ForwardSDE and any
    score network, and takes external data (x0_data) rather than simulating
    paths internally.

    Attributes:
        batch_size:      Training batch size (drawn from x0_data each epoch)
        n_epochs:        Number of training epochs
        lr:              Adam learning rate
        weight_decay:    Adam weight decay
        lr_decay_step:   StepLR step size (epochs)
        lr_decay_gamma:  StepLR multiplicative decay factor
        T:               Terminal diffusion time
        t_eps:           Minimum time to sample (avoids t≈0 score singularity)
        lambda_physics:  Method A: weight on physics penalty (0 = disabled)
        log_every:       Log loss every this many epochs
    """
    batch_size: int = 512
    n_epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_decay_step: int = 500
    lr_decay_gamma: float = 0.5
    T: float = 1.0
    t_eps: float = 0.01
    lambda_physics: float = 0.0
    log_every: int = 100
