"""
Configuration dataclasses for PINN solvers and training.

This module defines configuration objects using Python dataclasses with
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

        if self.lr_patience <= 0:
            raise ValueError(f"lr_patience must be positive, got {self.lr_patience}")

        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")

        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError(f"gradient_clip_val must be positive or None, got {self.gradient_clip_val}")

        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")


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
        """Validate PINN configuration parameters."""
        if len(self.x_range) != 2:
            raise ValueError(f"x_range must be a tuple of (x_min, x_max), got {self.x_range}")

        if self.x_range[0] >= self.x_range[1]:
            raise ValueError(f"x_min must be < x_max, got x_range={self.x_range}")

        if len(self.t_range) != 2:
            raise ValueError(f"t_range must be a tuple of (t_min, t_max), got {self.t_range}")

        if self.t_range[0] < 0:
            raise ValueError(f"t_min must be non-negative, got t_range={self.t_range}")

        if self.t_range[0] >= self.t_range[1]:
            raise ValueError(f"t_min must be < t_max, got t_range={self.t_range}")

        if self.num_collocation <= 0:
            raise ValueError(f"num_collocation must be positive, got {self.num_collocation}")

        if self.num_boundary < 0:
            raise ValueError(f"num_boundary must be non-negative, got {self.num_boundary}")

        if self.num_initial <= 0:
            raise ValueError(f"num_initial must be positive, got {self.num_initial}")

        if self.num_integration <= 0:
            raise ValueError(f"num_integration must be positive, got {self.num_integration}")

        if self.num_mc_samples <= 0:
            raise ValueError(f"num_mc_samples must be positive, got {self.num_mc_samples}")

        if self.device not in ['cpu', 'cuda', 'mps']:
            # Also allow explicit device indices like 'cuda:0'
            if not (self.device.startswith('cuda:') or self.device.startswith('mps:')):
                raise ValueError(f"device must be 'cpu', 'cuda', 'mps', or explicit device, got {self.device}")

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
        """Validate Score-PINN configuration parameters."""
        if self.n_collocation <= 0:
            raise ValueError(f"n_collocation must be positive, got {self.n_collocation}")

        if self.n_initial <= 0:
            raise ValueError(f"n_initial must be positive, got {self.n_initial}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.lambda_initial < 0:
            raise ValueError(f"lambda_initial must be non-negative, got {self.lambda_initial}")

        if self.lambda_residual < 0:
            raise ValueError(f"lambda_residual must be non-negative, got {self.lambda_residual}")

        if len(self.x_range) != 2:
            raise ValueError(f"x_range must be a tuple of (x_min, x_max), got {self.x_range}")

        if self.x_range[0] >= self.x_range[1]:
            raise ValueError(f"x_range[0] must be < x_range[1], got x_range={self.x_range}")

        if len(self.t_range) != 2:
            raise ValueError(f"t_range must be a tuple of (t_min, t_max), got {self.t_range}")

        if self.t_range[0] < 0:
            raise ValueError(f"t_min must be non-negative, got t_range={self.t_range}")

        if self.t_range[0] >= self.t_range[1]:
            raise ValueError(f"t_range[0] must be < t_range[1], got t_range={self.t_range}")

        if self.method not in ['score_pinn', 'score_matching', 'sliced_score_matching']:
            raise ValueError(f"method must be 'score_pinn', 'score_matching', or 'sliced_score_matching', got {self.method}")

        if self.n_projections <= 0:
            raise ValueError(f"n_projections must be positive, got {self.n_projections}")

        if self.n_hte_samples <= 0:
            raise ValueError(f"n_hte_samples must be positive, got {self.n_hte_samples}")

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
