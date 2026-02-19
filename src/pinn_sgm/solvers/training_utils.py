"""
Shared utility functions for training PINN solvers.

Provides common optimizer setup, scheduler configuration, and gradient
norm computation to avoid duplication across solver classes.
"""

from typing import Iterable
import torch
import torch.nn as nn
from ..config import TrainingConfig


def setup_optimizer(
    parameters: Iterable[nn.Parameter],
    training_config: TrainingConfig
) -> torch.optim.Optimizer:
    """
    Create optimizer from training configuration.

    Args:
        parameters: Model parameters to optimize
        training_config: Training configuration

    Returns:
        Configured optimizer

    Raises:
        ValueError: If optimizer type is not supported
    """
    if training_config.optimizer == 'adam':
        return torch.optim.AdamW(
            parameters,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
    elif training_config.optimizer == 'lbfgs':
        return torch.optim.LBFGS(
            parameters,
            lr=training_config.learning_rate,
            max_iter=20
        )
    else:
        raise ValueError(f"Unknown optimizer: {training_config.optimizer}")


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler from training configuration.

    Args:
        optimizer: Optimizer to attach scheduler to
        training_config: Training configuration

    Returns:
        Configured scheduler

    Raises:
        ValueError: If scheduler type is not supported
    """
    if training_config.lr_scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.lr_decay_step,
            gamma=training_config.lr_decay_rate
        )
    elif training_config.lr_scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_config.lr_decay_rate,
            patience=training_config.lr_patience
        )
    else:
        raise ValueError(f"Unknown lr_scheduler: {training_config.lr_scheduler}")


def compute_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """
    Compute L2 norm of gradients across all parameters.

    Should be called after backward() but before gradient clipping.

    Args:
        parameters: Model parameters with gradients

    Returns:
        L2 norm of all gradients
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    return total_norm ** 0.5
