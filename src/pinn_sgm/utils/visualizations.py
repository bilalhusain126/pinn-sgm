"""
Visualization utilities for PINN solutions and score functions.

This module provides plotting functions for analyzing PINN-solved PDEs,
score functions, and training dynamics.
"""

import logging
from typing import Optional, List, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

# Suppress matplotlib mathtext font substitution warnings
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_density_evolution(
    network: torch.nn.Module,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    time_points: Optional[List[float]] = None,
    analytical_solution: Optional[callable] = None,
    num_points: int = 200,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot probability density evolution over time.

    Args:
        network: Trained PINN network
        x_range: Spatial domain (x_min, x_max)
        time_points: List of time points to plot (default: [0.1, 0.5, 1.0])
        analytical_solution: Optional analytical solution function(x, t)
        num_points: Number of spatial points
        device: Computation device
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if time_points is None:
        time_points = [0.1, 0.5, 1.0]

    # Create spatial grid
    x = torch.linspace(x_range[0], x_range[1], num_points, device=device)
    x_np = x.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, len(time_points), figsize=figsize)
    if len(time_points) == 1:
        axes = [axes]

    network.eval()
    with torch.no_grad():
        for idx, t_val in enumerate(time_points):
            t = torch.full((num_points,), t_val, device=device)

            # PINN prediction
            p_pred = network(x.unsqueeze(-1), t.unsqueeze(-1)).squeeze().cpu().numpy()

            # Plot PINN solution
            axes[idx].plot(x_np, p_pred, 'b-', linewidth=2, label='PINN')

            # Plot analytical solution if available
            if analytical_solution is not None:
                p_true = analytical_solution(x.unsqueeze(-1), t.unsqueeze(-1)).squeeze().cpu().numpy()
                axes[idx].plot(x_np, p_true, 'r--', linewidth=2, label='Analytical', alpha=0.7)

            axes[idx].set_xlabel('$x$', fontsize=12)
            axes[idx].set_ylabel('$p(x, t)$', fontsize=12)
            axes[idx].set_title(f'$t = {t_val:.2f}$', fontsize=12)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_score_field(
    score_extractor,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    time_points: Optional[List[float]] = None,
    num_points: int = 200,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (14, 5),
    analytical_score: Optional[callable] = None,
    y_range: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plot score function s(x, t) = ∇_x log p(x, t) over time.

    Args:
        score_extractor: ScoreExtractor object
        x_range: Spatial domain (x_min, x_max)
        time_points: List of time points to plot
        num_points: Number of spatial points
        device: Computation device
        figsize: Figure size
        analytical_score: Optional function(x, t) -> score for comparison
        y_range: Optional y-axis limits (y_min, y_max)

    Returns:
        Matplotlib figure
    """
    if time_points is None:
        time_points = [0.1, 0.5, 1.0]

    # Create spatial grid
    x = torch.linspace(x_range[0], x_range[1], num_points, device=device)
    x_np = x.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, len(time_points), figsize=figsize)
    if len(time_points) == 1:
        axes = [axes]

    # Note: Score computation requires gradients, so no torch.no_grad()
    for idx, t_val in enumerate(time_points):
        t = torch.full((num_points,), t_val, device=device)
        x_input = x.unsqueeze(-1)
        t_input = t.unsqueeze(-1)

        # Compute PINN score
        scores_pinn = score_extractor(x_input, t_input).squeeze().detach().cpu().numpy()

        # Plot PINN score
        if analytical_score is not None:
            axes[idx].plot(x_np, scores_pinn, 'b-', linewidth=2, label='PINN')

            # Compute and plot analytical score
            with torch.no_grad():
                scores_analytical = analytical_score(x_input, t_input).squeeze().cpu().numpy()
            axes[idx].plot(x_np, scores_analytical, 'r--', linewidth=2, label='Analytical', alpha=0.7)
            axes[idx].legend(fontsize=10)
        else:
            axes[idx].plot(x_np, scores_pinn, 'g-', linewidth=2, label='PINN Score')

        axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[idx].set_xlabel('$x$', fontsize=12)
        axes[idx].set_ylabel(r'$s(x, t) = \nabla_x \log p(x, t)$')
        axes[idx].set_title(f'Score Field at $t = {t_val:.2f}$')
        axes[idx].grid(True, alpha=0.3)

        # Apply y-axis limits if specified
        if y_range is not None:
            axes[idx].set_ylim(y_range)

    plt.tight_layout()
    return fig


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot training history showing loss components over epochs.

    Args:
        history: Training history dictionary from PINNSolver
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    epochs = range(len(history['loss_total']))

    # Total loss
    axes[0, 0].plot(epochs, history['loss_total'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel(r'$\mathcal{L}_{\mathrm{total}}$', fontsize=12)
    axes[0, 0].set_title('Total Loss', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # PDE loss
    axes[0, 1].plot(epochs, history['loss_pde'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel(r'$\mathcal{L}_{\mathrm{PDE}}$', fontsize=12)
    axes[0, 1].set_title('PDE Residual Loss', fontsize=12)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # IC and BC losses
    axes[1, 0].plot(epochs, history['loss_ic'], 'g-', linewidth=1.5, label='IC')
    axes[1, 0].plot(epochs, history['loss_bc'], 'm-', linewidth=1.5, label='BC')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Initial & Boundary Condition Losses', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate (adaptive with ReduceLROnPlateau)
    axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_analysis(
    network: torch.nn.Module,
    analytical_solution: callable,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    t_range: Tuple[float, float] = (0.01, 1.0),
    num_x_points: int = 100,
    num_t_points: int = 50,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot absolute error heatmap comparing PINN solution to analytical solution.

    Args:
        network: Trained PINN network
        analytical_solution: Analytical solution function(x, t)
        x_range: Spatial domain
        t_range: Time domain
        num_x_points: Number of spatial points
        num_t_points: Number of time points
        device: Computation device
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create meshgrid
    x = torch.linspace(x_range[0], x_range[1], num_x_points, device=device)
    t = torch.linspace(t_range[0], t_range[1], num_t_points, device=device)
    X, T = torch.meshgrid(x, t, indexing='ij')

    # Flatten for evaluation
    x_flat = X.flatten().unsqueeze(-1)
    t_flat = T.flatten().unsqueeze(-1)

    # PINN prediction
    network.eval()
    with torch.no_grad():
        p_pred = network(x_flat, t_flat).squeeze()
        p_true = analytical_solution(x_flat, t_flat).squeeze()

    # Compute absolute error
    abs_error = torch.abs(p_pred - p_true)

    # Reshape
    abs_error = abs_error.reshape(num_x_points, num_t_points).cpu().numpy()
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Absolute error heatmap
    im = ax.contourf(T_np, X_np, abs_error, levels=50, cmap='viridis')
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$x$', fontsize=12)
    ax.set_title(r'Absolute Error: $|p_{\mathrm{PINN}} - p_{\mathrm{true}}|$', fontsize=12)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig
