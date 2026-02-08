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

            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('p(x, t)')
            axes[idx].set_title(f't = {t_val:.2f}')
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
    figsize: Tuple[int, int] = (14, 5)
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

    with torch.no_grad():
        for idx, t_val in enumerate(time_points):
            t = torch.full((num_points,), t_val, device=device)

            # Compute score
            scores = score_extractor(x.unsqueeze(-1), t.unsqueeze(-1)).squeeze().cpu().numpy()

            # Plot score function
            axes[idx].plot(x_np, scores, 'g-', linewidth=2)
            axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)

            axes[idx].set_xlabel('x')
            axes[idx].set_ylabel('s(x, t) = ∇_x log p(x, t)')
            axes[idx].set_title(f'Score Field at t = {t_val:.2f}')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_density_heatmap(
    network: torch.nn.Module,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    num_x_points: int = 200,
    num_t_points: int = 100,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot 2D heatmap of density p(x, t) over space and time.

    Args:
        network: Trained PINN network
        x_range: Spatial domain (x_min, x_max)
        t_range: Time domain (t_min, t_max)
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

    # Flatten for network evaluation
    x_flat = X.flatten().unsqueeze(-1)
    t_flat = T.flatten().unsqueeze(-1)

    # Evaluate network
    network.eval()
    with torch.no_grad():
        p_flat = network(x_flat, t_flat)

    # Reshape to grid
    P = p_flat.squeeze().reshape(num_x_points, num_t_points).cpu().numpy()
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Heatmap
    im = ax.contourf(T_np, X_np, P, levels=50, cmap='viridis')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_title('Probability Density Evolution p(x, t)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density')

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
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # PDE loss
    axes[0, 1].plot(epochs, history['loss_pde'], 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PDE Residual Loss')
    axes[0, 1].set_title('PDE Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # IC and BC losses
    axes[1, 0].plot(epochs, history['loss_ic'], 'g-', linewidth=1.5, label='IC Loss')
    axes[1, 0].plot(epochs, history['loss_bc'], 'm-', linewidth=1.5, label='BC Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Initial & Boundary Condition Losses')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
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
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot error analysis comparing PINN solution to analytical solution.

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

    # Compute errors
    abs_error = torch.abs(p_pred - p_true)
    rel_error = abs_error / (torch.abs(p_true) + 1e-8)

    # Reshape
    abs_error = abs_error.reshape(num_x_points, num_t_points).cpu().numpy()
    rel_error = rel_error.reshape(num_x_points, num_t_points).cpu().numpy()
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Absolute error
    im1 = axes[0].contourf(T_np, X_np, abs_error, levels=50, cmap='Reds')
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Space (x)')
    axes[0].set_title('Absolute Error |p_PINN - p_true|')
    plt.colorbar(im1, ax=axes[0])

    # Relative error
    im2 = axes[1].contourf(T_np, X_np, rel_error, levels=50, cmap='Reds')
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Space (x)')
    axes[1].set_title('Relative Error |p_PINN - p_true| / |p_true|')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    return fig


def plot_3d_surface(
    network: torch.nn.Module,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    t_range: Tuple[float, float] = (0.0, 1.0),
    num_x_points: int = 100,
    num_t_points: int = 50,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot 3D surface of density p(x, t).

    Args:
        network: Trained PINN network
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

    # Evaluate network
    network.eval()
    with torch.no_grad():
        p_flat = network(x_flat, t_flat)

    # Reshape
    P = p_flat.squeeze().reshape(num_x_points, num_t_points).cpu().numpy()
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    # Create 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(T_np, X_np, P, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space (x)')
    ax.set_zlabel('Density p(x, t)')
    ax.set_title('3D Probability Density Evolution')

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    return fig


def plot_score_comparison(
    score_extractor,
    analytical_score: Optional[callable] = None,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    t_value: float = 0.5,
    num_points: int = 200,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Compare extracted score with analytical score (if available).

    Args:
        score_extractor: ScoreExtractor object
        analytical_score: Optional analytical score function(x, t)
        x_range: Spatial domain
        t_value: Time value
        num_points: Number of points
        device: Computation device
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create grid
    x = torch.linspace(x_range[0], x_range[1], num_points, device=device)
    t = torch.full((num_points,), t_value, device=device)
    x_np = x.cpu().numpy()

    # Compute extracted score
    with torch.no_grad():
        scores_extracted = score_extractor(x.unsqueeze(-1), t.unsqueeze(-1)).squeeze().cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot extracted score
    ax.plot(x_np, scores_extracted, 'b-', linewidth=2, label='Extracted (PINN)')

    # Plot analytical score if available
    if analytical_score is not None:
        scores_true = analytical_score(x.unsqueeze(-1), t.unsqueeze(-1)).squeeze().cpu().numpy()
        ax.plot(x_np, scores_true, 'r--', linewidth=2, label='Analytical', alpha=0.7)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('s(x, t) = ∇_x log p(x, t)')
    ax.set_title(f'Score Function at t = {t_value:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
