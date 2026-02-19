"""
Deep-dive analysis and diagnostics.

Functions for error analysis, score magnitude studies, and convergence diagnostics.
"""

import logging
from typing import Tuple, List
import torch
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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

    Currently supports 1D spatial + time (creates 2D heatmap).

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
        inputs = torch.cat([x_flat, t_flat], dim=-1)
        p_pred = network(inputs).squeeze()
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
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title(r'Absolute Error: $|p_{\mathrm{PINN}} - p_{\mathrm{true}}|$')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def plot_score_magnitude_analysis(
    solver,
    equation,
    eval_times: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0],
    n_samples: int = 1000,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Analyze score magnitude evolution and prediction errors over time.

    Plots:
    1. Mean score norm ||s(x,t)|| vs time (predicted vs analytical)
    2. Mean relative error vs time

    Args:
        solver: ScorePINNSolver object
        equation: Equation with analytical_score method
        eval_times: List of time points to evaluate
        n_samples: Number of random spatial samples
        x_range: Spatial domain for sampling (applied to all dimensions)
        device: Computation device
        figsize: Figure size

    Returns:
        Matplotlib figure with 2 subplots
    """
    spatial_dim = solver.spatial_dim

    # Sample random points in the domain
    x_random = torch.rand(n_samples, spatial_dim, device=device)
    x_random = x_random * (x_range[1] - x_range[0]) + x_range[0]  # Scale to [x_min, x_max]^d

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Collect score norms
    norms_pred = []
    norms_true = []
    rel_errors = []

    # Check if analytical solution is available
    t_test = torch.full((1, 1), eval_times[0], dtype=torch.float32, device=device)
    x_test = x_random[:1]
    has_analytical = equation.analytical_score(x_test, t_test) is not None

    if not has_analytical:
        logger.warning("No analytical solution available for validation. Plotting predicted score magnitudes only.")

    for t_val in eval_times:
        t_tensor = torch.full((n_samples, 1), t_val, dtype=torch.float32, device=device)

        with torch.no_grad():
            s_pred = solver.predict_score(x_random, t_tensor).cpu().numpy()
            s_true = equation.analytical_score(x_random, t_tensor)

        # Compute predicted norms
        norm_pred = np.linalg.norm(s_pred, axis=1)
        norms_pred.append(norm_pred.mean())

        if has_analytical:
            s_true = s_true.cpu().numpy()
            # Compute true norms
            norm_true = np.linalg.norm(s_true, axis=1)
            norms_true.append(norm_true.mean())

            # Compute relative error
            rel_error = np.mean(np.linalg.norm(s_pred - s_true, axis=1)) / (np.mean(np.linalg.norm(s_true, axis=1)) + 1e-10)
            rel_errors.append(rel_error)

    # Plot 1: Score norm evolution
    if has_analytical:
        axes[0].plot(eval_times, norms_true, 'bo-', label='Analytical', linewidth=2, markersize=8)
    axes[0].plot(eval_times, norms_pred, 'rs--', label='Predicted', linewidth=2, markersize=8)
    axes[0].set_xlabel('Time $t$')
    axes[0].set_ylabel(r'Mean $\|s(x,t)\|$')
    title_suffix = " (No Analytical Solution)" if not has_analytical else ""
    axes[0].set_title(f'Score Magnitude Evolution ({spatial_dim}D){title_suffix}')
    axes[0].legend()

    # Plot 2: Relative error over time (only if analytical solution available)
    if has_analytical:
        axes[1].plot(eval_times, np.array(rel_errors) * 100, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Time $t$')
        axes[1].set_ylabel('Mean Relative Error (%)')
        axes[1].set_title(f'Prediction Error Over Time ({spatial_dim}D)')
    else:
        axes[1].text(0.5, 0.5, 'No Analytical Solution\nAvailable for Comparison',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_xlabel('Time $t$')
        axes[1].set_title('Error Analysis (N/A)')

    plt.tight_layout()

    # Log summary
    logger.info("Score Magnitude Summary (%dD):", spatial_dim)
    if has_analytical:
        logger.info("  Mean analytical score norm: %.2f ± %.2f", np.mean(norms_true), np.std(norms_true))
    logger.info("  Mean predicted score norm: %.2f ± %.2f", np.mean(norms_pred), np.std(norms_pred))
    if has_analytical:
        logger.info("  Mean relative error: %.2f%%", np.mean(rel_errors) * 100)

    return fig
