"""
Deep-dive analysis and diagnostics.

Functions for error analysis and score quality evaluation.
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
    # --- Meshgrid ---
    x = torch.linspace(x_range[0], x_range[1], num_x_points, device=device)
    t = torch.linspace(t_range[0], t_range[1], num_t_points, device=device)
    X, T = torch.meshgrid(x, t, indexing='ij')

    # --- Flatten inputs ---
    x_flat = X.flatten().unsqueeze(-1)
    t_flat = T.flatten().unsqueeze(-1)

    # --- PINN prediction ---
    network.eval()
    with torch.no_grad():
        inputs = torch.cat([x_flat, t_flat], dim=-1)
        p_pred = network(inputs).squeeze()
        p_true = analytical_solution(x_flat, t_flat).squeeze()

    # --- Absolute error ---
    abs_error = torch.abs(p_pred - p_true)

    # --- Reshape for plotting ---
    abs_error = abs_error.reshape(num_x_points, num_t_points).cpu().numpy()
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    # --- Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- Heatmap ---
    im = ax.contourf(T_np, X_np, abs_error, levels=50, cmap='viridis')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title(r'Absolute Error: $|p_{\mathrm{PINN}} - p_{\mathrm{true}}|$')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig



def plot_score_mse(
    solver,
    equation,
    eval_times: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0],
    n_samples: int = 1000,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (7, 4)
) -> plt.Figure:
    """
    Plot mean score MSE E[||s_PINN(x,t) - s_true(x,t)||^2] over time.

    Samples x uniformly from x_range (applied to all dimensions). Requires the
    equation to expose analytical_score.

    Args:
        solver: Solver with predict_score(x, t) and spatial_dim attributes
        equation: Equation with analytical_score method
        eval_times: Time points to evaluate
        n_samples: Number of random spatial samples
        x_range: Spatial domain for sampling (applied to all dimensions)
        device: Computation device
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    spatial_dim = solver.spatial_dim
    mse_vals = []

    x_samples = torch.rand(n_samples, spatial_dim, device=device)
    x_samples = x_samples * (x_range[1] - x_range[0]) + x_range[0]

    for t_val in eval_times:
        t_input = torch.full((n_samples, 1), t_val, dtype=torch.float32, device=device)

        s_true = equation.analytical_score(x_samples, t_input)
        with torch.no_grad():
            s_pinn = solver.predict_score(x_samples, t_input)

        mse = ((s_pinn - s_true) ** 2).sum(dim=-1).mean().item()
        mse_vals.append(mse)
        logger.info("Score MSE at t=%.2f: %.6e", t_val, mse)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(eval_times, mse_vals, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel(r'Mean $\|\hat{s} - s_\mathrm{true}\|^2$')
    ax.set_title(f'Score MSE Over Time ({spatial_dim}D)')

    plt.tight_layout()
    logger.info(
        "Score MSE summary: min=%.4e, max=%.4e, mean=%.4e",
        min(mse_vals), max(mse_vals), sum(mse_vals) / len(mse_vals)
    )
    return fig
