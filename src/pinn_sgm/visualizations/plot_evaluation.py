"""
Prediction evaluation and comparison with analytical solutions.

Functions for visualizing density and score predictions across dimensions.
"""

import logging
from typing import Optional, List, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_density_evolution(
    network: torch.nn.Module,
    spatial_dim: int,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    time_points: Optional[List[float]] = None,
    analytical_solution: Optional[callable] = None,
    dim: int = 0,
    fixed_values: Optional[torch.Tensor] = None,
    num_points: int = 200,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot probability density evolution over time.

    For multidimensional problems, plots 1D slice along specified dimension.

    Args:
        network: Trained PINN network
        spatial_dim: Spatial dimensionality
        x_range: Range for the varying dimension (x_min, x_max)
        time_points: List of time points to plot (default: [0.1, 0.5, 1.0])
        analytical_solution: Optional analytical solution function(x, t)
        dim: Which spatial dimension to vary (0 to spatial_dim-1)
        fixed_values: Values for other dimensions (default: all zeros)
        num_points: Number of points along varying dimension
        device: Computation device
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if time_points is None:
        time_points = [0.1, 0.5, 1.0]

    if dim < 0 or dim >= spatial_dim:
        raise ValueError(f"dim must be in range [0, {spatial_dim-1}], got {dim}")

    # --- Spatial grid ---
    x_vals = torch.linspace(x_range[0], x_range[1], num_points, device=device)

    # --- Fixed values for other dimensions ---
    if fixed_values is None:
        fixed_values = torch.zeros(spatial_dim, device=device)
    elif fixed_values.shape[0] != spatial_dim:
        raise ValueError(f"fixed_values must have shape ({spatial_dim},), got {fixed_values.shape}")

    # --- Figure ---
    fig, axes = plt.subplots(1, len(time_points), figsize=figsize)
    if len(time_points) == 1:
        axes = [axes]

    network.eval()
    with torch.no_grad():
        for idx, t_val in enumerate(time_points):
            # --- Input preparation ---
            x_input = fixed_values.unsqueeze(0).repeat(num_points, 1)  # [num_points, spatial_dim]
            x_input[:, dim] = x_vals

            t_input = torch.full((num_points, 1), t_val, device=device)

            # --- PINN prediction ---
            inputs = torch.cat([x_input, t_input], dim=-1)
            p_pred = network(inputs).squeeze().cpu().numpy()

            # --- Plot PINN ---
            x_plot = x_vals.cpu().numpy()
            axes[idx].plot(x_plot, p_pred, 'b-', linewidth=2, label='PINN')

            # --- Analytical solution ---
            if analytical_solution is not None:
                p_true = analytical_solution(x_input, t_input).squeeze().cpu().numpy()
                axes[idx].plot(x_plot, p_true, 'r--', linewidth=2, label='Analytical', alpha=0.7)

            # --- Labels ---
            if spatial_dim == 1:
                x_label = '$x$'
            else:
                x_label = f'$x_{{{dim+1}}}$'

            axes[idx].set_xlabel(x_label)
            axes[idx].set_ylabel('$p(x, t)$')
            axes[idx].set_title(f'$t = {t_val:.2f}$')
            axes[idx].legend()

    # --- Multidimensional title ---
    if spatial_dim > 1:
        other_dims = [f'$x_{{{i+1}}}={fixed_values[i].item():.1f}$'
                      for i in range(spatial_dim) if i != dim]
        fig.suptitle(f'Density Evolution (varying $x_{{{dim+1}}}$, fixed: {", ".join(other_dims)})',
                     fontsize=14, y=1.02)

    plt.tight_layout()
    return fig


def plot_score_field(
    solver,
    equation,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    time_points: Optional[List[float]] = None,
    dim: int = 0,
    fixed_values: Optional[torch.Tensor] = None,
    num_points: int = 200,
    device: torch.device = torch.device('cpu'),
    figsize: Tuple[int, int] = (14, 5),
    y_range: Optional[Tuple[float, float]] = None
) -> plt.Figure:
    """
    Plot score function s(x, t) = ∇_x log p(x, t) over time.

    For multidimensional problems, plots 1D slice along specified dimension,
    showing the score component corresponding to that dimension.

    Args:
        solver: ScorePINNSolver object
        equation: Equation with analytical_score method
        x_range: Range for the varying dimension (x_min, x_max)
        time_points: List of time points to plot
        dim: Which spatial dimension to vary and plot (0 to spatial_dim-1)
        fixed_values: Values for other dimensions (default: all zeros)
        num_points: Number of points along varying dimension
        device: Computation device
        figsize: Figure size
        y_range: Optional y-axis limits (y_min, y_max)

    Returns:
        Matplotlib figure
    """
    if time_points is None:
        time_points = [0.1, 0.5, 1.0]

    spatial_dim = solver.spatial_dim

    if dim < 0 or dim >= spatial_dim:
        raise ValueError(f"dim must be in range [0, {spatial_dim-1}], got {dim}")

    # --- Spatial grid ---
    x_vals = torch.linspace(x_range[0], x_range[1], num_points, device=device)

    # --- Fixed values for other dimensions ---
    if fixed_values is None:
        fixed_values = torch.zeros(spatial_dim, device=device)
    elif fixed_values.shape[0] != spatial_dim:
        raise ValueError(f"fixed_values must have shape ({spatial_dim},), got {fixed_values.shape}")

    # --- Figure ---
    fig, axes = plt.subplots(1, len(time_points), figsize=figsize)
    if len(time_points) == 1:
        axes = [axes]

    for idx, t_val in enumerate(time_points):
        # --- Input preparation ---
        x_input = fixed_values.unsqueeze(0).repeat(num_points, 1)  # [num_points, spatial_dim]
        x_input[:, dim] = x_vals

        t_input = torch.full((num_points, 1), t_val, device=device)

        # --- Compute score ---
        with torch.no_grad():
            s_pred = solver.predict_score(x_input, t_input).cpu().numpy()  # [num_points, spatial_dim]
            s_true = equation.analytical_score(x_input, t_input)  # [num_points, spatial_dim] or None

        # --- Extract dimension component ---
        s_pred_dim = s_pred[:, dim]

        # --- Plot ---
        x_plot = x_vals.cpu().numpy()

        # --- Analytical solution ---
        if s_true is not None:
            s_true_dim = s_true.cpu().numpy()[:, dim]
            axes[idx].plot(x_plot, s_true_dim, 'b-', linewidth=2, label='Analytical')

        axes[idx].plot(x_plot, s_pred_dim, 'r--', linewidth=2, label='Predicted', alpha=0.8)

        # --- Labels ---
        if spatial_dim == 1:
            x_label = '$x$'
            y_label = r'$s(x, t)$'
        else:
            x_label = f'$x_{{{dim+1}}}$'
            y_label = f'$s_{{{dim+1}}}(x, t)$'

        axes[idx].set_xlabel(x_label)
        axes[idx].set_ylabel(y_label)
        axes[idx].set_title(f'$t = {t_val:.2f}$')
        axes[idx].legend(fontsize=10)

        # --- Y-axis limits ---
        if y_range is not None:
            axes[idx].set_ylim(y_range)

        # --- Error annotation ---
        if s_true is not None:
            abs_error = np.abs(s_pred_dim - s_true_dim)
            rel_error = np.mean(abs_error) / (np.mean(np.abs(s_true_dim)) + 1e-10)
            max_error = np.max(abs_error)

            axes[idx].text(0.05, 0.95, f'Mean rel: {rel_error*100:.2f}%\nMax abs: {max_error:.3f}',
                           transform=axes[idx].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75), fontsize=9)

    # --- Multidimensional title ---
    if spatial_dim > 1:
        other_dims = [f'$x_{{{i+1}}}={fixed_values[i].item():.1f}$'
                      for i in range(spatial_dim) if i != dim]
        fig.suptitle(f'Score Component {dim+1} (varying $x_{{{dim+1}}}$, fixed: {", ".join(other_dims)})',
                     fontsize=14, y=1.02)

    plt.tight_layout()
    return fig
