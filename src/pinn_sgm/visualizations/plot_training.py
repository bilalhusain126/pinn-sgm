"""
Training diagnostics and loss visualizations.

Functions for monitoring training progress, loss components, and gradient norms.
"""

import logging
from typing import Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot training history showing loss components over epochs.

    Auto-detects solver type and adapts plot layout:
    - Vanilla PINN: Shows PDE, IC, BC losses
    - Score-PINN: Shows Initial, Residual losses

    Args:
        history: Training history dictionary from PINNSolver or ScorePINNSolver
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    epochs = range(len(history['loss_total']))

    # Detect solver type based on keys
    is_score_pinn = 'loss_initial' in history and 'loss_residual' in history
    is_vanilla_pinn = 'loss_pde' in history and 'loss_ic' in history

    # Total loss (always present)
    axes[0, 0].plot(epochs, history['loss_total'], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel(r'$\mathcal{L}_{\mathrm{total}}$')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_yscale('log')

    if is_score_pinn:
        # Score-PINN: Initial condition loss
        axes[0, 1].plot(epochs, history['loss_initial'], 'orange', linewidth=1.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel(r'$\mathcal{L}_{\mathrm{initial}}$')
        axes[0, 1].set_title('Initial Condition Loss')
        axes[0, 1].set_yscale('log')

        # Score-PINN: Residual loss
        axes[1, 0].plot(epochs, history['loss_residual'], 'green', linewidth=1.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel(r'$\mathcal{L}_{\mathrm{residual}}$')
        axes[1, 0].set_title('PDE Residual Loss')
        axes[1, 0].set_yscale('log')

    elif is_vanilla_pinn:
        # Vanilla PINN: PDE loss
        axes[0, 1].plot(epochs, history['loss_pde'], 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel(r'$\mathcal{L}_{\mathrm{PDE}}$')
        axes[0, 1].set_title('PDE Residual Loss')
        axes[0, 1].set_yscale('log')

        # Vanilla PINN: IC and BC losses
        axes[1, 0].plot(epochs, history['loss_ic'], 'g-', linewidth=1.5, label='IC')
        axes[1, 0].plot(epochs, history['loss_bc'], 'm-', linewidth=1.5, label='BC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Initial & Boundary Condition Losses')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()

    else:
        # Fallback: plot whatever losses are available
        logger.warning("Could not auto-detect solver type. Plotting available losses.")
        loss_keys = [k for k in history.keys() if k.startswith('loss_') and k != 'loss_total']
        for idx, key in enumerate(loss_keys[:2]):  # Plot up to 2 additional losses
            ax = axes[0, 1] if idx == 0 else axes[1, 0]
            ax.plot(epochs, history[key], linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(key.replace('_', ' ').title())
            ax.set_yscale('log')

    # Gradient norm (always present)
    if 'grad_norm' in history:
        axes[1, 1].plot(epochs, history['grad_norm'], 'purple', linewidth=1.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel(r'$\|\nabla \mathcal{L}\|$')
        axes[1, 1].set_title('Gradient Norm')
        axes[1, 1].set_yscale('log')
    else:
        # Fallback: show learning rate if gradient norm not available
        if 'learning_rate' in history:
            axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=1.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_yscale('log')

    plt.tight_layout()
    return fig
