"""
Physics-Informed Neural Network (PINN) solver for PDEs.

This module implements the core PINN algorithm using automatic differentiation
to enforce PDE constraints during neural network training.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..equations.base import BasePDE
from ..nets.mlp import DensityMLP
from ..config import PINNConfig, TrainingConfig

logger = logging.getLogger(__name__)


class PINNSolver:
    """
    Physics-Informed Neural Network solver for PDEs.

    This solver uses a neural network to approximate the solution u(x, t)
    and trains it by minimizing the PDE residual, initial condition error,
    and boundary condition error at collocation points.

    The total loss is:
        L = L_PDE + L_IC + L_BC + L_norm

    where:
        - L_PDE: Mean squared PDE residual in domain interior
        - L_IC: Mean squared error on initial condition
        - L_BC: Mean squared error on boundary conditions
        - L_norm: Normalization constraint (for probability densities)

    Args:
        equation: PDE equation to solve
        network: Neural network approximating u(x, t)
        pinn_config: PINN configuration
        training_config: Training configuration
    """

    def __init__(
        self,
        equation: BasePDE,
        network: Optional[nn.Module] = None,
        pinn_config: Optional[PINNConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        # Use default configs if not provided
        self.pinn_config = pinn_config if pinn_config is not None else PINNConfig()
        self.training_config = training_config if training_config is not None else TrainingConfig()

        # PDE equation
        self.equation = equation

        # Neural network
        if network is None:
            # Default: DensityMLP with 3 hidden layers of 64 neurons
            self.network = DensityMLP(
                input_dim=2,  # (x, t)
                hidden_dims=[64, 64, 64],
                activation='tanh'
            )
        else:
            self.network = network

        # Move to device
        self.device = self.pinn_config.torch_device
        self.dtype = self.pinn_config.dtype
        self.network = self.network.to(device=self.device, dtype=self.dtype)

        # Optimizer
        if self.training_config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.training_config.learning_rate
            )
        elif self.training_config.optimizer == 'lbfgs':
            self.optimizer = optim.LBFGS(
                self.network.parameters(),
                lr=self.training_config.learning_rate,
                max_iter=20
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")

        # Learning rate scheduler
        if self.training_config.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.lr_decay_step,
                gamma=self.training_config.lr_decay_rate
            )
        elif self.training_config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.training_config.lr_decay_rate,
                patience=self.training_config.lr_patience
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.training_config.lr_scheduler}")

        # Training history
        self.history = {
            'loss_total': [],
            'loss_pde': [],
            'loss_ic': [],
            'loss_bc': [],
            'loss_norm': [],
            'learning_rate': []
        }

    def _compute_derivatives(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute solution and its derivatives using automatic differentiation.

        This is the core innovation of PINNs: we can compute exact derivatives
        of the neural network output with respect to inputs using the chain rule.

        Args:
            x: Spatial coordinates [Batch, 1], requires_grad=True
            t: Temporal coordinates [Batch, 1], requires_grad=True

        Returns:
            u: Solution u(x, t) [Batch, 1]
            u_t: Time derivative ∂u/∂t [Batch, 1]
            u_x: Spatial derivative ∂u/∂x [Batch, 1]
            u_xx: Second spatial derivative ∂²u/∂x² [Batch, 1]
        """
        # Forward pass through network
        u = self.network(x, t)  # [Batch, 1]

        # First-order derivatives using autograd
        grad_outputs = torch.ones_like(u)

        u_grads = torch.autograd.grad(
            outputs=u,
            inputs=[x, t],
            grad_outputs=grad_outputs,
            create_graph=True,  # Allow second derivatives
            retain_graph=True
        )
        u_x = u_grads[0]  # ∂u/∂x [Batch, 1]
        u_t = u_grads[1]  # ∂u/∂t [Batch, 1]

        # Second-order spatial derivative
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0]  # ∂²u/∂x² [Batch, 1]

        return u, u_t, u_x, u_xx

    def _sample_collocation_points(self) -> Dict[str, torch.Tensor]:
        """
        Sample collocation points for PDE, initial, and boundary conditions.

        Returns:
            Dictionary with keys:
                - 'x_pde', 't_pde': Interior points for PDE residual
                - 'x_ic': Points for initial condition
                - 'x_bc', 't_bc': Boundary points (if applicable)
        """
        cfg = self.pinn_config

        # Interior points 
        x_pde = torch.rand(cfg.num_collocation, 1, device=self.device, dtype=self.dtype)
        x_pde = cfg.x_min + (cfg.x_max - cfg.x_min) * x_pde
        t_pde = cfg.T * torch.rand(cfg.num_collocation, 1, device=self.device, dtype=self.dtype)

        # Initial condition points (t=0)
        x_ic = torch.rand(cfg.num_initial, 1, device=self.device, dtype=self.dtype)
        x_ic = cfg.x_min + (cfg.x_max - cfg.x_min) * x_ic

        # Boundary points (x at boundaries, t random)
        if cfg.num_boundary > 0:
            t_bc = cfg.T * torch.rand(cfg.num_boundary, 1, device=self.device, dtype=self.dtype)
            x_bc_left = torch.full((cfg.num_boundary // 2, 1), cfg.x_min, device=self.device, dtype=self.dtype)
            x_bc_right = torch.full((cfg.num_boundary - cfg.num_boundary // 2, 1), cfg.x_max, device=self.device, dtype=self.dtype)
            x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
        else:
            x_bc = torch.empty(0, 1, device=self.device, dtype=self.dtype)
            t_bc = torch.empty(0, 1, device=self.device, dtype=self.dtype)

        return {
            'x_pde': x_pde,
            't_pde': t_pde,
            'x_ic': x_ic,
            'x_bc': x_bc,
            't_bc': t_bc
        }

    def _compute_pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual loss: L_PDE = (1/N) Σ |F[u]|²

        Args:
            x: Spatial coordinates [Batch, 1]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Mean squared PDE residual
        """
        # Enable gradient computation
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        # Compute derivatives
        u, u_t, u_x, u_xx = self._compute_derivatives(x, t)

        # Compute PDE residual
        residual = self.equation.pde_residual(x, t, u, u_t, u_x, u_xx)

        # Mean squared error
        loss = torch.mean(residual ** 2)

        return loss

    def _compute_ic_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss: L_IC = (1/N) Σ |u(x,0) - u_0(x)|²

        Args:
            x: Spatial coordinates [Batch, 1]

        Returns:
            Mean squared initial condition error
        """
        t_zero = torch.zeros_like(x)

        # Network prediction at t=0
        u_pred = self.network(x, t_zero)

        # True initial condition
        u_true = self.equation.initial_condition(x)

        # Mean squared error
        loss = torch.mean((u_pred - u_true) ** 2)

        return loss

    def _compute_bc_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss: L_BC = (1/N) Σ |B[u] - g|²

        Args:
            x: Boundary coordinates [Batch, 1]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Mean squared boundary condition error
        """
        if x.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Network prediction at boundary
        u_pred = self.network(x, t)

        # Boundary condition residual
        residual = self.equation.boundary_condition(x, t, u_pred)

        # Mean squared error
        loss = torch.mean(residual ** 2)

        return loss

    def _compute_normalization_loss(self, t: torch.Tensor) -> torch.Tensor:
        """
        Enforce normalization constraint: ∫ p(x, t) dx = 1

        This is critical for probability densities. We approximate the integral
        using the trapezoidal rule on a fine grid.

        Args:
            t: Time value (scalar or single-element tensor)

        Returns:
            Normalization violation: (∫p dx - 1)²
        """
        if not self.pinn_config.enforce_normalization:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Create integration grid
        num_integration_points = 500
        x_grid = torch.linspace(
            self.pinn_config.x_min,
            self.pinn_config.x_max,
            num_integration_points,
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(-1)  # [N, 1]

        # Time broadcast
        if isinstance(t, float):
            t_val = t
        else:
            t_val = t.item() if t.numel() == 1 else t.mean().item()

        t_grid = torch.full_like(x_grid, t_val)

        # Evaluate network on grid
        with torch.no_grad():
            p_values = self.network(x_grid, t_grid).squeeze()  # [N]

        # Trapezoidal integration
        dx = (self.pinn_config.x_max - self.pinn_config.x_min) / (num_integration_points - 1)
        integral = torch.trapezoid(p_values, dx=dx)

        # Normalization constraint
        loss = (integral - 1.0) ** 2

        return loss

    def _compute_total_loss(self, collocation_points: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute total loss combining all terms.

        Args:
            collocation_points: Dictionary of sampled points

        Returns:
            Total loss value
        """
        # PDE loss
        loss_pde = self._compute_pde_loss(
            collocation_points['x_pde'],
            collocation_points['t_pde']
        )

        # Initial condition loss
        loss_ic = self._compute_ic_loss(collocation_points['x_ic'])

        # Boundary condition loss
        loss_bc = self._compute_bc_loss(
            collocation_points['x_bc'],
            collocation_points['t_bc']
        )

        # Normalization loss (at a random time)
        t_norm = self.pinn_config.T * torch.rand(1, device=self.device, dtype=self.dtype)
        loss_norm = self._compute_normalization_loss(t_norm)

        # Total loss (weighted sum)
        loss_total = (
            loss_pde +
            loss_ic +
            loss_bc +
            self.pinn_config.normalization_weight * loss_norm
        )

        # Store in history
        self.history['loss_total'].append(loss_total.item())
        self.history['loss_pde'].append(loss_pde.item())
        self.history['loss_ic'].append(loss_ic.item())
        self.history['loss_bc'].append(loss_bc.item())
        self.history['loss_norm'].append(loss_norm.item())
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

        return loss_total

    def train(self) -> Dict[str, Any]:
        """
        Train the PINN to solve the PDE.

        Returns:
            Dictionary containing:
                - 'network': Trained network
                - 'history': Training history
                - 'final_loss': Final loss value
        """
        self.network.train()

        # Training header
        start_time = time.time()
        if self.training_config.verbose:
            print(f"\n{'Epoch':<12} {'Loss':<12} {'PDE':<12} {'IC':<12} {'BC':<12} {'Time':<12}")
            print("-" * 72)

        for epoch in range(self.training_config.epochs):
            # Sample new collocation points each epoch
            collocation_points = self._sample_collocation_points()

            # Optimizer step
            self.optimizer.zero_grad()

            # Compute loss
            loss = self._compute_total_loss(collocation_points)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.training_config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.training_config.gradient_clip_val
                )

            # Optimizer step
            self.optimizer.step()

            # Learning rate decay
            if self.training_config.lr_scheduler == 'plateau':
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step()

            # Periodic logging
            if epoch % self.training_config.log_interval == 0 and self.training_config.verbose:
                elapsed = time.time() - start_time
                it_per_sec = (epoch + 1) / elapsed if elapsed > 0 else 0
                epoch_str = f"{epoch}/{self.training_config.epochs}"
                time_str = f"{it_per_sec:.2f}it/s"
                print(
                    f"{epoch_str:<12} "
                    f"{loss.item():<12.4e} "
                    f"{self.history['loss_pde'][-1]:<12.4e} "
                    f"{self.history['loss_ic'][-1]:<12.4e} "
                    f"{self.history['loss_bc'][-1]:<12.4e} "
                    f"{time_str:<12}"
                )

        # Final summary
        if self.training_config.verbose:
            total_time = time.time() - start_time
            print("-" * 72)
            final_loss = self.history['loss_total'][-1]
            final_pde = self.history['loss_pde'][-1]
            final_ic = self.history['loss_ic'][-1]
            final_bc = self.history['loss_bc'][-1]
            print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)")
            print(f"Final: Total={final_loss:.4e}, PDE={final_pde:.4e}, IC={final_ic:.4e}, BC={final_bc:.4e}\n")

        return {
            'network': self.network,
            'history': self.history,
            'final_loss': self.history['loss_total'][-1]
        }

    def predict(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate trained network at given points.

        Args:
            x: Spatial coordinates [N, 1] or [N]
            t: Temporal coordinates [N, 1] or [N]

        Returns:
            Solution values u(x, t) [N, 1]
        """
        self.network.eval()

        with torch.no_grad():
            # Ensure proper shape
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            if t.dim() == 1:
                t = t.unsqueeze(-1)

            # Convert to proper device/dtype
            x = x.to(device=self.device, dtype=self.dtype)
            t = t.to(device=self.device, dtype=self.dtype)

            # Forward pass
            u = self.network(x, t)

        return u

    def evaluate_error(
        self,
        num_test_points: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate solution error against analytical solution (if available).

        Args:
            num_test_points: Number of test points

        Returns:
            Dictionary of error metrics
        """
        # Sample test points
        x_test = torch.linspace(
            self.pinn_config.x_min,
            self.pinn_config.x_max,
            num_test_points,
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(-1)

        t_test = self.pinn_config.T * torch.rand(num_test_points, 1, device=self.device, dtype=self.dtype)

        # PINN prediction
        u_pred = self.predict(x_test, t_test)

        # Analytical solution
        u_true = self.equation.analytical_solution(x_test, t_test)

        if u_true is None:
            logger.warning("No analytical solution available for error computation")
            return {}

        # Compute absolute errors 
        abs_error = torch.abs(u_pred - u_true)

        errors = {
            'l2_error': torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item(),
            'max_abs_error': torch.max(abs_error).item(),
            'mean_abs_error': torch.mean(abs_error).item(),
        }

        return errors

    def save(self, filepath: str) -> None:
        """
        Save trained model and training history.

        Args:
            filepath: Path to save file (.pth)
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'pinn_config': self.pinn_config,
            'training_config': self.training_config
        }, filepath)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load trained model and training history.

        Args:
            filepath: Path to saved file (.pth)
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        logger.info(f"Model loaded from {filepath}")
