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
import numpy as np

from ..equations.base import BasePDE
from ..config import PINNConfig, TrainingConfig
from .training_utils import setup_optimizer, setup_scheduler, compute_gradient_norm

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
        self.network = network

        # Move to device
        self.device = self.pinn_config.torch_device
        self.dtype = self.pinn_config.dtype
        self.network = self.network.to(device=self.device, dtype=self.dtype)

        # Optimizer and scheduler are created at train() time so that
        # different training configs can be used without reconstructing the solver.
        self.optimizer = None
        self.scheduler = None

        # Training history
        self.history = {
            'loss_total': [],
            'loss_pde': [],
            'loss_ic': [],
            'loss_bc': [],
            'loss_norm': [],
            'learning_rate': [],
            'grad_norm': []
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

        Supports both 1D and multi-dimensional spatial problems:
        - 1D: u_x is [Batch, 1], u_xx is [Batch, 1, 1] (Hessian matrix)
        - nD: u_x is [Batch, n], u_xx is [Batch, n, n] (Hessian matrix)

        Args:
            x: Spatial coordinates [Batch, spatial_dim], requires_grad=True
            t: Temporal coordinates [Batch, 1], requires_grad=True

        Returns:
            u: Solution u(x, t) [Batch, 1]
            u_t: Time derivative ∂u/∂t [Batch, 1]
            u_x: Spatial gradient ∇u [Batch, spatial_dim]
            u_xx: Spatial Hessian ∇²u [Batch, spatial_dim, spatial_dim]
        """
        # --- Forward pass ---
        inputs = torch.cat([x, t], dim=-1)  # [Batch, spatial_dim + 1]
        u = self.network(inputs)  # [Batch, 1]

        # --- First-order derivatives ---
        grad_outputs = torch.ones_like(u)

        u_grads = torch.autograd.grad(
            outputs=u,
            inputs=[x, t],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )
        u_x = u_grads[0]  # ∂u/∂x [Batch, spatial_dim]
        u_t = u_grads[1]  # ∂u/∂t [Batch, 1]

        # --- Hessian computation ---
        spatial_dim = self.equation.spatial_dim
        batch_size = x.shape[0]

        if spatial_dim == 1:
            # 1D case: simple second derivative
            u_xx = torch.autograd.grad(
                outputs=u_x,
                inputs=x,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
                retain_graph=True
            )[0]  # ∂²u/∂x² [Batch, 1]
            u_xx = u_xx.unsqueeze(-1)  # Reshape to [Batch, 1, 1] for consistency
        else:
            # Multi-dimensional case: compute full Hessian matrix
            u_xx = torch.zeros(batch_size, spatial_dim, spatial_dim, device=x.device, dtype=x.dtype)

            for i in range(spatial_dim):
                # Compute ∂²u/∂xⱼ∂xᵢ for all j
                u_xx_i = torch.autograd.grad(
                    outputs=u_x[:, i],
                    inputs=x,
                    grad_outputs=torch.ones(batch_size, device=x.device, dtype=x.dtype),
                    create_graph=True,
                    retain_graph=True
                )[0]  # [Batch, spatial_dim]

                u_xx[:, i, :] = u_xx_i

        return u, u_t, u_x, u_xx

    def _sample_collocation_points(self) -> Dict[str, torch.Tensor]:
        """
        Sample collocation points for PDE, initial, and boundary conditions.

        Supports both 1D and multi-dimensional spatial domains.

        Returns:
            Dictionary with keys:
                - 'x_pde', 't_pde': Interior points for PDE residual
                - 'x_ic': Points for initial condition
                - 'x_bc', 't_bc': Boundary points (if applicable)
        """
        cfg = self.pinn_config
        spatial_dim = self.equation.spatial_dim

        # --- Interior points ---
        x_pde = torch.rand(cfg.num_collocation, spatial_dim, device=self.device, dtype=self.dtype)
        x_pde = cfg.x_min + (cfg.x_max - cfg.x_min) * x_pde
        t_pde = cfg.T * torch.rand(cfg.num_collocation, 1, device=self.device, dtype=self.dtype)

        # --- Initial condition points ---
        x_ic = torch.rand(cfg.num_initial, spatial_dim, device=self.device, dtype=self.dtype)
        x_ic = cfg.x_min + (cfg.x_max - cfg.x_min) * x_ic

        # --- Boundary points ---
        if cfg.num_boundary > 0:
            t_bc = cfg.T * torch.rand(cfg.num_boundary, 1, device=self.device, dtype=self.dtype)

            if spatial_dim == 1:
                # 1D: boundary consists of two points
                x_bc_left = torch.full((cfg.num_boundary // 2, 1), cfg.x_min, device=self.device, dtype=self.dtype)
                x_bc_right = torch.full((cfg.num_boundary - cfg.num_boundary // 2, 1), cfg.x_max, device=self.device, dtype=self.dtype)
                x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
            else:
                # Multi-D: sample on boundary hypercube faces
                # For simplicity, sample uniformly on all faces
                points_per_face = cfg.num_boundary // (2 * spatial_dim)
                boundary_points = []

                for dim in range(spatial_dim):
                    # Sample on lower face (x_dim = x_min)
                    x_face = torch.rand(points_per_face, spatial_dim, device=self.device, dtype=self.dtype)
                    x_face = cfg.x_min + (cfg.x_max - cfg.x_min) * x_face
                    x_face[:, dim] = cfg.x_min
                    boundary_points.append(x_face)

                    # Sample on upper face (x_dim = x_max)
                    x_face = torch.rand(points_per_face, spatial_dim, device=self.device, dtype=self.dtype)
                    x_face = cfg.x_min + (cfg.x_max - cfg.x_min) * x_face
                    x_face[:, dim] = cfg.x_max
                    boundary_points.append(x_face)

                x_bc = torch.cat(boundary_points, dim=0)
                # Adjust t_bc size to match
                t_bc = t_bc[:x_bc.shape[0]]
        else:
            x_bc = torch.empty(0, spatial_dim, device=self.device, dtype=self.dtype)
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
            x: Spatial coordinates [Batch, spatial_dim]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Mean squared PDE residual
        """
        # --- Enable gradients ---
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        # --- Derivatives ---
        u, u_t, u_x, u_xx = self._compute_derivatives(x, t)

        # --- PDE residual ---
        residual = self.equation.pde_residual(x, t, u, u_t, u_x, u_xx)
        loss = torch.mean(residual ** 2)

        return loss

    def _compute_ic_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition loss: L_IC = (1/N) Σ |u(x,0) - u_0(x)|²

        Args:
            x: Spatial coordinates [Batch, spatial_dim]

        Returns:
            Mean squared initial condition error
        """
        t_zero = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

        # --- Network prediction at t=0 ---
        inputs = torch.cat([x, t_zero], dim=-1)
        u_pred = self.network(inputs)

        # --- True initial condition ---
        u_true = self.equation.initial_condition(x)

        loss = torch.mean((u_pred - u_true) ** 2)

        return loss

    def _compute_bc_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss: L_BC = (1/N) Σ |B[u] - g|²

        Args:
            x: Boundary coordinates [Batch, spatial_dim]
            t: Temporal coordinates [Batch, 1]

        Returns:
            Mean squared boundary condition error
        """
        if x.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # --- Network prediction ---
        inputs = torch.cat([x, t], dim=-1)
        u_pred = self.network(inputs)

        # --- Boundary residual ---
        residual = self.equation.boundary_condition(x, t, u_pred)
        loss = torch.mean(residual ** 2)

        return loss

    def _compute_normalization_loss(self, t: torch.Tensor) -> torch.Tensor:
        """
        Enforce normalization constraint: ∫ p(x, t) dx = 1

        This is critical for probability densities. We approximate the integral:
        - 1D: trapezoidal rule on a fine grid
        - Multi-D: Monte Carlo integration

        Args:
            t: Time value (scalar or single-element tensor)

        Returns:
            Normalization violation: (∫p dx - 1)²
        """
        if not self.pinn_config.enforce_normalization:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        spatial_dim = self.equation.spatial_dim

        # Time value
        if isinstance(t, float):
            t_val = t
        else:
            t_val = t.item() if t.numel() == 1 else t.mean().item()

        if spatial_dim == 1:
            # --- 1D: trapezoidal rule ---
            num_integration_points = self.pinn_config.num_integration
            x_grid = torch.linspace(
                self.pinn_config.x_min,
                self.pinn_config.x_max,
                num_integration_points,
                device=self.device,
                dtype=self.dtype
            ).unsqueeze(-1)  # [N, 1]

            t_grid = torch.full((num_integration_points, 1), t_val, device=self.device, dtype=self.dtype)

            with torch.no_grad():
                inputs = torch.cat([x_grid, t_grid], dim=-1)
                p_values = self.network(inputs).squeeze()  # [N]

            dx = (self.pinn_config.x_max - self.pinn_config.x_min) / (num_integration_points - 1)
            integral = torch.trapezoid(p_values, dx=dx)

        else:
            # --- Multi-D: Monte Carlo ---
            num_mc_samples = self.pinn_config.num_mc_samples
            x_mc = torch.rand(num_mc_samples, spatial_dim, device=self.device, dtype=self.dtype)
            x_mc = self.pinn_config.x_min + (self.pinn_config.x_max - self.pinn_config.x_min) * x_mc
            t_mc = torch.full((num_mc_samples, 1), t_val, device=self.device, dtype=self.dtype)

            with torch.no_grad():
                inputs = torch.cat([x_mc, t_mc], dim=-1)
                p_values = self.network(inputs).squeeze()  # [N]

            volume = (self.pinn_config.x_max - self.pinn_config.x_min) ** spatial_dim
            integral = torch.mean(p_values) * volume

        # --- Normalization violation ---
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
        # --- PDE loss ---
        loss_pde = self._compute_pde_loss(
            collocation_points['x_pde'],
            collocation_points['t_pde']
        )

        # --- Initial condition loss ---
        loss_ic = self._compute_ic_loss(collocation_points['x_ic'])

        # --- Boundary condition loss ---
        loss_bc = self._compute_bc_loss(
            collocation_points['x_bc'],
            collocation_points['t_bc']
        )

        # --- Normalization loss ---
        t_norm = self.pinn_config.T * torch.rand(1, device=self.device, dtype=self.dtype)
        loss_norm = self._compute_normalization_loss(t_norm)

        # --- Weighted total ---
        loss_total = (
            self.pinn_config.pde_weight * loss_pde +
            self.pinn_config.ic_weight * loss_ic +
            self.pinn_config.bc_weight * loss_bc +
            self.pinn_config.normalization_weight * loss_norm
        )

        # --- Update history ---
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

        # --- Optimizer setup ---
        self.optimizer = setup_optimizer(self.network.parameters(), self.training_config)
        self.scheduler = setup_scheduler(self.optimizer, self.training_config)

        # --- Training header ---
        start_time = time.time()
        if self.training_config.verbose:
            print(f"\n{'Epoch':<12} {'Loss':<12} {'PDE':<12} {'IC':<12} {'BC':<12} {'LR':<12} {'Time':<12}")
            print("-" * 84)

        for epoch in range(self.training_config.epochs):
            # --- Collocation points ---
            collocation_points = self._sample_collocation_points()

            # --- Zero gradients ---
            self.optimizer.zero_grad()

            # --- Compute loss ---
            loss = self._compute_total_loss(collocation_points)

            # --- Backward pass ---
            loss.backward()

            # --- Gradient norm ---
            total_grad_norm = compute_gradient_norm(self.network.parameters())
            self.history['grad_norm'].append(total_grad_norm)

            # --- Gradient clipping ---
            if self.training_config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.training_config.gradient_clip_val
                )

            # --- Optimizer step ---
            self.optimizer.step()

            # --- LR scheduler ---
            if self.training_config.lr_scheduler == 'plateau':
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step()

            # --- Logging ---
            if epoch % self.training_config.log_interval == 0 and self.training_config.verbose:
                elapsed = time.time() - start_time
                it_per_sec = (epoch + 1) / elapsed if elapsed > 0 else 0
                epoch_str = f"{epoch}/{self.training_config.epochs}"
                time_str = f"{it_per_sec:.2f}it/s"
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"{epoch_str:<12} "
                    f"{loss.item():<12.4e} "
                    f"{self.history['loss_pde'][-1]:<12.4e} "
                    f"{self.history['loss_ic'][-1]:<12.4e} "
                    f"{self.history['loss_bc'][-1]:<12.4e} "
                    f"{current_lr:<12.4e} "
                    f"{time_str:<12}"
                )

        # --- Final summary ---
        if self.training_config.verbose:
            total_time = time.time() - start_time
            print("-" * 84)
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
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            if t.dim() == 1:
                t = t.unsqueeze(-1)

            x = x.to(device=self.device, dtype=self.dtype)
            t = t.to(device=self.device, dtype=self.dtype)

            # --- Forward pass ---
            inputs = torch.cat([x, t], dim=-1)
            u = self.network(inputs)

        return u
