"""
Score-PINN solver for Stochastic Differential Equations (SDEs).

Based on: "Score-Based Physics-Informed Neural Networks for High-Dimensional
Fokker-Planck Equations" by Hu et al. (2024)
arXiv:2402.07465

Learns the score function s(x,t) = ∇ log p(x,t) by solving the Score PDE
using physics-informed neural networks. The learned score can be used for
score-based generative modeling and sampling.

Works with any SDE of the form: dX_t = f(X_t, t) dt + G(X_t, t) dW_t
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..equations.base import BaseSDE
from ..config import ScorePINNConfig, TrainingConfig
from .training_utils import setup_optimizer, setup_scheduler, compute_gradient_norm


class ScorePINNSolver:
    """
    Score-PINN solver for Stochastic Differential Equations (SDEs).

    Implements three methods for learning the score function:
    1. Score-PINN: Uses Score PDE with physics-informed loss
    2. Score Matching (SM): Matches conditional score on SDE trajectories
    3. Sliced Score Matching (SSM): SDE-agnostic projection-based approach

    Key equations:
    - Score PDE: ∂tst(x) = ∇x{L[st(x)]}
    - L[s] = (1/2)∇·(GGᵀs) + (1/2)||Gᵀs||² - ⟨A,s⟩ - ∇·A
    - A = f - (1/2)∇·(GGᵀ)

    Performance:
    - Uses Hutchinson Trace Estimation (HTE) by default for efficient
      divergence computation, making it practical for high dimensions
    - HTE reduces divergence computation from O(d) to O(1) gradient calls
    """

    def __init__(
        self,
        equation: BaseSDE,
        score_network: nn.Module,
        config: ScorePINNConfig,
        device: str = 'cpu'
    ):
        """
        Initialize Score-PINN solver.

        Args:
            equation: SDE to solve (BaseSDE instance)
                     Must implement drift(x, t), diffusion(x, t), and initial_score(x).
                     The analytical_score(x, t) is optional (only for validation).
            score_network: Neural network for score approximation
                          Should accept concatenated input [x, t] of shape [Batch, spatial_dim + 1]
                          and output score vector of shape [Batch, spatial_dim]
            config: Solver configuration
            device: Device for computation ('cpu' or 'cuda')
        """
        if not isinstance(equation, BaseSDE):
            raise TypeError(
                f"equation must be a BaseSDE instance, got {type(equation).__name__}. "
                "For Score-PINN, the equation must define drift(), diffusion(), and initial_score()."
            )

        self.equation = equation
        self.config = config
        self.device = device

        # Get spatial dimension from equation
        self.spatial_dim = equation.spatial_dim

        # Set score network
        self.score_network = score_network.to(device)

        # Infer dtype from network parameters
        self.dtype = next(score_network.parameters()).dtype

        self.optimizer = None
        self.scheduler = None
        self.history = {
            'loss_total': [],
            'loss_initial': [],
            'loss_residual': [],
            'learning_rate': [],
            'grad_norm': []
        }

    def _compute_L_operator(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L[s] operator:
            L[s] = (1/2)∇·(GGᵀs) + (1/2)||Gᵀs||² - ⟨A,s⟩ - ∇·A

        where A = f - (1/2)∇·(GGᵀ) for general SDE: dX = f dt + G dW

        For constant coefficients (f and G independent of x, t):
            - A = f (since ∇·(GGᵀ) = 0)
            - ∇·A = 0 (since f is constant)
        Simplified: L[s] = (1/2)∇·(Ds) + (1/2)sᵀDs - fᵀs

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Time coordinates [Batch, 1]
            s: Score function [Batch, spatial_dim]

        Returns:
            L_s: L[s] operator [Batch]
        """
        batch_size = x.shape[0]

        # Get drift and diffusion from equation (supports state-dependent coefficients)
        f = self.equation.drift(x, t)  # [Batch, spatial_dim]
        D = self.equation.diffusion_squared(x, t)  # [Batch, spatial_dim, spatial_dim]

        # Term 1: (1/2)∇·(Ds)
        # Compute Ds = D @ s
        Ds = torch.bmm(D, s.unsqueeze(-1)).squeeze(-1)  # [Batch, spatial_dim]

        if self.config.use_hte:
            # Use Hutchinson Trace Estimation (HTE)
            # HTE: Tr(∇f) ≈ E[v^T ∇f v] where v ~ Rademacher(±1)
            # This replaces O(batch_size * spatial_dim) gradient computations with O(n_samples)

            div_Ds_estimates = []
            for _ in range(self.config.n_hte_samples):
                # Sample Rademacher vector: each element is ±1 with equal probability
                v = torch.randint(0, 2, (batch_size, self.spatial_dim), device=self.device, dtype=s.dtype)
                v = 2.0 * v - 1.0  # Convert {0,1} to {-1,1}

                # Compute v^T ∇(Ds) using vector-Jacobian product
                vjp = torch.autograd.grad(
                    Ds,
                    x,
                    grad_outputs=v,
                    create_graph=True,
                    retain_graph=True
                )[0]  # [Batch, spatial_dim]

                # HTE estimate: Tr(∇Ds) ≈ v^T (∇Ds) v = sum(v * vjp)
                div_Ds_estimates.append((v * vjp).sum(dim=-1))  # [Batch]

            # Average over samples
            div_Ds = torch.stack(div_Ds_estimates).mean(dim=0)  # [Batch]
        else:
            # Exact computation (expensive for high dimensions)
            div_Ds = torch.zeros(batch_size, device=self.device)
            for j in range(self.spatial_dim):
                grad_Ds_j = torch.autograd.grad(
                    Ds[:, j].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True
                )[0]
                div_Ds += grad_Ds_j[:, j]

        term1 = 0.5 * div_Ds

        # Term 2: (1/2)||Gᵀs||² = (1/2)sᵀDs where D = GGᵀ
        # Since D = GGᵀ, we have ||Gᵀs||² = (Gᵀs)ᵀ(Gᵀs) = sᵀGGᵀs = sᵀDs
        # Ds is already computed above, so reuse it
        term2 = 0.5 * (s * Ds).sum(dim=-1)  # [Batch]

        # Term 3: -⟨A, s⟩
        # For constant coefficients: A = f
        # For state-dependent: A = f - (1/2)∇·(GGᵀ)
        # We optimize for the constant case (most common)
        if self.equation.is_constant_coefficients():
            # A = f (since ∇·D = 0 for constant D)
            A = f
            div_A = 0.0  # ∇·A = 0 for constant drift
        else:
            # State-dependent case: compute A = f - (1/2)∇·(GGᵀ)
            # Warning: O(d²) autograd calls — very expensive for d > ~5.
            # Prefer is_constant_coefficients() == True for high-dimensional problems.
            # (∇·D)ᵢ = Σⱼ ∂Dᵢⱼ/∂xⱼ
            div_D = torch.zeros(batch_size, self.spatial_dim, device=self.device, dtype=s.dtype)
            for i in range(self.spatial_dim):
                for j in range(self.spatial_dim):
                    grad_Dij = torch.autograd.grad(
                        D[:, i, j].sum(),
                        x,
                        create_graph=True,
                        retain_graph=True
                    )[0]  # [Batch, spatial_dim]
                    div_D[:, i] += grad_Dij[:, j]

            A = f - 0.5 * div_D  # [Batch, spatial_dim]

            # ∇·A = Σⱼ ∂Aⱼ/∂xⱼ
            div_A = torch.zeros(batch_size, device=self.device, dtype=s.dtype)
            for j in range(self.spatial_dim):
                grad_Aj = torch.autograd.grad(
                    A[:, j].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True
                )[0]  # [Batch, spatial_dim]
                div_A += grad_Aj[:, j]

        term3 = -(A * s).sum(dim=-1)  # [Batch]

        # Term 4: -∇·A (skip for constant coefficients)
        term4 = -div_A if not self.equation.is_constant_coefficients() else 0.0

        # L[s] = term1 + term2 + term3 + term4
        L_s = term1 + term2 + term3 + term4

        return L_s  # [Batch]

    def _compute_score_pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Score PDE residual:
            ∂tst(x) - ∇x{L[st(x)]}

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Time coordinates [Batch, 1]

        Returns:
            residual: PDE residual [Batch]
        """
        batch_size = x.shape[0]

        # Enable gradients
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        # Compute score (concatenate x and t)
        xt = torch.cat([x, t], dim=-1)
        s = self.score_network(xt)  # [Batch, spatial_dim]

        # Compute time derivative ∂s/∂t
        s_t_list = []
        for i in range(self.spatial_dim):
            s_t_i = torch.autograd.grad(
                s[:, i].sum(),
                t,
                create_graph=True,
                retain_graph=True
            )[0]  # [Batch, 1]
            s_t_list.append(s_t_i.squeeze(-1))

        s_t = torch.stack(s_t_list, dim=-1)  # [Batch, spatial_dim]

        # Compute L[s]
        L_s = self._compute_L_operator(x, t, s)  # [Batch]

        # Compute ∇x{L[s]}
        grad_L_s = torch.autograd.grad(
            L_s.sum(),
            x,
            create_graph=True,
            retain_graph=True
        )[0]  # [Batch, spatial_dim]

        # Residual: ∂ts - ∇x{L[s]}
        residual = s_t - grad_L_s  # [Batch, spatial_dim]

        # Return scalar residual (mean squared)
        loss = (residual ** 2).mean()

        return loss

    def _compute_initial_condition_loss(self) -> torch.Tensor:
        """
        Compute initial condition loss:
            ||s(x, 0) - s₀(x)||² where s₀(x) = ∇ₓ log p₀(x)

        Uses the initial score s₀(x) from the equation's initial_score() method.
        For singular initial conditions (e.g., Dirac delta), the score is evaluated
        at a small positive time t_ε (from config) to avoid singularities.

        Returns:
            loss: Initial condition loss (scalar)
        """
        # Sample initial points
        x_init = self._sample_initial_points(self.config.n_initial)

        # Evaluate network at regularization time t_epsilon
        t_init = torch.full((self.config.n_initial, 1), self.config.t_epsilon, device=self.device, dtype=self.dtype)

        # Compute predicted score (concatenate x and t)
        xt_init = torch.cat([x_init, t_init], dim=-1)
        s_pred = self.score_network(xt_init)  # [n_initial, spatial_dim]

        # Compute true initial score from equation (at same t_epsilon)
        s_true = self.equation.initial_score(x_init, t_epsilon=self.config.t_epsilon)  # [n_initial, spatial_dim]

        # MSE loss
        loss = torch.mean((s_pred - s_true) ** 2)

        return loss

    def _compute_residual_loss(self) -> torch.Tensor:
        """
        Compute PDE residual loss for Score-PINN.

        Returns:
            loss: Residual loss (scalar)
        """
        # Sample collocation points
        x_pde, t_pde = self._sample_collocation_points(self.config.n_collocation)

        # Compute residual for batch
        total_loss = 0.0
        n_batches = (self.config.n_collocation + self.config.batch_size - 1) // self.config.batch_size

        for i in range(n_batches):
            start_idx = i * self.config.batch_size
            end_idx = min((i + 1) * self.config.batch_size, self.config.n_collocation)

            x_batch = x_pde[start_idx:end_idx]
            t_batch = t_pde[start_idx:end_idx]

            residual_loss = self._compute_score_pde_residual(x_batch, t_batch)
            total_loss += residual_loss * (end_idx - start_idx)

        return total_loss / self.config.n_collocation

    def _sample_initial_points(self, n_points: int) -> torch.Tensor:
        """Sample points from initial distribution."""
        # Sample from initial distribution p0(x)
        # For now, use uniform sampling in domain
        if self.spatial_dim == 1:
            x = torch.rand(n_points, 1, device=self.device)
            x = x * (self.config.x_max - self.config.x_min) + self.config.x_min
        else:
            x = torch.rand(n_points, self.spatial_dim, device=self.device)
            x = x * (self.config.x_max - self.config.x_min) + self.config.x_min

        return x

    def _sample_collocation_points(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample collocation points in space-time domain."""
        # Sample spatial points
        if self.spatial_dim == 1:
            x = torch.rand(n_points, 1, device=self.device)
            x = x * (self.config.x_max - self.config.x_min) + self.config.x_min
        else:
            x = torch.rand(n_points, self.spatial_dim, device=self.device)
            x = x * (self.config.x_max - self.config.x_min) + self.config.x_min

        # Sample time points (avoid t=0 for PDE residual)
        t = torch.rand(n_points, 1, device=self.device)
        t = t * (self.config.t_max - self.config.t_min) + self.config.t_min
        t = torch.clamp(t, min=1e-6)  # Avoid t=0

        return x, t

    def train(self, training_config: TrainingConfig) -> Dict[str, List[float]]:
        """
        Train the score function s(x,t) = ∇ log p(x,t).

        Uses Score-PINN loss:
            L = λ_initial * E[(s0 - ∇log p0)²] + λ_residual * E[(∂ts - ∇x{L[s]})²]

        Args:
            training_config: Training configuration

        Returns:
            history: Training history
        """
        # Setup optimizer and scheduler
        self.optimizer = setup_optimizer(self.score_network.parameters(), training_config)
        self.scheduler = setup_scheduler(self.optimizer, training_config)

        # Training header
        start_time = time.time()
        if training_config.verbose:
            print(f"\n{'Epoch':<12} {'Total':<12} {'Initial':<12} {'Residual':<12} {'LR':<12} {'Time':<12}")
            print("-" * 72)

        # Training loop
        for epoch in range(training_config.epochs):
            self.optimizer.zero_grad()

            # Compute losses
            loss_initial = self._compute_initial_condition_loss()
            loss_residual = self._compute_residual_loss()

            # Total loss
            loss_total = (
                self.config.lambda_initial * loss_initial +
                self.config.lambda_residual * loss_residual
            )

            # Backward pass
            loss_total.backward()

            # Compute gradient norm (before clipping)
            total_grad_norm = compute_gradient_norm(self.score_network.parameters())

            # Gradient clipping
            if training_config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.score_network.parameters(),
                    training_config.gradient_clip_val
                )

            self.optimizer.step()

            # Update scheduler
            if training_config.lr_scheduler == 'step':
                self.scheduler.step()
            elif training_config.lr_scheduler == 'plateau':
                self.scheduler.step(loss_total)

            # Record history
            self.history['loss_total'].append(loss_total.item())
            self.history['loss_initial'].append(loss_initial.item())
            self.history['loss_residual'].append(loss_residual.item())
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['grad_norm'].append(total_grad_norm)

            # Periodic logging
            if training_config.verbose and epoch % training_config.log_interval == 0:
                elapsed = time.time() - start_time
                it_per_sec = (epoch + 1) / elapsed if elapsed > 0 else 0
                epoch_str = f"{epoch}/{training_config.epochs}"
                time_str = f"{it_per_sec:.2f}it/s"
                current_lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"{epoch_str:<12} "
                    f"{loss_total.item():<12.4e} "
                    f"{loss_initial.item():<12.4e} "
                    f"{loss_residual.item():<12.4e} "
                    f"{current_lr:<12.4e} "
                    f"{time_str:<12}"
                )

        # Final summary
        if training_config.verbose:
            total_time = time.time() - start_time
            print("-" * 72)
            print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)")
            print(
                f"Final: Total={self.history['loss_total'][-1]:.4e}, "
                f"Initial={self.history['loss_initial'][-1]:.4e}, "
                f"Residual={self.history['loss_residual'][-1]:.4e}\n"
            )

        return self.history

    def predict_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict score function s(x,t) = ∇ log p(x,t).

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Time coordinates [Batch, 1]

        Returns:
            score: Score vector [Batch, spatial_dim]
        """
        self.score_network.eval()
        with torch.no_grad():
            # Concatenate x and t
            xt = torch.cat([x, t], dim=-1)
            score = self.score_network(xt)
        return score
