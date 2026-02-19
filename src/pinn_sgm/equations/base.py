"""
Abstract base classes for PDEs and SDEs in PINN framework.

Defines the interface for equations to be solved using
Physics-Informed Neural Networks and Score-PINN methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import torch


class BasePDE(ABC):
    """
    Abstract base class for partial differential equations.

    All PDE implementations must inherit from this class and implement
    the abstract methods defining the equation structure.

    A general PDE has the form:
        ∂u/∂t + F[u, ∇u, ∇²u, x, t] = 0    in Ω × [0, T]
        u(x, 0) = u_0(x)                   in Ω
        B[u] = g(x, t)                     on ∂Ω × [0, T]

    where F is the PDE operator and B is the boundary operator.
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize base PDE.

        Args:
            spatial_dim: Spatial dimension of the problem
            device: Computation device
            dtype: Tensor data type
        """
        self.spatial_dim = spatial_dim
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual F[u] at collocation points.

        The residual should be zero for the exact solution.

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Temporal coordinates [Batch, 1]
            u: Solution values [Batch, 1]
            u_t: Time derivative ∂u/∂t [Batch, 1]
            u_x: Spatial gradient ∇u [Batch, spatial_dim]
            u_xx: Spatial Hessian ∇²u [Batch, spatial_dim]

        Returns:
            PDE residual [Batch, 1], should be zero for exact solution
        """
        pass

    @abstractmethod
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute initial condition u(x, 0).

        Args:
            x: Spatial coordinates [Batch, spatial_dim]

        Returns:
            Initial values u(x, 0) [Batch, 1]
        """
        pass

    def boundary_condition(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition residual.

        Default implementation returns zero (no boundary conditions).
        Override for problems with non-trivial boundary conditions.

        Args:
            x: Boundary coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]
            u: Solution at boundary [Batch, 1]

        Returns:
            Boundary residual [Batch, 1]
        """
        return torch.zeros_like(u)

    def analytical_solution(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute analytical solution (if available).

        Default implementation returns None. Override for equations
        with known analytical solutions.

        Args:
            x: Spatial coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]

        Returns:
            Analytical solution [Batch, 1] or None
        """
        return None

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to equation's device and dtype."""
        return tensor.to(device=self.device, dtype=self.dtype)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(spatial_dim={self.spatial_dim}, device={self.device})"


class BaseSDE(ABC):
    """
    Abstract base class for Stochastic Differential Equations (SDEs).

    An SDE has the general form:
        dX_t = f(X_t, t) dt + G(X_t, t) dW_t

    where:
        - f(x, t) ∈ ℝⁿ is the drift coefficient
        - G(x, t) ∈ ℝⁿˣᵐ is the diffusion coefficient
        - W_t ∈ ℝᵐ is an m-dimensional Brownian motion

    The probability density p(x, t) of X_t satisfies the Fokker-Planck equation:
        ∂p/∂t + ∇·(f p) - (1/2)∑ᵢⱼ ∂²/∂xᵢ∂xⱼ[(GGᵀ)ᵢⱼ p] = 0

    The score function s(x, t) = ∇ log p(x, t) satisfies the Score PDE:
        ∂s/∂t = ∇{L[s(x,t)]}
    where L[s] = (1/2)∇·(GGᵀs) + (1/2)||Gᵀs||² - ⟨A,s⟩ - ∇·A
    and A = f - (1/2)∇·(GGᵀ)

    This class is independent of BasePDE. Equations that need both the PDE
    interface (for PINNSolver) and the SDE interface (for ScorePINNSolver)
    should inherit from both: class MyEquation(BasePDE, BaseSDE).
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        self.spatial_dim = spatial_dim
        self.device = device
        self.dtype = dtype

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to equation's device and dtype."""
        return tensor.to(device=self.device, dtype=self.dtype)

    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift coefficient f(x, t).

        Args:
            x: State coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]

        Returns:
            Drift vector f(x, t) [Batch, spatial_dim]
        """
        pass

    @abstractmethod
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient G(x, t).

        Args:
            x: State coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]

        Returns:
            Diffusion matrix G(x, t) [Batch, spatial_dim, spatial_dim]
            For diagonal diffusion, returns [Batch, spatial_dim, spatial_dim] diagonal matrix
        """
        pass

    def diffusion_squared(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute D(x, t) = G(x, t) G(x, t)ᵀ.

        Default implementation computes GGᵀ from diffusion().
        Override for efficiency if D is known directly.

        Args:
            x: State coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]

        Returns:
            D(x, t) = GGᵀ [Batch, spatial_dim, spatial_dim]
        """
        G = self.diffusion(x, t)  # [Batch, spatial_dim, spatial_dim]
        D = torch.bmm(G, G.transpose(-2, -1))  # GGᵀ
        return D

    @abstractmethod
    def initial_score(self, x: torch.Tensor, t_epsilon: float = 0.1) -> torch.Tensor:
        """
        Compute initial score s₀(x) = ∇ₓ log p₀(x) at small time t_epsilon.

        This is REQUIRED for Score-PINN training. It defines the initial condition
        for the Score PDE: s(x, 0) = s₀(x).

        For singular initial conditions (e.g., Dirac delta), the score is evaluated
        at a small positive time t_epsilon to avoid singularities. For smooth initial
        distributions, t_epsilon can be ignored and the score computed at t=0.

        The initial score can be computed:
        - Analytically (if p₀ has a closed form)
        - Via automatic differentiation (if p₀ is differentiable)
        - From samples using score matching

        Args:
            x: State coordinates [Batch, spatial_dim]
            t_epsilon: Regularization time for singular ICs (default: 0.1)

        Returns:
            Initial score s₀(x) [Batch, spatial_dim]

        Example:
            >>> # Gaussian initial condition: p₀ ~ N(μ₀, Σ₀)
            >>> def initial_score(self, x, t_epsilon=0.1):
            ...     # For smooth Gaussian, ignore t_epsilon
            ...     return -self.Sigma0_inv @ (x - self.mu0)
            >>>
            >>> # Dirac delta: use t_epsilon for regularization
            >>> def initial_score(self, x, t_epsilon=0.1):
            ...     t_small = torch.full((x.shape[0], 1), t_epsilon)
            ...     return self.analytical_score(x, t_small)
        """
        pass

    def analytical_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute analytical score s(x, t) = ∇ₓ log p(x, t) for ALL t.

        This is OPTIONAL and used only for validation and error computation.
        If the full analytical solution is not available, return None.
        Score-PINN will learn s(x, t) from the Score PDE.

        Args:
            x: State coordinates [Batch, spatial_dim]
            t: Time values [Batch, 1]

        Returns:
            Score vector s(x, t) [Batch, spatial_dim] or None if not available

        Example:
            >>> # If analytical solution not available
            >>> def analytical_score(self, x, t):
            ...     return None  # Score-PINN will learn it!
        """
        return None

    def is_constant_coefficients(self) -> bool:
        """
        Check if drift and diffusion are constant (independent of x and t).

        Default implementation returns False. Override if coefficients are constant.

        Returns:
            True if f and G are constant, False otherwise
        """
        return False
