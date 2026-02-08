"""
Abstract base class for PDEs in PINN framework.

Defines the interface for PDE equations to be solved using
Physics-Informed Neural Networks.

References:
    - PhD Research Document: Section 2.2.1 (Problem Formulation)
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
        u(x, 0) = u_0(x)                     in Ω
        B[u] = g(x, t)                       on ∂Ω × [0, T]

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
