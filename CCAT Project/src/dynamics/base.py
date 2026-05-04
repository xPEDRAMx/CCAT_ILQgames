"""Abstract base class for continuous-time single-player dynamics with an RK4
discretizer and an analytic discrete-time linearization helper.

Mirrors `SinglePlayerDynamicalSystem` from the C++ ilqgames code, with the
convention that `linearize_discrete` returns matrices ``(A, B)`` such that

    x_{k+1} - x*_{k+1} ~= A (x_k - x*_k) + B (u_k - u*_k).

These are obtained from the continuous Jacobians by an Euler step, matching
the discretization used in ``ilqgames``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class DynamicalSystem(ABC):
    """Single-player continuous-time dynamical system."""

    def __init__(self, x_dim: int, u_dim: int, dt: float = 0.1) -> None:
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dt = dt

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time dynamics ``\\dot x = f(x, u)``."""

    @abstractmethod
    def jacobians_continuous(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(df/dx, df/du)`` evaluated at ``(x, u)``."""

    def integrate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """RK4 step of length ``dt`` (matches ilqgames default integrator)."""

        dt = self.dt
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def linearize_discrete(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Discrete-time linearization via Euler step on the continuous Jacobians.

        ``ilqgames`` uses an Euler discretization of the linearization even
        though the rollout itself uses RK4. We match that convention exactly
        so that the LQ approximation is consistent with the C++ reference.
        """

        Ac, Bc = self.jacobians_continuous(x, u)
        n = self.x_dim
        A = np.eye(n) + self.dt * Ac
        B = self.dt * Bc
        return A, B
