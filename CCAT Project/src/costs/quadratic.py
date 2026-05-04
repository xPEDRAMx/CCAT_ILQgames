"""Plain quadratic costs on a single scalar state dimension or control input.

Mirrors ``QuadraticCost`` from the C++ ilqgames code. Penalizes
``0.5 * (z[i] - nominal)^2`` for one component ``i``.
"""

from __future__ import annotations

import numpy as np

from .base import ControlCost, StateCost


class QuadraticCost(StateCost):
    """``c(x) = 0.5 * (x[dim] - nominal)^2`` for a single state dimension."""

    def __init__(
        self,
        dim: int,
        nominal: float = 0.0,
        name: str = "",
        apply_after_time: int = -1,
    ) -> None:
        super().__init__(name=name, apply_after_time=apply_after_time)
        self.dim = dim
        self.nominal = float(nominal)

    def value(self, x: np.ndarray, k: int = 0) -> float:
        diff = x[self.dim] - self.nominal
        return 0.5 * float(diff * diff)

    def grad(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        g = np.zeros_like(x, dtype=float)
        g[self.dim] = x[self.dim] - self.nominal
        return g

    def hess(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        H = np.zeros((x.size, x.size))
        H[self.dim, self.dim] = 1.0
        return H


class QuadraticControlCost(ControlCost):
    """``c(u) = 0.5 * (u[dim] - nominal)^2`` for a single control dimension."""

    def __init__(
        self,
        dim: int,
        nominal: float = 0.0,
        name: str = "",
        apply_after_time: int = -1,
    ) -> None:
        super().__init__(name=name, apply_after_time=apply_after_time)
        self.dim = dim
        self.nominal = float(nominal)

    def value(self, u: np.ndarray, k: int = 0) -> float:
        diff = u[self.dim] - self.nominal
        return 0.5 * float(diff * diff)

    def grad(self, u: np.ndarray, k: int = 0) -> np.ndarray:
        g = np.zeros_like(u, dtype=float)
        g[self.dim] = u[self.dim] - self.nominal
        return g

    def hess(self, u: np.ndarray, k: int = 0) -> np.ndarray:
        H = np.zeros((u.size, u.size))
        H[self.dim, self.dim] = 1.0
        return H
