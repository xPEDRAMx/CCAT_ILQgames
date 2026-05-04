"""One-sided ("semi-")quadratic costs.

Penalizes a single dimension only when it is on a chosen side of a threshold
(used for soft state/control bounds before we layer a proper Augmented
Lagrangian on top). ``oriented_right=True`` penalizes ``z >= threshold``,
``oriented_right=False`` penalizes ``z <= threshold``.
"""

from __future__ import annotations

import numpy as np

from .base import ControlCost, StateCost


class _SemiquadraticBase:
    """Mixin implementing the semiquadratic value/grad/hess on dimension ``dim``."""

    def __init__(self, dim: int, threshold: float, oriented_right: bool) -> None:
        self.dim = dim
        self.threshold = float(threshold)
        self.oriented_right = bool(oriented_right)

    def _active_diff(self, z: np.ndarray) -> float:
        diff = z[self.dim] - self.threshold
        if self.oriented_right and diff > 0.0:
            return diff
        if (not self.oriented_right) and diff < 0.0:
            return diff
        return 0.0

    def _value(self, z: np.ndarray) -> float:
        d = self._active_diff(z)
        return 0.5 * d * d

    def _grad(self, z: np.ndarray) -> np.ndarray:
        g = np.zeros_like(z, dtype=float)
        d = self._active_diff(z)
        if d != 0.0:
            g[self.dim] = d
        return g

    def _hess(self, z: np.ndarray) -> np.ndarray:
        H = np.zeros((z.size, z.size))
        d = self._active_diff(z)
        if d != 0.0:
            H[self.dim, self.dim] = 1.0
        return H


class SemiquadraticCost(_SemiquadraticBase, StateCost):
    """Soft one-sided bound on a single state dimension."""

    def __init__(
        self,
        dim: int,
        threshold: float,
        oriented_right: bool,
        name: str = "",
        apply_after_time: int = -1,
    ) -> None:
        StateCost.__init__(self, name=name, apply_after_time=apply_after_time)
        _SemiquadraticBase.__init__(
            self, dim=dim, threshold=threshold, oriented_right=oriented_right
        )

    def value(self, x: np.ndarray, k: int = 0) -> float:
        return self._value(x)

    def grad(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        return self._grad(x)

    def hess(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        return self._hess(x)


class SemiquadraticControlCost(_SemiquadraticBase, ControlCost):
    """Soft one-sided bound on a single control dimension."""

    def __init__(
        self,
        dim: int,
        threshold: float,
        oriented_right: bool,
        name: str = "",
        apply_after_time: int = -1,
    ) -> None:
        ControlCost.__init__(self, name=name, apply_after_time=apply_after_time)
        _SemiquadraticBase.__init__(
            self, dim=dim, threshold=threshold, oriented_right=oriented_right
        )

    def value(self, u: np.ndarray, k: int = 0) -> float:
        return self._value(u)

    def grad(self, u: np.ndarray, k: int = 0) -> np.ndarray:
        return self._grad(u)

    def hess(self, u: np.ndarray, k: int = 0) -> np.ndarray:
        return self._hess(u)
