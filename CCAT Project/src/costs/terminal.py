"""Terminal-time goal cost.

Implements the slide's terminal term

    psi_i(X_i(T)) = (b/2) * || X_i(T) - X_i^d ||^2

restricted to the position sub-state ``(p_x, p_y)`` of player ``i``. The cost
is "active" only at the final time step ``T`` (in the discretization), via
``apply_after_time = horizon - 1`` set by the example builder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import StateCost


class GoalCost(StateCost):
    """Quadratic position deviation from a goal point, applied at one time step."""

    def __init__(
        self,
        position_indices: Sequence[int],
        goal: Sequence[float],
        name: str = "goal",
        apply_after_time: int = -1,
    ) -> None:
        super().__init__(name=name, apply_after_time=apply_after_time)
        self.idx = tuple(int(i) for i in position_indices)
        self.goal = np.asarray(goal, dtype=float)
        if len(self.idx) != self.goal.size:
            raise ValueError("position_indices and goal must have the same length.")

    def _diff(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[i] for i in self.idx]) - self.goal

    def value(self, x: np.ndarray, k: int = 0) -> float:
        d = self._diff(x)
        return 0.5 * float(d @ d)

    def grad(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        g = np.zeros_like(x, dtype=float)
        d = self._diff(x)
        for j, i in enumerate(self.idx):
            g[i] = d[j]
        return g

    def hess(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        H = np.zeros((x.size, x.size))
        for i in self.idx:
            H[i, i] = 1.0
        return H
