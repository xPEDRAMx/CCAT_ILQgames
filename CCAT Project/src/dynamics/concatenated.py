"""Stack independent single-player systems into one multi-player system.

Mirrors ``ConcatenatedDynamicalSystem`` from the C++ ilqgames. The big state
``x`` is the vertical concatenation of each subsystem's state, and the
multi-player linearization is block-diagonal in ``A`` / block-column in each
``B_i``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import DynamicalSystem


class ConcatenatedSystem:
    """Concatenate single-player dynamical systems into one big system."""

    def __init__(self, subsystems: List[DynamicalSystem], dt: float = 0.1) -> None:
        if len(subsystems) == 0:
            raise ValueError("Need at least one subsystem.")
        for sub in subsystems:
            if abs(sub.dt - dt) > 1e-12:
                raise ValueError(
                    f"Subsystem dt={sub.dt} does not match concatenated dt={dt}."
                )
        self.subsystems = subsystems
        self.dt = dt
        self.x_dims = [sub.x_dim for sub in subsystems]
        self.u_dims = [sub.u_dim for sub in subsystems]
        self.x_dim = int(sum(self.x_dims))
        self._x_offsets = np.cumsum([0] + self.x_dims).tolist()
        self.num_players = len(subsystems)

    def player_state(self, x: np.ndarray, i: int) -> np.ndarray:
        return x[self._x_offsets[i] : self._x_offsets[i + 1]]

    def player_state_indices(self, i: int) -> Tuple[int, int]:
        return self._x_offsets[i], self._x_offsets[i + 1]

    def f(self, x: np.ndarray, us: List[np.ndarray]) -> np.ndarray:
        out = np.zeros(self.x_dim)
        for i, sub in enumerate(self.subsystems):
            xi = self.player_state(x, i)
            ui = us[i]
            out[self._x_offsets[i] : self._x_offsets[i + 1]] = sub.f(xi, ui)
        return out

    def integrate(self, x: np.ndarray, us: List[np.ndarray]) -> np.ndarray:
        """RK4 step using the joint dynamics (per-subsystem RK4 is equivalent
        because the subsystems are decoupled)."""

        dt = self.dt
        k1 = self.f(x, us)
        k2 = self.f(x + 0.5 * dt * k1, us)
        k3 = self.f(x + 0.5 * dt * k2, us)
        k4 = self.f(x + dt * k3, us)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def linearize_discrete(
        self, x: np.ndarray, us: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Block-diagonal big-A; per-player big-B that injects player i's
        controls into player i's state block."""

        n = self.x_dim
        A = np.eye(n)
        Bs = [np.zeros((n, m_i)) for m_i in self.u_dims]
        for i, sub in enumerate(self.subsystems):
            xi = self.player_state(x, i)
            ui = us[i]
            Ai, Bi = sub.linearize_discrete(xi, ui)
            r0, r1 = self._x_offsets[i], self._x_offsets[i + 1]
            A[r0:r1, r0:r1] = Ai
            Bs[i][r0:r1, :] = Bi
        return A, Bs
