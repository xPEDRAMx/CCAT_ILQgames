"""4-state unicycle / kinematic vehicle model in the CCAT slide form.

State (slide 16/19):

    x = [p_x, p_y, theta, v]^T

Control:

    u = [kappa, a]^T,   where kappa is path curvature

Continuous-time dynamics:

    \\dot p_x   = v cos(theta)
    \\dot p_y   = v sin(theta)
    \\dot theta = v * kappa
    \\dot v     = a

Note this differs from the C++ ``SinglePlayerUnicycle4D`` only in that the
yaw-rate input is parameterized as ``kappa = omega/v`` (curvature). With
``omega = v * kappa`` the two are equivalent. The CCAT formulation uses
``kappa`` because the lateral-acceleration cost ``kappa^2 * v^4`` is then
clean and matches the slide's smoothness term.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import DynamicalSystem


class Unicycle4D(DynamicalSystem):
    """4D unicycle in the CCAT slide form (curvature input)."""

    PX, PY, THETA, V = 0, 1, 2, 3
    KAPPA, A = 0, 1

    def __init__(self, dt: float = 0.1) -> None:
        super().__init__(x_dim=4, u_dim=2, dt=dt)

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta, v = x[self.THETA], x[self.V]
        kappa, accel = u[self.KAPPA], u[self.A]
        return np.array(
            [
                v * np.cos(theta),
                v * np.sin(theta),
                v * kappa,
                accel,
            ],
            dtype=float,
        )

    def jacobians_continuous(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        theta, v = x[self.THETA], x[self.V]
        kappa = u[self.KAPPA]

        Ac = np.zeros((4, 4))
        Ac[self.PX, self.THETA] = -v * np.sin(theta)
        Ac[self.PX, self.V] = np.cos(theta)
        Ac[self.PY, self.THETA] = v * np.cos(theta)
        Ac[self.PY, self.V] = np.sin(theta)
        Ac[self.THETA, self.V] = kappa
        # df_v/dx = 0, df_theta/dtheta = 0

        Bc = np.zeros((4, 2))
        Bc[self.THETA, self.KAPPA] = v
        Bc[self.V, self.A] = 1.0
        return Ac, Bc
