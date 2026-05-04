"""Lightweight numerical containers used throughout the solver.

These mirror the Eigen-backed structs in the original C++ ilqgames
(`LinearDynamicsApproximation`, `QuadraticCostApproximation`, `Strategy`,
`OperatingPoint`), but use plain numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LinearDynamics:
    """Discrete-time linearization of multi-player dynamics at one time step.

    Represents

        delta_x_{k+1} = A @ delta_x_k + sum_i B[i] @ delta_u_i_k.
    """

    A: np.ndarray  # shape (n, n)
    B: List[np.ndarray]  # length N, each (n, m_i)


@dataclass
class QuadraticCostApprox:
    """Quadratic approximation of a single player's stage (or terminal) cost.

    The expansion is

        c(x, u_1, ..., u_N) ~= c0
              + l_x^T dx + 0.5 dx^T Q_xx dx
              + sum_j ( r_j^T du_j + 0.5 du_j^T R_jj du_j )

    Off-diagonal control–control and state–control terms are zero in
    `ilqgames` and are not represented here (matching the C++ code).
    """

    Q: np.ndarray  # state Hessian, (n, n)
    l: np.ndarray  # state grad, (n,)
    R: Dict[int, np.ndarray]  # player_idx -> (m_j, m_j) Hessian
    r: Dict[int, np.ndarray]  # player_idx -> (m_j,) gradient


@dataclass
class Strategy:
    """Time-indexed affine state-error feedback law for one player.

        u_k = u_ref_k - P_k @ (x_k - x_ref_k) - alpha_k.

    For an open-loop strategy, `Ps` is identically zero and only `alphas`
    is populated.
    """

    Ps: List[np.ndarray]  # list of (m_i, n)
    alphas: List[np.ndarray]  # list of (m_i,)

    @classmethod
    def zero(cls, horizon: int, x_dim: int, u_dim: int) -> "Strategy":
        return cls(
            Ps=[np.zeros((u_dim, x_dim)) for _ in range(horizon)],
            alphas=[np.zeros(u_dim) for _ in range(horizon)],
        )


@dataclass
class OperatingPoint:
    """A trajectory: states and per-player controls over time."""

    xs: List[np.ndarray]  # length T+1, each (n,)
    us: List[List[np.ndarray]]  # us[k][i] -> (m_i,)
    t0: float = 0.0

    def horizon(self) -> int:
        return len(self.us)
