"""Aggregate a player's stage cost and produce its quadratic approximation.

This mirrors ``PlayerCost`` from C++ ilqgames. The structure is:

    J_i(x, u_1, ..., u_N, k) = sum_s w_s * c_s(x, k)             # state costs
                             + sum_p sum_t w_t * c_t(u_p, k)     # control costs

where each "control cost" is associated with a specific player ``p`` (which
need not be ``i`` — agent ``i`` may want to penalize agent ``j``'s controls,
though this is rarely used).

The quadratic approximation about an operating point ``(x, u_1, ..., u_N)``
returns ``Q, l, R, r`` matching ``QuadraticCostApprox`` in
``src.utils.types``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import ControlCost, Cost, StateCost
from ..utils.types import QuadraticCostApprox


class PlayerCost:
    """Sum of weighted state and control costs for a single player."""

    def __init__(self, x_dim: int, u_dims: List[int]) -> None:
        self.x_dim = x_dim
        self.u_dims = list(u_dims)
        self.num_players = len(u_dims)

        # state_costs[k] = list of (cost, weight)
        self._state_costs: List[Tuple[StateCost, float]] = []
        # control_costs[player_idx] = list of (cost, weight)
        self._control_costs: Dict[int, List[Tuple[ControlCost, float]]] = {
            i: [] for i in range(self.num_players)
        }

    # ----- registration ----------------------------------------------------
    def add_state_cost(self, cost: StateCost, weight: float = 1.0) -> "PlayerCost":
        if not isinstance(cost, StateCost):
            raise TypeError("add_state_cost requires a StateCost.")
        self._state_costs.append((cost, float(weight)))
        return self

    def add_control_cost(
        self, player_idx: int, cost: ControlCost, weight: float = 1.0
    ) -> "PlayerCost":
        if not isinstance(cost, ControlCost):
            raise TypeError("add_control_cost requires a ControlCost.")
        if player_idx not in self._control_costs:
            raise KeyError(f"Unknown player_idx {player_idx} (have {self.num_players} players).")
        self._control_costs[player_idx].append((cost, float(weight)))
        return self

    # ----- evaluation ------------------------------------------------------
    def value(self, x: np.ndarray, us: List[np.ndarray], k: int) -> float:
        v = 0.0
        for cost, w in self._state_costs:
            if cost.is_active(k):
                v += w * cost.value(x, k)
        for p, cost_list in self._control_costs.items():
            up = us[p]
            for cost, w in cost_list:
                if cost.is_active(k):
                    v += w * cost.value(up, k)
        return v

    def quadraticize(
        self, x: np.ndarray, us: List[np.ndarray], k: int
    ) -> QuadraticCostApprox:
        """Return the quadratic approximation of this player's cost at ``(x, us, k)``."""

        n = self.x_dim
        Q = np.zeros((n, n))
        l = np.zeros(n)
        R: Dict[int, np.ndarray] = {}
        r: Dict[int, np.ndarray] = {}
        for p, m in enumerate(self.u_dims):
            R[p] = np.zeros((m, m))
            r[p] = np.zeros(m)

        for cost, w in self._state_costs:
            if not cost.is_active(k):
                continue
            l += w * cost.grad(x, k)
            Q += w * cost.hess(x, k)

        for p, cost_list in self._control_costs.items():
            up = us[p]
            for cost, w in cost_list:
                if not cost.is_active(k):
                    continue
                r[p] += w * cost.grad(up, k)
                R[p] += w * cost.hess(up, k)

        # Symmetrize for numerical safety.
        Q = 0.5 * (Q + Q.T)
        for p in R:
            R[p] = 0.5 * (R[p] + R[p].T)

        return QuadraticCostApprox(Q=Q, l=l, R=R, r=r)
