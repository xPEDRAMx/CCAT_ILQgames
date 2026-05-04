from __future__ import annotations

from typing import List, Optional

import numpy as np

from costs.base_cost import BaseCost, QuadraticApprox


class GoalCost(BaseCost):
    """
    Quadratic goal-tracking cost on a subset of state indices.
    """

    def __init__(
        self,
        state_indices: List[int],
        goal: np.ndarray,
        weight: float = 1.0,
        terminal: bool = False,
    ) -> None:
        self.state_indices = list(state_indices)
        self.goal = np.asarray(goal, dtype=float).reshape(-1)
        self.weight = float(weight)
        self.terminal = bool(terminal)
        if len(self.state_indices) != self.goal.shape[0]:
            raise ValueError("state_indices and goal must have the same length.")

    def evaluate(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> float:
        _ = us, k
        xv = np.asarray(x, dtype=float).reshape(-1)[self.state_indices]
        d = xv - self.goal
        return 0.5 * self.weight * float(d @ d)

    def quadraticize(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> QuadraticApprox:
        nx = np.asarray(x, dtype=float).reshape(-1).shape[0]
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        for local_idx, global_idx in enumerate(self.state_indices):
            Q[global_idx, global_idx] += self.weight
            q[global_idx] += -self.weight * self.goal[local_idx]

        R = {i: np.zeros((np.asarray(u).reshape(-1).shape[0], np.asarray(u).reshape(-1).shape[0])) for i, u in enumerate(us)}
        r = {i: np.zeros(np.asarray(u).reshape(-1).shape[0]) for i, u in enumerate(us)}
        return QuadraticApprox(Q=Q, q=q, R=R, r=r, S={}, const=0.5 * self.weight * float(self.goal @ self.goal))


class ControlEffortCost(BaseCost):
    """
    Quadratic effort cost on one player's control.
    """

    def __init__(self, player_index: int, weight: float = 1.0) -> None:
        self.player_index = int(player_index)
        self.weight = float(weight)

    def evaluate(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> float:
        _ = x, k
        ui = np.asarray(us[self.player_index], dtype=float).reshape(-1)
        return 0.5 * self.weight * float(ui @ ui)

    def quadraticize(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> QuadraticApprox:
        nx = np.asarray(x, dtype=float).reshape(-1).shape[0]
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        R = {}
        r = {}
        for i, u in enumerate(us):
            nu_i = np.asarray(u).reshape(-1).shape[0]
            if i == self.player_index:
                R[i] = self.weight * np.eye(nu_i)
            else:
                R[i] = np.zeros((nu_i, nu_i), dtype=float)
            r[i] = np.zeros(nu_i, dtype=float)
        return QuadraticApprox(Q=Q, q=q, R=R, r=r, S={}, const=0.0)


class PairwiseDistanceCost(BaseCost):
    """
    Quadratic spacing cost between two scalar state components:
        0.5 * w * ((x[i] - x[j]) - d_ref)^2
    """

    def __init__(self, idx_a: int, idx_b: int, distance_ref: float = 0.0, weight: float = 1.0) -> None:
        self.idx_a = int(idx_a)
        self.idx_b = int(idx_b)
        self.distance_ref = float(distance_ref)
        self.weight = float(weight)

    def evaluate(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> float:
        _ = us, k
        xv = np.asarray(x, dtype=float).reshape(-1)
        d = (xv[self.idx_a] - xv[self.idx_b]) - self.distance_ref
        return 0.5 * self.weight * float(d * d)

    def quadraticize(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> QuadraticApprox:
        nx = np.asarray(x, dtype=float).reshape(-1).shape[0]
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        w = self.weight
        i = self.idx_a
        j = self.idx_b
        Q[i, i] += w
        Q[j, j] += w
        Q[i, j] += -w
        Q[j, i] += -w
        q[i] += -w * self.distance_ref
        q[j] += w * self.distance_ref

        R = {k: np.zeros((np.asarray(u).reshape(-1).shape[0], np.asarray(u).reshape(-1).shape[0])) for k, u in enumerate(us)}
        r = {k: np.zeros(np.asarray(u).reshape(-1).shape[0]) for k, u in enumerate(us)}
        return QuadraticApprox(Q=Q, q=q, R=R, r=r, S={}, const=0.5 * w * self.distance_ref * self.distance_ref)


class CollisionAvoidanceCost(BaseCost):
    """
    Smooth proximity penalty between two agents in 2D:
        w * exp(-||p_a - p_b||^2 / sigma^2)
    """

    def __init__(
        self,
        idx_ax: int,
        idx_ay: int,
        idx_bx: int,
        idx_by: int,
        weight: float = 1.0,
        sigma: float = 2.0,
        terminal: bool = False,
    ) -> None:
        self.idx_ax = int(idx_ax)
        self.idx_ay = int(idx_ay)
        self.idx_bx = int(idx_bx)
        self.idx_by = int(idx_by)
        self.weight = float(weight)
        self.sigma = float(sigma)
        self.terminal = bool(terminal)
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0.")

    def evaluate(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> float:
        _ = us, k
        xv = np.asarray(x, dtype=float).reshape(-1)
        dx = xv[self.idx_ax] - xv[self.idx_bx]
        dy = xv[self.idx_ay] - xv[self.idx_by]
        d2 = dx * dx + dy * dy
        return float(self.weight * np.exp(-d2 / (self.sigma * self.sigma)))

    def quadraticize(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> QuadraticApprox:
        # Iterative solver currently uses finite differences directly on evaluate(),
        # so we provide a zero placeholder here.
        nx = np.asarray(x, dtype=float).reshape(-1).shape[0]
        Q = np.zeros((nx, nx), dtype=float)
        q = np.zeros(nx, dtype=float)
        R = {i: np.zeros((np.asarray(u).reshape(-1).shape[0], np.asarray(u).reshape(-1).shape[0])) for i, u in enumerate(us)}
        r = {i: np.zeros(np.asarray(u).reshape(-1).shape[0]) for i, u in enumerate(us)}
        return QuadraticApprox(Q=Q, q=q, R=R, r=r, S={}, const=0.0)
