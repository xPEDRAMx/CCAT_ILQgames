"""Iterative LQ Game (ILQGames) outer loop.

Algorithm (Fridovich-Keil et al., ICRA 2020):

    1.  Roll out the current strategies through the *nonlinear* dynamics
        starting from ``x0``. This produces an operating point
        ``xi^k = (xs, u_1, ..., u_N)``.
    2.  Linearize the dynamics about ``xi^k`` and quadraticize each player's
        cost about ``xi^k``.
    3.  Solve the resulting time-varying finite-horizon LQ Nash game (in
        feedback or open-loop form) for new strategies.
    4.  Backtracking line search over the damping ``eta in (0, 1]`` on the
        affine update so that the new operating point's cost does not
        increase.
    5.  Stop when ||xs_new - xs_old|| < state_tol or when the change in the
        joint cost is below ``cost_tol``.

The line search scales ``alphas`` (and only ``alphas``, never ``Ps``), as in
the C++ reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..costs.player_cost import PlayerCost
from ..dynamics.concatenated import ConcatenatedSystem
from ..utils.types import (
    LinearDynamics,
    OperatingPoint,
    QuadraticCostApprox,
    Strategy,
)
from .lq_feedback import solve_lq_feedback
from .lq_open_loop import solve_lq_open_loop


@dataclass
class ILQParams:
    horizon: int = 50
    max_iters: int = 50
    state_tol: float = 1e-3
    cost_tol: float = 1e-4
    line_search_initial: float = 1.0
    line_search_decay: float = 0.5
    line_search_min: float = 1.0 / 64.0
    equilibrium: str = "feedback"  # "feedback" or "open_loop"
    # Add tiny diagonal regularization to make the LQ solve numerically stable
    # even when proximity / exponential costs produce indefinite Hessians.
    state_regularization: float = 1e-3
    control_regularization: float = 1e-3


class ILQSolver:
    """Outer iLQ loop tying together dynamics, costs, and an LQ Nash solver."""

    def __init__(
        self,
        dynamics: ConcatenatedSystem,
        player_costs: List[PlayerCost],
        x0: np.ndarray,
        params: Optional[ILQParams] = None,
        callback: Optional[Callable[[int, OperatingPoint, float], None]] = None,
    ) -> None:
        self.dynamics = dynamics
        self.player_costs = list(player_costs)
        self.num_players = len(player_costs)
        if dynamics.num_players != self.num_players:
            raise ValueError("Dynamics and player_costs disagree on number of players.")
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.params = params or ILQParams()
        self.callback = callback

        if self.params.equilibrium not in ("feedback", "open_loop"):
            raise ValueError(
                f"equilibrium must be 'feedback' or 'open_loop' (got {self.params.equilibrium!r})."
            )

        # Initial strategies — zero feedback, zero feedforward.
        self.strategies: List[Strategy] = [
            Strategy.zero(self.params.horizon, dynamics.x_dim, dynamics.u_dims[i])
            for i in range(self.num_players)
        ]
        self.operating_point: OperatingPoint = self._zero_operating_point()

    # ------------------------------------------------------------------
    def _zero_operating_point(self) -> OperatingPoint:
        T = self.params.horizon
        xs = [self.x0.copy() for _ in range(T + 1)]
        us = [
            [np.zeros(self.dynamics.u_dims[i]) for i in range(self.num_players)]
            for _ in range(T)
        ]
        return OperatingPoint(xs=xs, us=us, t0=0.0)

    # ------------------------------------------------------------------
    def _rollout(
        self,
        last_op: OperatingPoint,
        strategies: List[Strategy],
        scale: float,
    ) -> OperatingPoint:
        """Roll out ``last_op`` under the perturbed feedback law

            u_{i,k} = u_ref_{i,k} - P_{i,k} (x_k - x_ref_k) - scale * alpha_{i,k}.
        """

        T = self.params.horizon
        xs: List[np.ndarray] = [self.x0.copy()]
        us: List[List[np.ndarray]] = []

        for k in range(T):
            x = xs[k]
            dx = x - last_op.xs[k]
            uk: List[np.ndarray] = []
            for i in range(self.num_players):
                u_ref_i = last_op.us[k][i]
                P_i = strategies[i].Ps[k]
                a_i = strategies[i].alphas[k]
                uk.append(u_ref_i - P_i @ dx - scale * a_i)
            us.append(uk)
            xs.append(self.dynamics.integrate(x, uk))

        return OperatingPoint(xs=xs, us=us, t0=last_op.t0)

    # ------------------------------------------------------------------
    def _total_costs(self, op: OperatingPoint) -> List[float]:
        T = self.params.horizon
        totals = [0.0] * self.num_players
        for k in range(T):
            for i in range(self.num_players):
                totals[i] += self.player_costs[i].value(op.xs[k], op.us[k], k)
        # Terminal time step: treat us as zero (only state cost matters).
        zero_us = [np.zeros(self.dynamics.u_dims[i]) for i in range(self.num_players)]
        for i in range(self.num_players):
            totals[i] += self.player_costs[i].value(op.xs[T], zero_us, T)
        return totals

    # ------------------------------------------------------------------
    def _linearize(self, op: OperatingPoint) -> List[LinearDynamics]:
        T = self.params.horizon
        out: List[LinearDynamics] = []
        for k in range(T):
            A, Bs = self.dynamics.linearize_discrete(op.xs[k], op.us[k])
            out.append(LinearDynamics(A=A, B=Bs))
        # Pad final step with the last linearization (unused but matches list length).
        out.append(out[-1])
        return out

    def _regularize(self, q: QuadraticCostApprox) -> QuadraticCostApprox:
        n = q.Q.shape[0]
        Q = q.Q + self.params.state_regularization * np.eye(n)
        Q = 0.5 * (Q + Q.T)
        R = {
            p: 0.5 * (Rp + Rp.T) + self.params.control_regularization * np.eye(Rp.shape[0])
            for p, Rp in q.R.items()
        }
        return QuadraticCostApprox(Q=Q, l=q.l, R=R, r=q.r)

    def _quadraticize(
        self, op: OperatingPoint
    ) -> List[List[QuadraticCostApprox]]:
        T = self.params.horizon
        out: List[List[QuadraticCostApprox]] = []
        for k in range(T):
            row = [
                self._regularize(
                    self.player_costs[i].quadraticize(op.xs[k], op.us[k], k)
                )
                for i in range(self.num_players)
            ]
            out.append(row)
        # Final step: state-only quadratic, evaluated at terminal x with zero us.
        zero_us = [np.zeros(self.dynamics.u_dims[i]) for i in range(self.num_players)]
        row = [
            self._regularize(
                self.player_costs[i].quadraticize(op.xs[T], zero_us, T)
            )
            for i in range(self.num_players)
        ]
        out.append(row)
        return out

    # ------------------------------------------------------------------
    def solve(self) -> Tuple[OperatingPoint, List[Strategy], List[List[float]]]:
        """Run iLQ to convergence; return final ``(operating_point, strategies, cost_log)``."""

        # Initial roll-out at scale 0 (so each player just keeps zero input).
        self.operating_point = self._rollout(self.operating_point, self.strategies, scale=0.0)

        cost_log: List[List[float]] = []
        last_costs = self._total_costs(self.operating_point)
        cost_log.append(last_costs)
        if self.callback is not None:
            self.callback(0, self.operating_point, sum(last_costs))

        for it in range(1, self.params.max_iters + 1):
            lin = self._linearize(self.operating_point)
            quad = self._quadraticize(self.operating_point)

            if self.params.equilibrium == "feedback":
                new_strategies = solve_lq_feedback(lin, quad)
            else:
                new_strategies = solve_lq_open_loop(
                    lin, quad, np.zeros(self.dynamics.x_dim)
                )

            scale = self.params.line_search_initial
            improved = False
            new_op: Optional[OperatingPoint] = None
            new_costs: Optional[List[float]] = None
            last_finite_op: Optional[OperatingPoint] = None
            last_finite_costs: Optional[List[float]] = None
            while scale >= self.params.line_search_min:
                candidate_op = self._rollout(
                    self.operating_point, new_strategies, scale=scale
                )
                if not _is_finite_op(candidate_op):
                    scale *= self.params.line_search_decay
                    continue
                candidate_costs = self._total_costs(candidate_op)
                last_finite_op = candidate_op
                last_finite_costs = candidate_costs
                if sum(candidate_costs) < sum(last_costs) + 1e-9:
                    new_op = candidate_op
                    new_costs = candidate_costs
                    improved = True
                    break
                scale *= self.params.line_search_decay

            if not improved:
                if last_finite_op is None:
                    # Every candidate step blew up. Keep the previous
                    # operating point and bail out — further iterations
                    # would reuse the same diverging strategies.
                    break
                # Take the smallest finite step as a fallback.
                new_op = last_finite_op
                new_costs = last_finite_costs

            assert new_op is not None and new_costs is not None
            state_change = max(
                float(np.linalg.norm(new_op.xs[k] - self.operating_point.xs[k]))
                for k in range(self.params.horizon + 1)
            )
            cost_change = abs(sum(new_costs) - sum(last_costs))

            self.strategies = new_strategies
            self.operating_point = new_op
            last_costs = new_costs
            cost_log.append(new_costs)

            if self.callback is not None:
                self.callback(it, self.operating_point, sum(last_costs))

            if state_change < self.params.state_tol and cost_change < self.params.cost_tol:
                break

        return self.operating_point, self.strategies, cost_log


def _is_finite_op(op: OperatingPoint) -> bool:
    for x in op.xs:
        if not np.all(np.isfinite(x)):
            return False
    for uk in op.us:
        for u in uk:
            if not np.all(np.isfinite(u)):
                return False
    return True
