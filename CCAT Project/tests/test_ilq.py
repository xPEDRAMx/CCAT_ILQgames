"""End-to-end smoke test: iLQ on a tiny 2-player toy converges to a sensible
trajectory."""

from __future__ import annotations

import numpy as np

from src.costs import (
    GoalCost,
    PlayerCost,
    QuadraticControlCost,
    QuadraticCost,
)
from src.dynamics import ConcatenatedSystem, Unicycle4D
from src.solver import ILQParams, ILQSolver


def _build_two_player_problem(horizon: int = 30):
    dynamics = ConcatenatedSystem([Unicycle4D(dt=0.1), Unicycle4D(dt=0.1)], dt=0.1)
    n = dynamics.x_dim

    def per_player(player_idx: int, goal):
        offset = player_idx * 4
        pc = PlayerCost(x_dim=n, u_dims=[2, 2])
        # Soft cruising-speed reg.
        pc.add_state_cost(
            QuadraticCost(dim=offset + Unicycle4D.V, nominal=4.0), weight=1.0
        )
        # Smooth controls.
        pc.add_control_cost(player_idx, QuadraticControlCost(dim=Unicycle4D.A), weight=1.0)
        pc.add_control_cost(player_idx, QuadraticControlCost(dim=Unicycle4D.KAPPA), weight=0.2)
        # Terminal goal.
        pc.add_state_cost(
            GoalCost(
                position_indices=(offset + Unicycle4D.PX, offset + Unicycle4D.PY),
                goal=goal,
                apply_after_time=horizon,
            ),
            weight=20.0,
        )
        return pc

    pcs = [per_player(0, (10.0, 0.0)), per_player(1, (-10.0, 0.0))]

    x0 = np.zeros(n)
    x0[Unicycle4D.PX] = -10.0; x0[Unicycle4D.V] = 4.0
    x0[4 + Unicycle4D.PX] = 10.0; x0[4 + Unicycle4D.THETA] = np.pi
    x0[4 + Unicycle4D.V] = 4.0
    return dynamics, pcs, x0


def test_ilq_feedback_converges():
    dyn, pcs, x0 = _build_two_player_problem(horizon=30)
    solver = ILQSolver(dyn, pcs, x0=x0, params=ILQParams(horizon=30, max_iters=20))
    op, _, log = solver.solve()
    assert len(op.xs) == 31
    final_total = sum(log[-1])
    initial_total = sum(log[0])
    # iLQ should not increase the joint cost.
    assert final_total <= initial_total + 1e-6


def test_ilq_open_loop_runs():
    dyn, pcs, x0 = _build_two_player_problem(horizon=20)
    solver = ILQSolver(
        dyn, pcs, x0=x0,
        params=ILQParams(horizon=20, max_iters=10, equilibrium="open_loop"),
    )
    op, _, log = solver.solve()
    assert len(op.xs) == 21
    assert np.all(np.isfinite(op.xs[-1]))
