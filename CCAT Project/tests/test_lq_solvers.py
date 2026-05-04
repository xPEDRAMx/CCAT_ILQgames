"""Smoke tests for the LQ Nash solvers.

We verify on a trivial 1-player LQR-as-a-game problem that:

* the feedback solver reproduces the standard discrete-time LQR (Riccati)
  gain to high precision;
* the open-loop solver drives the state to zero from a perturbed init when
  there is no input cost interaction with other players.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.solver.lq_feedback import solve_lq_feedback
from src.solver.lq_open_loop import solve_lq_open_loop
from src.utils.types import LinearDynamics, QuadraticCostApprox


def _double_integrator_lin(T: int, dt: float = 0.1):
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.0], [dt]])
    return [LinearDynamics(A=A, B=[B]) for _ in range(T + 1)]


def _double_integrator_cost(T: int, q: float = 1.0, r: float = 0.1):
    Q = np.diag([q, q])
    R = np.array([[r]])
    return [
        [QuadraticCostApprox(Q=Q, l=np.zeros(2), R={0: R}, r={0: np.zeros(1)})]
        for _ in range(T + 1)
    ]


def test_feedback_lqr_single_player_riccati():
    """Single-player feedback Nash collapses to LQR; check gain matches Riccati."""
    T = 50
    lin = _double_integrator_lin(T)
    quad = _double_integrator_cost(T)

    strategies = solve_lq_feedback(lin, quad)
    P0 = strategies[0].Ps[0]

    # Discrete-time Riccati for double integrator with same Q, R.
    from scipy.linalg import solve_discrete_are
    A = lin[0].A
    B = lin[0].B[0]
    Q = quad[0][0].Q
    R = quad[0][0].R[0]
    S = solve_discrete_are(A, B, Q, R)
    K_lqr = np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)

    # Feedback gain at k = 0 should be close to the steady-state Riccati gain
    # for sufficiently long horizons.
    assert P0.shape == (1, 2)
    assert np.allclose(P0, K_lqr, atol=5e-3)


def test_open_loop_drives_state_to_zero():
    """1 player, double integrator, init [1, 0]: open-loop should pull x to 0."""
    T = 30
    lin = _double_integrator_lin(T)
    quad = _double_integrator_cost(T, q=10.0, r=0.01)
    x0 = np.array([1.0, 0.0])

    strategies = solve_lq_open_loop(lin, quad, delta_x0=x0)

    # Roll forward under the open-loop alphas (P is zero for open-loop).
    x = x0.copy()
    for k in range(T):
        u = -strategies[0].alphas[k]  # iLQ convention: u = -alpha when P=0 and u_ref=0
        x = lin[k].A @ x + lin[k].B[0] @ u

    # With high state weight and low control weight the state should be near 0.
    assert np.linalg.norm(x) < 0.2


def test_open_loop_returns_zero_feedback_gains():
    T = 5
    lin = _double_integrator_lin(T)
    quad = _double_integrator_cost(T)
    strategies = solve_lq_open_loop(lin, quad, delta_x0=np.zeros(2))
    for k in range(T):
        assert np.allclose(strategies[0].Ps[k], 0.0)
