"""Three-player unsignalized-intersection scene from the CCAT slides.

Three vehicles (kinematic unicycles) approach a four-leg intersection from
different directions and must reach goals on the opposite side. Each player
optimizes a CCAT-style cost (slide 16 / 17):

    travel-time + lateral-accel smoothness (curvature term)
                + longitudinal-accel smoothness (a^2)
                + soft speed bound + terminal goal cost
                + exponential pairwise proximity (vehicle-vehicle)

The example uses the **feedback** Nash equilibrium solver by default.
Switch to ``equilibrium="open_loop"`` in :class:`ILQParams` to mimic the
HV-HV setting from slide 22.

Run it as

    python -m src.examples.three_player_intersection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..costs import (
    ExponentialProximityCost,
    GoalCost,
    PlayerCost,
    QuadraticControlCost,
    QuadraticCost,
    SemiquadraticCost,
)
from ..dynamics import ConcatenatedSystem, Unicycle4D
from ..solver import ILQParams, ILQSolver
from ..safety import gttc


# ---------- problem setup ----------------------------------------------------

DT = 0.1
HORIZON = 50          # 5 s @ 10 Hz
NUM_PLAYERS = 3
PER_PLAYER_X = 4
PER_PLAYER_U = 2

# Joint-state slot indices for each player.
PXI = [0, 4, 8]
PYI = [1, 5, 9]
THI = [2, 6, 10]
VI  = [3, 7, 11]


@dataclass
class CCATWeights:
    """CCAT slide-16 cost weights, per-player.

    The goal cost is split into a small per-step running attractor plus a
    larger terminal hit so that the LQ approximation is well-conditioned at
    every backward-Riccati step (matching how the deprecated upstream
    Python example uses negatively-weighted ``ProximityCost`` toward the
    goal).
    """

    beta_lat: float = 1.0
    beta_long: float = 1.0
    beta_v: float = 1.0
    nominal_v: float = 4.0
    v_min: float = 0.0
    v_max: float = 8.0
    v_bound_weight: float = 5.0
    eta_m: float = 0.5
    beta_m: float = 30.0
    goal_running_weight: float = 0.20
    goal_terminal_weight: float = 10.0


def _player_cost(
    player_idx: int,
    goal_xy: Tuple[float, float],
    weights: CCATWeights,
) -> PlayerCost:
    pc = PlayerCost(x_dim=NUM_PLAYERS * PER_PLAYER_X, u_dims=[PER_PLAYER_U] * NUM_PLAYERS)

    # Travel-time / nominal speed regularization (slide's beta1 + speed term):
    #   penalize deviation from a nominal cruising speed.
    pc.add_state_cost(
        QuadraticCost(dim=VI[player_idx], nominal=weights.nominal_v, name="speed_reg"),
        weight=weights.beta_v,
    )

    # Speed bounds (soft).
    pc.add_state_cost(
        SemiquadraticCost(
            dim=VI[player_idx], threshold=weights.v_max, oriented_right=True,
            name="v_max",
        ),
        weight=weights.v_bound_weight,
    )
    pc.add_state_cost(
        SemiquadraticCost(
            dim=VI[player_idx], threshold=weights.v_min, oriented_right=False,
            name="v_min",
        ),
        weight=weights.v_bound_weight,
    )

    # Longitudinal-accel smoothness: 0.5 * a^2.
    pc.add_control_cost(
        player_idx,
        QuadraticControlCost(dim=Unicycle4D.A, nominal=0.0, name="long_accel"),
        weight=weights.beta_long,
    )

    # Lateral-accel surrogate: 0.5 * kappa^2 (a simple proxy that quadraticizes
    # cleanly, in lieu of the full kappa^2 v^4 which mixes state and control;
    # the full term is available in src.costs.ccat.CurvatureCost for use with
    # an Augmented-Lagrangian-style outer loop).
    pc.add_control_cost(
        player_idx,
        QuadraticControlCost(dim=Unicycle4D.KAPPA, nominal=0.0, name="lat_accel_surrogate"),
        weight=weights.beta_lat,
    )

    # Per-step goal attractor (active at every step). This keeps the
    # backward-Riccati state cost well-conditioned.
    pc.add_state_cost(
        GoalCost(
            position_indices=(PXI[player_idx], PYI[player_idx]),
            goal=goal_xy,
            name="goal_running",
        ),
        weight=weights.goal_running_weight,
    )

    # Terminal goal cost (extra hit on the last time step only).
    pc.add_state_cost(
        GoalCost(
            position_indices=(PXI[player_idx], PYI[player_idx]),
            goal=goal_xy,
            name="goal_terminal",
            apply_after_time=HORIZON,
        ),
        weight=weights.goal_terminal_weight,
    )

    # Pairwise proximity to the other agents (slide 17 moving-obstacle term).
    for other in range(NUM_PLAYERS):
        if other == player_idx:
            continue
        pc.add_state_cost(
            ExponentialProximityCost(
                i_indices=(PXI[player_idx], PYI[player_idx], THI[player_idx], VI[player_idx]),
                j_indices=(PXI[other], PYI[other], THI[other], VI[other]),
                mass_i=1.0,
                mass_j=1.0,
                eta=weights.eta_m,
                name=f"prox_{player_idx}_{other}",
            ),
            weight=weights.beta_m,
        )

    return pc


def build_problem(weights: CCATWeights = CCATWeights()):
    dynamics = ConcatenatedSystem(
        subsystems=[Unicycle4D(dt=DT) for _ in range(NUM_PLAYERS)],
        dt=DT,
    )

    # Initial conditions:
    #   P1: westbound (coming from +x, heading -x).
    #   P2: eastbound (coming from -x, heading +x).
    #   P3: northbound (coming from -y, heading +y).
    x0 = np.zeros(NUM_PLAYERS * PER_PLAYER_X)
    # Player 1
    x0[PXI[0]] = 18.0; x0[PYI[0]] = 2.5
    x0[THI[0]] = np.pi; x0[VI[0]] = 4.0
    # Player 2
    x0[PXI[1]] = -18.0; x0[PYI[1]] = -2.5
    x0[THI[1]] = 0.0;   x0[VI[1]] = 4.0
    # Player 3
    x0[PXI[2]] = 0.0; x0[PYI[2]] = -18.0
    x0[THI[2]] = np.pi / 2; x0[VI[2]] = 4.0

    goals = [(-18.0, 2.5), (18.0, -2.5), (0.0, 18.0)]
    player_costs = [_player_cost(i, goals[i], weights) for i in range(NUM_PLAYERS)]

    return dynamics, player_costs, x0, goals


# ---------- entry point ------------------------------------------------------
def main(plot: bool = True, equilibrium: str = "feedback") -> None:
    dynamics, player_costs, x0, goals = build_problem()

    params = ILQParams(
        horizon=HORIZON,
        max_iters=40,
        equilibrium=equilibrium,
        line_search_initial=0.1,
        line_search_decay=0.5,
        line_search_min=1.0 / 1024.0,
    )
    solver = ILQSolver(dynamics, player_costs, x0=x0, params=params)
    op, strategies, cost_log = solver.solve()

    final_costs = cost_log[-1]
    print(f"\nThree-player intersection ({equilibrium} Nash, {len(cost_log)-1} iters):")
    for i, c in enumerate(final_costs):
        gx, gy = goals[i]
        x_T = op.xs[-1]
        d_to_goal = float(np.hypot(x_T[PXI[i]] - gx, x_T[PYI[i]] - gy))
        print(f"  player {i+1}: J = {c:9.3f},  ||x_T - goal|| = {d_to_goal:6.3f} m")

    # Minimum pairwise GTTC over the horizon.
    pairs = [(0, 1), (0, 2), (1, 2)]
    min_gttc = float("inf")
    worst_pair = None
    for (i, j) in pairs:
        for x in op.xs:
            tt, _, _ = gttc(
                pos_i=(x[PXI[i]], x[PYI[i]]),
                vel_i=(x[VI[i]] * np.cos(x[THI[i]]), x[VI[i]] * np.sin(x[THI[i]])),
                pos_j=(x[PXI[j]], x[PYI[j]]),
                vel_j=(x[VI[j]] * np.cos(x[THI[j]]), x[VI[j]] * np.sin(x[THI[j]])),
                length_i=4.0, length_j=4.0,
            )
            if 0.0 < tt < min_gttc:
                min_gttc = tt
                worst_pair = (i, j)
    if worst_pair is not None:
        print(f"  worst GTTC = {min_gttc:.2f} s on pair {worst_pair}")
    else:
        print("  no positive GTTC encountered (no approaching conflicts).")

    if plot:
        try:
            import matplotlib.pyplot as plt
            from ..viz import plot_topdown
        except ImportError:
            return
        ax = plot_topdown(
            op.xs,
            position_indices=[(PXI[i], PYI[i]) for i in range(NUM_PLAYERS)],
            labels=[f"P{i+1}" for i in range(NUM_PLAYERS)],
            title=f"Three-player intersection ({equilibrium} Nash)",
            goals=goals,
        )
        out = f"three_player_intersection_{equilibrium}.png"
        plt.savefig(out, dpi=140, bbox_inches="tight")
        print(f"  trajectory plot saved to {out}")
        if not plt.get_backend().lower().startswith("agg"):
            plt.show()


if __name__ == "__main__":
    main()
