"""Shared cost-builder used by the bundled scenarios."""

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
from ..dynamics import Unicycle4D


@dataclass
class AgentParams:
    """Per-agent cost weights and constraints used by ``build_player_cost``."""

    nominal_v: float = 4.0
    v_min: float = 0.0
    v_max: float = 8.0
    v_bound_weight: float = 5.0
    speed_reg_weight: float = 1.0
    long_accel_weight: float = 1.0
    lat_accel_weight: float = 1.0
    eta_proximity: float = 0.5
    proximity_weight: float = 30.0
    mass: float = 1.0
    goal_running_weight: float = 0.20
    goal_terminal_weight: float = 10.0


def state_indices(player_idx: int) -> Tuple[int, int, int, int]:
    """Joint-state indices ``(px, py, theta, v)`` for ``player_idx``."""

    base = 4 * player_idx
    return base + Unicycle4D.PX, base + Unicycle4D.PY, base + Unicycle4D.THETA, base + Unicycle4D.V


def build_player_cost(
    player_idx: int,
    num_players: int,
    horizon: int,
    goal_xy: Tuple[float, float],
    agent: AgentParams,
    other_agents: List[AgentParams],
) -> PlayerCost:
    """Assemble a CCAT-style cost for one player.

    Includes:
      - speed regularization toward ``nominal_v``
      - soft v_min / v_max bounds (semiquadratic)
      - control smoothness (a^2, kappa^2)
      - per-step goal attractor + terminal goal hit
      - exponential pairwise proximity to every other player
    """

    pxi, pyi, thi, vi = state_indices(player_idx)
    n = 4 * num_players
    pc = PlayerCost(x_dim=n, u_dims=[2] * num_players)

    pc.add_state_cost(QuadraticCost(dim=vi, nominal=agent.nominal_v),
                      weight=agent.speed_reg_weight)
    pc.add_state_cost(SemiquadraticCost(dim=vi, threshold=agent.v_max,
                                        oriented_right=True),
                      weight=agent.v_bound_weight)
    pc.add_state_cost(SemiquadraticCost(dim=vi, threshold=agent.v_min,
                                        oriented_right=False),
                      weight=agent.v_bound_weight)

    pc.add_control_cost(player_idx,
                        QuadraticControlCost(dim=Unicycle4D.A),
                        weight=agent.long_accel_weight)
    pc.add_control_cost(player_idx,
                        QuadraticControlCost(dim=Unicycle4D.KAPPA),
                        weight=agent.lat_accel_weight)

    pc.add_state_cost(GoalCost(position_indices=(pxi, pyi), goal=goal_xy,
                               name="goal_running"),
                      weight=agent.goal_running_weight)
    pc.add_state_cost(GoalCost(position_indices=(pxi, pyi), goal=goal_xy,
                               name="goal_terminal", apply_after_time=horizon),
                      weight=agent.goal_terminal_weight)

    for other in range(num_players):
        if other == player_idx:
            continue
        oa = other_agents[other]
        pxj, pyj, thj, vj = state_indices(other)
        pc.add_state_cost(
            ExponentialProximityCost(
                i_indices=(pxi, pyi, thi, vi),
                j_indices=(pxj, pyj, thj, vj),
                mass_i=agent.mass,
                mass_j=oa.mass,
                eta=agent.eta_proximity,
                name=f"prox_{player_idx}_{other}",
            ),
            weight=agent.proximity_weight,
        )

    return pc


def make_initial_state(specs: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """Concatenate per-agent ``(px, py, theta, v)`` tuples into one state."""

    out = np.zeros(4 * len(specs), dtype=float)
    for i, (px, py, th, v) in enumerate(specs):
        out[4 * i + Unicycle4D.PX] = px
        out[4 * i + Unicycle4D.PY] = py
        out[4 * i + Unicycle4D.THETA] = th
        out[4 * i + Unicycle4D.V] = v
    return out
