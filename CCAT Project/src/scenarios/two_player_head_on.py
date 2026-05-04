"""Two vehicles approaching each other on a shared two-lane road.

Both want to traverse the corridor in opposite directions; with a slightly
offset initial lateral position, the iLQ-Nash solver finds a smooth swerve
that keeps them apart while staying close to their respective lane centers.
"""

from __future__ import annotations

import numpy as np

from ..dynamics import ConcatenatedSystem, Unicycle4D
from ._common import AgentParams, build_player_cost, make_initial_state, state_indices
from .registry import Scenario, register


def _factory() -> Scenario:
    dt = 0.1
    horizon = 60
    num_players = 2

    dynamics = ConcatenatedSystem([Unicycle4D(dt=dt) for _ in range(num_players)], dt=dt)
    agents = [
        AgentParams(nominal_v=5.0, v_max=8.0, lat_accel_weight=2.0, proximity_weight=40.0),
        AgentParams(nominal_v=5.0, v_max=8.0, lat_accel_weight=2.0, proximity_weight=40.0),
    ]

    # Same lane (y = 0), heading toward each other along x.
    # Slight initial lateral offset to break symmetry and trigger swerving.
    x0 = make_initial_state(
        [
            (-22.0, 0.5, 0.0, 5.0),
            (22.0, -0.5, np.pi, 5.0),
        ]
    )
    goals = [(22.0, 0.5), (-22.0, -0.5)]
    pcs = [build_player_cost(i, num_players, horizon, goals[i], agents[i], agents) for i in range(num_players)]
    pos_idx = [(state_indices(i)[0], state_indices(i)[1]) for i in range(num_players)]

    polylines = [
        (np.array([-25.0, 25.0]), np.array([0.5, 0.5])),
        (np.array([-25.0, 25.0]), np.array([-0.5, -0.5])),
        (np.array([-25.0, 25.0]), np.array([0.0, 0.0])),
    ]

    return Scenario(
        key="two_player_head_on",
        title="Two-player head-on encounter",
        description="Two vehicles travel in opposite directions on a two-lane road and must negotiate a safe pass.",
        dynamics=dynamics,
        player_costs=pcs,
        x0=x0,
        goals=goals,
        labels=["P1 (east-bound)", "P2 (west-bound)"],
        position_indices=pos_idx,
        horizon=horizon,
        dt=dt,
        line_search_initial=0.1,
        plot_lims=(-26.0, 26.0, -10.0, 10.0),
        polylines=polylines,
        agent_lengths=[4.0, 4.0],
    )


register(_factory)
