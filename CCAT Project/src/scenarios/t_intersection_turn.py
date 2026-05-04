"""Permitted left-turn at a T-intersection conflicting with through traffic.

Slide 13 of the deck: an LT vehicle wants to turn across opposing through
traffic. We model this as two players: the LT vehicle and the opposing
through vehicle. The LT vehicle's goal is on the far cross-street, forcing
it to yield or accelerate to clear the conflict point.
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
        AgentParams(nominal_v=4.0, v_max=7.0, proximity_weight=35.0,
                    lat_accel_weight=1.5),
        AgentParams(nominal_v=5.5, v_max=9.0, proximity_weight=35.0,
                    lat_accel_weight=2.0),
    ]

    # Player 1 starts in the LT lane (heading north) and needs to reach the
    # west cross-street goal. Player 2 is the opposing through vehicle
    # (heading south) on the same approach.
    x0 = make_initial_state(
        [
            (1.5, -16.0, np.pi / 2.0, 4.0),
            (-1.5, 16.0, -np.pi / 2.0, 5.5),
        ]
    )
    goals = [(-15.0, 5.0), (-1.5, -16.0)]
    pcs = [build_player_cost(i, num_players, horizon, goals[i], agents[i], agents) for i in range(num_players)]
    pos_idx = [(state_indices(i)[0], state_indices(i)[1]) for i in range(num_players)]

    polylines = [
        (np.array([1.5, 1.5]), np.array([-22.0, 22.0])),
        (np.array([-1.5, -1.5]), np.array([-22.0, 22.0])),
        (np.array([-22.0, 22.0]), np.array([5.0, 5.0])),
    ]

    return Scenario(
        key="t_intersection_turn",
        title="T-intersection: permitted left turn vs through traffic",
        description="A left-turning vehicle must cross the path of an opposing through vehicle (slide 13).",
        dynamics=dynamics,
        player_costs=pcs,
        x0=x0,
        goals=goals,
        labels=["P1 (left-turn)", "P2 (through, opposing)"],
        position_indices=pos_idx,
        horizon=horizon,
        dt=dt,
        line_search_initial=0.1,
        plot_lims=(-22.0, 22.0, -22.0, 22.0),
        polylines=polylines,
        agent_lengths=[4.0, 4.0],
    )


register(_factory)
