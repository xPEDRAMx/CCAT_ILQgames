"""4-leg unsignalized intersection with three vehicles converging on it.

Mirrors the canonical scene from the slide deck. P1 westbound, P2 eastbound,
P3 northbound — each must reach a goal on the far side of the intersection
without conflicting with the others.
"""

from __future__ import annotations

import numpy as np

from ..dynamics import ConcatenatedSystem, Unicycle4D
from ._common import AgentParams, build_player_cost, make_initial_state, state_indices
from .registry import Scenario, register


def _factory() -> Scenario:
    dt = 0.1
    horizon = 50
    num_players = 3

    dynamics = ConcatenatedSystem([Unicycle4D(dt=dt) for _ in range(num_players)], dt=dt)
    agents = [AgentParams() for _ in range(num_players)]

    x0 = make_initial_state(
        [
            (18.0, 2.5, np.pi, 4.0),
            (-18.0, -2.5, 0.0, 4.0),
            (0.0, -18.0, np.pi / 2, 4.0),
        ]
    )
    goals = [(-18.0, 2.5), (18.0, -2.5), (0.0, 18.0)]
    pcs = [build_player_cost(i, num_players, horizon, goals[i], agents[i], agents) for i in range(num_players)]
    pos_idx = [(state_indices(i)[0], state_indices(i)[1]) for i in range(num_players)]

    # Lane center polylines for visualization only.
    polylines = [
        (np.array([-22.0, 22.0]), np.array([2.5, 2.5])),
        (np.array([-22.0, 22.0]), np.array([-2.5, -2.5])),
        (np.array([0.0, 0.0]), np.array([-22.0, 22.0])),
    ]

    return Scenario(
        key="three_player_intersection",
        title="Three-player unsignalized intersection",
        description="Westbound, eastbound and northbound vehicles converge on a four-leg unsignalized intersection.",
        dynamics=dynamics,
        player_costs=pcs,
        x0=x0,
        goals=goals,
        labels=["P1 (west-bound)", "P2 (east-bound)", "P3 (north-bound)"],
        position_indices=pos_idx,
        horizon=horizon,
        dt=dt,
        line_search_initial=0.1,
        plot_lims=(-22.0, 22.0, -22.0, 22.0),
        polylines=polylines,
        agent_lengths=[4.0, 4.0, 4.0],
    )


register(_factory)
