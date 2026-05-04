"""Vehicle yielding to a pedestrian crossing in the conflict zone.

Slide 13 of the deck (LT conflicting with pedestrians / cyclists). We model
the pedestrian as a slow unicycle with tighter speed and acceleration
bounds. The vehicle must slow down or steer around the pedestrian.
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
    vehicle = AgentParams(
        nominal_v=5.5, v_max=9.0, proximity_weight=80.0,
        long_accel_weight=1.0, lat_accel_weight=2.0, mass=2.0, eta_proximity=0.6,
    )
    pedestrian = AgentParams(
        nominal_v=1.2, v_min=0.0, v_max=2.0, v_bound_weight=15.0,
        proximity_weight=60.0, long_accel_weight=4.0, lat_accel_weight=4.0,
        mass=0.2, eta_proximity=0.6,
        speed_reg_weight=2.0,
    )
    agents = [vehicle, pedestrian]

    # Vehicle approaches from the east on the eastbound lane (heading west)
    # toward an exit on the west side. Pedestrian crosses south-to-north
    # through the conflict point.
    x0 = make_initial_state(
        [
            (20.0, 0.0, np.pi, 5.5),
            (0.0, -8.0, np.pi / 2, 1.2),
        ]
    )
    goals = [(-20.0, 0.0), (0.0, 8.0)]
    pcs = [
        build_player_cost(0, num_players, horizon, goals[0], vehicle, agents),
        build_player_cost(1, num_players, horizon, goals[1], pedestrian, agents),
    ]
    pos_idx = [(state_indices(i)[0], state_indices(i)[1]) for i in range(num_players)]

    polylines = [
        (np.array([-25.0, 25.0]), np.array([0.0, 0.0])),
        # Crosswalk (y = 0 area) shaded by two parallel lines.
        (np.array([-2.5, -2.5]), np.array([-9.0, 9.0])),
        (np.array([2.5, 2.5]), np.array([-9.0, 9.0])),
    ]

    return Scenario(
        key="pedestrian_crossing",
        title="Vehicle vs pedestrian at a midblock crossing",
        description="A vehicle approaches a midblock crossing while a pedestrian traverses the conflict zone.",
        dynamics=dynamics,
        player_costs=pcs,
        x0=x0,
        goals=goals,
        labels=["P1 (vehicle)", "P2 (pedestrian)"],
        position_indices=pos_idx,
        horizon=horizon,
        dt=dt,
        line_search_initial=0.05,
        plot_lims=(-22.0, 22.0, -10.0, 10.0),
        polylines=polylines,
        agent_lengths=[4.0, 0.5],
    )


register(_factory)
