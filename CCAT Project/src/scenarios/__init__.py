"""Registry of pre-baked CCAT test scenarios.

Each scenario is a callable that returns a :class:`Scenario` describing a
multi-agent ilqgames problem. The interactive UI in ``ccat.ui`` consumes the
registry to expose each scenario as a one-click test case.
"""

from .registry import REGISTRY, Scenario, get
from . import three_player_intersection  # noqa: F401  (registers itself)
from . import two_player_head_on  # noqa: F401
from . import t_intersection_turn  # noqa: F401
from . import pedestrian_crossing  # noqa: F401

__all__ = ["REGISTRY", "Scenario", "get"]
