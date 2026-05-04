"""Lightweight scenario registry consumed by the UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..costs import PlayerCost
from ..dynamics import ConcatenatedSystem


@dataclass
class Scenario:
    """A test case for the iLQ-games solver."""

    key: str
    title: str
    description: str
    dynamics: ConcatenatedSystem
    player_costs: List[PlayerCost]
    x0: np.ndarray
    goals: List[Tuple[float, float]]
    labels: List[str]
    position_indices: List[Tuple[int, int]]
    horizon: int
    dt: float
    line_search_initial: float = 0.1
    plot_lims: Tuple[float, float, float, float] = (-25, 25, -25, 25)
    polylines: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    agent_lengths: List[float] = field(default_factory=list)


REGISTRY: Dict[str, Callable[[], Scenario]] = {}


def register(factory: Callable[[], Scenario]) -> Callable[[], Scenario]:
    """Decorator that registers a scenario factory by its returned ``key``."""

    sc = factory()
    if sc.key in REGISTRY:
        raise ValueError(f"Scenario key {sc.key!r} already registered.")
    REGISTRY[sc.key] = factory
    return factory


def get(key: str) -> Scenario:
    if key not in REGISTRY:
        raise KeyError(f"Unknown scenario {key!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[key]()
