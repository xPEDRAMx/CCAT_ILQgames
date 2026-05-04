"""Cost base class hierarchy.

A ``Cost`` returns a scalar value at time step ``k`` together with an analytic
gradient and Hessian with respect to its inputs. Costs are split into:

* ``StateCost``  — depends on the *joint* state vector ``x`` only.
* ``ControlCost`` — depends on a single player's control ``u_i`` only
  (this matches how ``ilqgames`` parameterizes the per-player ``R_ii`` block).

A ``PlayerCost`` aggregates a list of (cost, weight) pairs and produces the
quadratic approximation needed by the LQ solver.

For brevity each cost subclass implements:

    value(z, k)   -> scalar
    grad(z, k)    -> (d,)
    hess(z, k)    -> (d, d)

where ``z`` is whatever input the cost depends on (state vector or control
vector). Costs that depend on time only via an ``apply_after_time`` switch
short-circuit to zero outside their active window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Cost(ABC):
    """Base class for any scalar cost with analytic gradient and Hessian."""

    def __init__(self, name: str = "", apply_after_time: int = -1) -> None:
        self.name = name
        self.apply_after_time = apply_after_time

    def is_active(self, k: int) -> bool:
        return k >= self.apply_after_time

    @abstractmethod
    def value(self, z: np.ndarray, k: int = 0) -> float: ...

    @abstractmethod
    def grad(self, z: np.ndarray, k: int = 0) -> np.ndarray: ...

    @abstractmethod
    def hess(self, z: np.ndarray, k: int = 0) -> np.ndarray: ...


class StateCost(Cost):
    """Marker subclass: ``value/grad/hess`` operate on the joint state ``x``."""


class ControlCost(Cost):
    """Marker subclass: ``value/grad/hess`` operate on one player's control."""
