from typing import List, Optional
import numpy as np

from costs.base_cost import BaseCost, QuadraticApprox


class PlayerCost:
    """
        Container for all cost terms of a single player.
    """

    def __init__(self, name: str):
        self.name = name
        self.cost_term: List[BaseCost] = []  # Store cost terms implementing BaseCost interface.

    def add_cost(self, cost: BaseCost):  # Costs should always be in the same format of the Base Cost
        """
            Add a cost term (state cost, control cost, proximity cost, etc.).
        """
        self.cost_term.append(cost)

    def evaluate_stage(self, x: np.ndarray, us: List[np.ndarray], k: Optional[int] = None) -> float:
        """Evaluate all non-terminal terms."""
        total = 0.0
        for c in self.cost_term:
            if bool(getattr(c, "terminal", False)):
                continue
            total += float(c.evaluate(x, us, k=k))
        return total

    def evaluate_terminal(self, x: np.ndarray, us: Optional[List[np.ndarray]] = None, k: Optional[int] = None) -> float:
        """Evaluate all terminal terms."""
        if us is None:
            us = []
        total = 0.0
        for c in self.cost_term:
            if not bool(getattr(c, "terminal", False)):
                continue
            total += float(c.evaluate(x, us, k=k))
        return total


