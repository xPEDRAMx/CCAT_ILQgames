"""2D point with a couple of vector helpers used by costs and plots."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Point:
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def norm(self) -> float:
        return float(np.hypot(self.x, self.y))
