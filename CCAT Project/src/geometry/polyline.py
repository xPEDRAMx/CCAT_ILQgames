"""Open polyline with O(N) signed-distance queries.

Used by lane-center costs and by the curb / refuge-island stationary-obstacle
penalty (CCAT slide 17, the term ``v_i^2 * exp(-eta_s * d_curb)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .point import Point


@dataclass
class Polyline:
    points: List[Point]

    @classmethod
    def from_xy(cls, xy: Sequence[Tuple[float, float]]) -> "Polyline":
        return cls([Point(x, y) for x, y in xy])

    def signed_distance(self, p: Point) -> Tuple[float, Point]:
        """Return (signed distance, nearest point on polyline) to ``p``.

        Sign is positive if ``p`` lies to the left of the polyline's
        direction of travel and negative on the right (consistent with
        ilqgames' ``Polyline2::ClosestPoint``).
        """

        q = p.as_array()
        best_d2 = np.inf
        best_proj = q.copy()
        best_sign = 1.0

        for a, b in zip(self.points[:-1], self.points[1:]):
            pa = a.as_array()
            pb = b.as_array()
            seg = pb - pa
            seg_len2 = float(seg @ seg)
            if seg_len2 < 1e-12:
                proj = pa
            else:
                t = float((q - pa) @ seg) / seg_len2
                t = max(0.0, min(1.0, t))
                proj = pa + t * seg

            diff = q - proj
            d2 = float(diff @ diff)
            if d2 < best_d2:
                best_d2 = d2
                best_proj = proj
                # 2D cross product (sign)
                cross = seg[0] * (q[1] - pa[1]) - seg[1] * (q[0] - pa[0])
                best_sign = 1.0 if cross >= 0.0 else -1.0

        d = best_sign * float(np.sqrt(best_d2))
        return d, Point(best_proj[0], best_proj[1])

    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.array([p.x for p in self.points])
        ys = np.array([p.y for p in self.points])
        return xs, ys
