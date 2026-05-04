"""Generalized Time-To-Collision (GTTC).

Slide 18 of the CCAT presentation. For two agents at positions ``c_i, c_j``
with velocities ``v_i, v_j`` (both 2D Cartesian) and effective lengths
``L_i, L_j``, define

    r          = c_i - c_j                                        (m)
    v_rel      = v_i - v_j                                        (m/s)
    d_ij       = ||r||                                            (m)
    d_dot      = (r^T v_rel) / d_ij                               (m/s)
    d_adjusted = d_ij - 0.5 (L_i + L_j)                           (m)
    GTTC       = - d_adjusted / d_dot                             (s)

Negative GTTC means the agents are moving apart. A "conflict" is declared
when ``0 < GTTC < threshold``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def gttc(
    pos_i: Sequence[float],
    vel_i: Sequence[float],
    pos_j: Sequence[float],
    vel_j: Sequence[float],
    length_i: float = 0.0,
    length_j: float = 0.0,
    eps: float = 1e-9,
) -> Tuple[float, float, float]:
    """Return ``(gttc, d_adjusted, d_dot)`` for one pair of agents.

    ``gttc`` is positive when agents are approaching, negative when moving
    apart, and ``+inf`` when ``|d_dot| < eps``.
    """

    r = np.asarray(pos_i, dtype=float) - np.asarray(pos_j, dtype=float)
    vrel = np.asarray(vel_i, dtype=float) - np.asarray(vel_j, dtype=float)
    d = float(np.linalg.norm(r))
    if d < eps:
        return float("inf"), 0.0, 0.0
    d_dot = float(r @ vrel) / d
    d_adj = d - 0.5 * (float(length_i) + float(length_j))
    if abs(d_dot) < eps:
        return float("inf"), d_adj, d_dot
    return -d_adj / d_dot, d_adj, d_dot


def conflict_mask(
    xs,  # iterable of length T+1 of joint state vectors
    pos_indices: Sequence[Tuple[int, int]],
    vel_indices: Sequence[Tuple[int, int]],
    pair: Tuple[int, int],
    threshold: float = 1.5,
    lengths: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """Return a boolean array marking time steps where the GTTC for one pair
    of agents falls below ``threshold`` (i.e. a conflict)."""

    i, j = pair
    pxi, pyi = pos_indices[i]
    vxi, vyi = vel_indices[i]
    pxj, pyj = pos_indices[j]
    vxj, vyj = vel_indices[j]
    out = []
    for x in xs:
        t, _, _ = gttc(
            pos_i=(x[pxi], x[pyi]),
            vel_i=(x[vxi], x[vyi]),
            pos_j=(x[pxj], x[pyj]),
            vel_j=(x[vxj], x[vyj]),
            length_i=lengths[0],
            length_j=lengths[1],
        )
        out.append(0.0 < t < threshold)
    return np.array(out, dtype=bool)
