"""Minimal top-down trajectory plotter for multi-player intersection scenes."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def plot_topdown(
    xs: Sequence[np.ndarray],
    position_indices: Sequence[Tuple[int, int]],
    labels: Optional[Sequence[str]] = None,
    title: str = "iLQ-Games trajectory",
    polylines: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    goals: Optional[Sequence[Tuple[float, float]]] = None,
    ax=None,
):
    """Render each player's (x, y) trajectory on a top-down view.

    Args:
        xs: list of length T+1 of joint state vectors.
        position_indices: per-player (px_idx, py_idx) tuples.
        labels: optional player labels.
        polylines: optional list of (xs, ys) arrays representing curbs etc.
        goals: optional list of per-player goal (x, y) markers.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    n_players = len(position_indices)
    labels = list(labels) if labels is not None else [f"player {i+1}" for i in range(n_players)]

    if polylines:
        for xs_p, ys_p in polylines:
            ax.plot(xs_p, ys_p, color="0.55", lw=1.2, ls="--", alpha=0.8)

    colors = plt.cm.tab10.colors
    for i, (px, py) in enumerate(position_indices):
        traj = np.array([[x[px], x[py]] for x in xs])
        ax.plot(traj[:, 0], traj[:, 1], lw=2.0, color=colors[i % 10], label=labels[i])
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i % 10], marker="o", s=40, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i % 10], marker="X", s=60, zorder=5)

    if goals:
        for i, (gx, gy) in enumerate(goals):
            ax.scatter(gx, gy, marker="*", s=140, color=colors[i % 10],
                       edgecolor="k", zorder=6)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    return ax
