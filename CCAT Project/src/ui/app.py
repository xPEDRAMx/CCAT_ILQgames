"""Interactive matplotlib UI for picking a scenario and watching the agents
move under the iLQ-Nash trajectory.

Layout::

    +--------------------------------+--------------+
    |                                | Scenario     |
    |                                |  o A         |
    |                                |  o B         |
    |       Top-down animation       |  o C         |
    |                                |              |
    |                                | Equilibrium  |
    |                                |  o feedback  |
    |                                |  o open-loop |
    |                                |              |
    |                                |  [  Run  ]   |
    |                                |  [ Replay ]  |
    +--------------------------------+--------------+
    | status bar (iters, cost, GTTC)                |
    +-----------------------------------------------+

The animation re-uses the iLQ trajectory ``op.xs`` and steps through it at
real time (``dt`` per frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..safety import gttc
from ..scenarios import REGISTRY, Scenario, get
from ..solver import ILQParams, ILQSolver
from ..utils.types import OperatingPoint


# ---------------------------------------------------------------------------
# Solve a scenario and return the rolled-out trajectory + metadata.
# ---------------------------------------------------------------------------
@dataclass
class SolveResult:
    scenario: Scenario
    operating_point: OperatingPoint
    iterations: int
    final_costs: List[float]
    min_gttc: float
    worst_pair: Optional[Tuple[int, int]]


def _min_pairwise_gttc(scenario: Scenario, op: OperatingPoint):
    pairs = []
    n = len(scenario.position_indices)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    best = float("inf")
    worst_pair = None
    Li = scenario.agent_lengths or [0.0] * n
    for i, j in pairs:
        pxi, pyi = scenario.position_indices[i]
        pxj, pyj = scenario.position_indices[j]
        thi, vi = pxi + 2, pxi + 3
        thj, vj = pxj + 2, pxj + 3
        for x in op.xs:
            tt, _, _ = gttc(
                pos_i=(x[pxi], x[pyi]),
                vel_i=(x[vi] * np.cos(x[thi]), x[vi] * np.sin(x[thi])),
                pos_j=(x[pxj], x[pyj]),
                vel_j=(x[vj] * np.cos(x[thj]), x[vj] * np.sin(x[thj])),
                length_i=Li[i],
                length_j=Li[j],
            )
            if 0.0 < tt < best:
                best = tt
                worst_pair = (i, j)
    return best, worst_pair


def solve_scenario(scenario: Scenario, equilibrium: str, max_iters: int = 40) -> SolveResult:
    solver = ILQSolver(
        scenario.dynamics,
        scenario.player_costs,
        x0=scenario.x0,
        params=ILQParams(
            horizon=scenario.horizon,
            max_iters=max_iters,
            equilibrium=equilibrium,
            line_search_initial=scenario.line_search_initial,
            line_search_min=1.0 / 1024.0,
        ),
    )
    op, _, log = solver.solve()
    min_gttc, worst_pair = _min_pairwise_gttc(scenario, op)
    return SolveResult(
        scenario=scenario,
        operating_point=op,
        iterations=len(log) - 1,
        final_costs=log[-1],
        min_gttc=min_gttc,
        worst_pair=worst_pair,
    )


# ---------------------------------------------------------------------------
# UI.
# ---------------------------------------------------------------------------
class _CCATApp:
    """Stateful container for the matplotlib widgets and animation."""

    def __init__(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons

        self.plt = plt
        self.scenario_keys = list(REGISTRY.keys())

        # Map keys -> human titles for the radio buttons (preserving order).
        scenarios = [get(k) for k in self.scenario_keys]
        self._scenario_titles = [sc.title for sc in scenarios]

        self.fig = plt.figure(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title("CCAT iLQ-Games — interactive UI")
        gs = self.fig.add_gridspec(
            nrows=8, ncols=12,
            left=0.05, right=0.97, top=0.96, bottom=0.06,
            wspace=0.4, hspace=0.6,
        )
        self.ax = self.fig.add_subplot(gs[0:7, 0:9])
        self.status_ax = self.fig.add_subplot(gs[7, 0:9])
        self.status_ax.axis("off")
        self.status_text = self.status_ax.text(
            0.0, 0.5, "Pick a scenario and click Run.", fontsize=11,
            ha="left", va="center", family="monospace",
        )

        # Scenario radio buttons.
        ax_scn = self.fig.add_subplot(gs[0:3, 9:12])
        ax_scn.set_title("Test case", fontsize=11, fontweight="bold")
        self.scenario_radio = RadioButtons(ax_scn, self._scenario_titles, active=0)
        for label in self.scenario_radio.labels:
            label.set_fontsize(9)

        # Equilibrium radio buttons.
        ax_eq = self.fig.add_subplot(gs[3:5, 9:12])
        ax_eq.set_title("Equilibrium", fontsize=11, fontweight="bold")
        self.eq_radio = RadioButtons(ax_eq, ["feedback Nash", "open-loop Nash"], active=0)
        for label in self.eq_radio.labels:
            label.set_fontsize(10)

        # Buttons.
        ax_run = self.fig.add_subplot(gs[5, 9:12])
        ax_replay = self.fig.add_subplot(gs[6, 9:12])
        self.run_btn = Button(ax_run, "Run iLQ + animate", color="0.85", hovercolor="0.95")
        self.replay_btn = Button(ax_replay, "Replay last", color="0.9", hovercolor="0.97")
        self.run_btn.on_clicked(self._on_run)
        self.replay_btn.on_clicked(self._on_replay)

        # Animation state.
        self.last_result: Optional[SolveResult] = None
        self.anim = None

        # Render the initial scenario (static).
        self._render_static_scene(get(self.scenario_keys[0]))

    # ----- helpers ---------------------------------------------------------
    def _selected_key(self) -> str:
        idx = self._scenario_titles.index(self.scenario_radio.value_selected)
        return self.scenario_keys[idx]

    def _selected_eq(self) -> str:
        return "feedback" if self.eq_radio.value_selected.startswith("feedback") else "open_loop"

    def _set_status(self, text: str) -> None:
        self.status_text.set_text(text)
        self.fig.canvas.draw_idle()

    # ----- scene rendering -------------------------------------------------
    def _render_static_scene(self, scenario: Scenario) -> None:
        self.ax.clear()
        xmin, xmax, ymin, ymax = scenario.plot_lims
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(scenario.title, fontsize=12, fontweight="bold")

        # Polylines / lane references.
        for xs_p, ys_p in scenario.polylines:
            self.ax.plot(xs_p, ys_p, color="0.55", lw=1.3, ls="--", alpha=0.85)

        # Initial positions (filled circles), goals (stars).
        cmap = self.plt.cm.tab10.colors
        for i, (px_idx, py_idx) in enumerate(scenario.position_indices):
            self.ax.scatter(scenario.x0[px_idx], scenario.x0[py_idx],
                            color=cmap[i % 10], s=70, zorder=4,
                            edgecolor="k", linewidth=0.8)
        for i, (gx, gy) in enumerate(scenario.goals):
            self.ax.scatter(gx, gy, marker="*", s=180, color=cmap[i % 10],
                            edgecolor="k", linewidth=0.8, zorder=5)
        self.fig.canvas.draw_idle()

    # ----- callbacks -------------------------------------------------------
    def _on_run(self, _event) -> None:
        # Stop any in-flight animation before kicking off a new solve.
        self._stop_animation()
        key = self._selected_key()
        eq = self._selected_eq()
        scenario = get(key)
        self._render_static_scene(scenario)
        self._set_status(f"Solving '{key}' with {eq} Nash ...")
        self.fig.canvas.start_event_loop(0.05)

        try:
            result = solve_scenario(scenario, eq)
        except Exception as exc:  # surface the error in the UI rather than crash
            self._set_status(f"ERROR: {exc!r}")
            return

        self.last_result = result
        self._summarize_result(result)
        self._animate(result)

    def _on_replay(self, _event) -> None:
        if self.last_result is None:
            self._set_status("Nothing to replay yet — click Run first.")
            return
        self._stop_animation()
        self._render_static_scene(self.last_result.scenario)
        self._summarize_result(self.last_result)
        self._animate(self.last_result)

    def _summarize_result(self, result: SolveResult) -> None:
        sc = result.scenario
        n = len(sc.position_indices)
        per_player = []
        for i in range(n):
            px_idx, py_idx = sc.position_indices[i]
            x_T = result.operating_point.xs[-1]
            d = float(np.hypot(x_T[px_idx] - sc.goals[i][0], x_T[py_idx] - sc.goals[i][1]))
            per_player.append(f"P{i+1}: J={result.final_costs[i]:7.1f}  d_goal={d:5.2f} m")
        gttc_str = (
            f"min GTTC={result.min_gttc:5.2f} s on pair {result.worst_pair}"
            if result.worst_pair is not None else "no positive GTTC"
        )
        self._set_status(
            f"iters={result.iterations:>3}  |  " + "  |  ".join(per_player) + f"  |  {gttc_str}"
        )

    # ----- animation -------------------------------------------------------
    def _stop_animation(self) -> None:
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self.anim = None

    def _animate(self, result: SolveResult) -> None:
        from matplotlib.animation import FuncAnimation

        sc = result.scenario
        op = result.operating_point
        n_players = len(sc.position_indices)
        cmap = self.plt.cm.tab10.colors

        # Track lines (full trajectory faded) + markers (current position).
        full_xy = []
        for i, (pxi, pyi) in enumerate(sc.position_indices):
            traj = np.array([[x[pxi], x[pyi]] for x in op.xs])
            full_xy.append(traj)
            self.ax.plot(traj[:, 0], traj[:, 1], color=cmap[i % 10], lw=1.0,
                         alpha=0.20, zorder=2)

        trails = []
        markers = []
        labels = sc.labels if sc.labels else [f"P{i+1}" for i in range(n_players)]
        for i in range(n_players):
            (line,) = self.ax.plot([], [], color=cmap[i % 10], lw=2.4,
                                   alpha=0.9, zorder=3, label=labels[i])
            trails.append(line)
            (mk,) = self.ax.plot([], [], "o", color=cmap[i % 10], markersize=10,
                                 markeredgecolor="k", zorder=6)
            markers.append(mk)
        self.ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

        time_text = self.ax.text(
            0.99, 0.01, "", transform=self.ax.transAxes, ha="right", va="bottom",
            fontsize=10, family="monospace",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.6"},
        )

        T = sc.horizon

        def init():
            for line in trails:
                line.set_data([], [])
            for mk in markers:
                mk.set_data([], [])
            time_text.set_text("")
            return [*trails, *markers, time_text]

        def update(frame: int):
            for i, line in enumerate(trails):
                line.set_data(full_xy[i][: frame + 1, 0], full_xy[i][: frame + 1, 1])
            for i, mk in enumerate(markers):
                mk.set_data([full_xy[i][frame, 0]], [full_xy[i][frame, 1]])
            time_text.set_text(f"t = {frame * sc.dt:5.2f} s   step {frame:>3}/{T}")
            return [*trails, *markers, time_text]

        interval_ms = max(20, int(1000 * sc.dt))
        self.anim = FuncAnimation(
            self.fig, update, frames=T + 1, init_func=init,
            interval=interval_ms, blit=False, repeat=False,
        )
        self.fig.canvas.draw_idle()


def launch() -> None:
    """Open the UI window. Blocks until the user closes it."""

    import matplotlib.pyplot as plt

    app = _CCATApp()
    plt.show()
    return app  # returned for testability
