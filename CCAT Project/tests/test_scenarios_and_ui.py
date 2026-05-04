"""Sanity tests for the scenario registry and the UI app construction.

Runs everything headlessly (Agg backend) — never opens a window.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from src.scenarios import REGISTRY, get
from src.ui.app import solve_scenario


@pytest.mark.parametrize("key", list(REGISTRY))
@pytest.mark.parametrize("eq", ["feedback", "open_loop"])
def test_each_scenario_solves(key, eq):
    sc = get(key)
    res = solve_scenario(sc, eq, max_iters=15)
    op = res.operating_point
    assert len(op.xs) == sc.horizon + 1
    for x in op.xs:
        assert np.all(np.isfinite(x))
    assert res.iterations >= 1


def test_ui_app_constructs_and_runs_headlessly():
    from src.ui.app import _CCATApp

    app = _CCATApp()
    assert len(app._scenario_titles) == len(REGISTRY)
    assert app._selected_eq() in {"feedback", "open_loop"}
    app._on_run(None)
    assert app.last_result is not None
    assert app.last_result.iterations >= 1

    # Replay should not raise.
    app._on_replay(None)
