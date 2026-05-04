# CCAT — ILQ Games for Complete-Corridor Intersections

A Python implementation of the iterative linear-quadratic (iLQ) differential-game
solver used in the CCAT project on multi-user (HV / AV / pedestrian / cyclist)
interactions at complete-corridor intersections.

The installable Python package lives under the **`src/`** directory (import as
`src`, e.g. `from src.solver import ILQSolver`).

## Mathematical formulation

Per-agent state and control (**slides 15–16 of 36**, *Problem Definitions* —
state/control variables and dynamics):

$$X_i(t) = [p_x^i,\; p_y^i,\; \theta_i,\; v_i]^\top, \qquad U_i(t) = [\kappa_i,\; a_i]^\top$$

with dynamics (**same slides**)

$$\dot X_i = \big[v_i\cos\theta_i,\; v_i\sin\theta_i,\; v_i\kappa_i,\; a_i\big]^\top.$$

Per-agent running + terminal cost (**slides 15–16 of 36** for the structure of
$L$ and $\psi$; **slide 16 of 36** labels the stationary vs moving-obstacle
pieces). The deck writes a **maximization**; we state the equivalent
**minimization** here:

$$J_i = \tfrac{b}{2}\lVert X_i(T) - X_i^d\rVert^2 + \int_0^T\Big[\beta_1 + \tfrac{\beta_2}{2}\kappa_i^2 v_i^4 + \tfrac{\beta_3}{2}a_i^2 + \tfrac{\beta_4}{2}v_i^2 e^{-\eta_s d_{curb,i}(t)} + \sum_{j\neq i}\tfrac{\beta_5}{2}m_i m_j v_{rel,ij}^2 e^{-\eta_m d_{ij}(t)}\Big]\,\mathrm{d}t$$

Box constraints on $a_i,\kappa_i,v_i$ (**slides 19–21 of 36**, and **slides 18–22 of 36** with the cost / Nash / PMP material):

$$a_{\min}\le a_i\le a_{\max},\quad
\kappa_{\min}\le \kappa_i\le \kappa_{\max},\quad
v_{\min}\le v_i\le v_{\max}.$$

**Generalized TTC (GTTC)** as a *post hoc* conflict metric (**slide 17 of 36**)
is implemented in `src/safety/gttc.py`; it is **not** one of the additive terms
inside $L$ on slides 15–16 (those are the exponential curb / moving-obstacle
costs).

**Open-loop Nash** definition and HV–HV **rolling horizon** (**slide 20 of 36**);
open-loop Nash + **Pontryagin** necessary conditions (**slides 21–22 of 36**).

## Algorithm (iLQ games)

The outer loop follows **slides 23–24 of 36** (*Solution Method* — nonlinear
game, LQ approximation, rollout / linearize / quadraticize, damping
$\eta\in[0,1]$). Each inner step solves an LQ game:

- **Open-loop LQ Nash** — **slide 25 of 36** (*Appendix A*); implemented in
  `src/solver/lq_open_loop.py`.
- **Feedback LQ Nash** — **slide 26 of 36** (*Appendix B*); implemented in
  `src/solver/lq_feedback.py` (same discrete-time structure as Başar & Olsder
  Corollary 6.1, cited on those slides).

Steps:

1. Roll out current strategies → operating point $\xi^k = (\bar x_k, \bar u_{1:N,k})$ (**slide 24 of 36**).
2. Linearize dynamics about $\xi^k$ → $A_k, B_{i,k}$ (**slide 24 of 36**).
3. Quadraticize each player's running and terminal cost about $\xi^k$ →
   $Q_{i,k}, l_{i,k}, R_{ij,k}, r_{ij,k}$ (**slide 24 of 36**).
4. Solve the finite-horizon **LQ Nash game** (**slides 25–26 of 36**; see above).
5. Backtracking line-search on $\eta\in(0,1]$ (**slide 24 of 36**).
6. Repeat until convergence.

## Slide deck ↔ code mapping

| Topics | Primary location in this repo |
|----------------------------------------------------------------|--------------------------------|
| 12–14 — problem figures (LT, merge, car-following) | `src/scenarios/*.py` (qualitative scenarios only) |
| 15–16 — $X_i$, $U_i$, $\dot X_i=f(X_i,U_i)$, running cost $L$, terminal $\psi$ | `src/dynamics/unicycle_4d.py`, `src/costs/*.py` |
| 16 — curb vs moving-obstacle terms inside $L$ | `src/costs/ccat.py` (`ExponentialPolylineDistanceCost`, `ExponentialProximityCost`, `CurvatureCost`) |
| 17 — GTTC definition & algorithm | `src/safety/gttc.py` |
| 18–19 — single-player cost + constraints + PMP sketch | `src/examples/` (tutorial-style); constraints soft in `src/costs/semiquadratic.py` until AL is added |
| 20 — multi-player game, HV–HV open-loop + rolling horizon | `src/solver/ilq.py` (`equilibrium="open_loop"`), `src/ui/app.py` |
| 21–22 — open-loop Nash + PMP / Hamiltonian system | Equilibrium choice in `ilq.py`; hard PMP solve not duplicated (we use iLQ instead) |
| 23–24 — nonlinear game + **ILQ** algorithm | `src/solver/ilq.py` |
| 25 — **open-loop LQ** Nash recursion | `src/solver/lq_open_loop.py` |
| 26 — **feedback LQ** Nash (Riccati-like coupled equations) | `src/solver/lq_feedback.py` |
| 27+ — LQR review appendices | No separate code (standard LQR theory) |

## Repository layout

```
README.md
CCAT Project/
├── pyproject.toml
├── src/
│   ├── utils/        # numerical types, dataclasses, RK4 integrator
│   ├── geometry/     # Point, Polyline, line-segment helpers
│   ├── dynamics/     # slides 15–16: Unicycle4D + ConcatenatedSystem
│   ├── costs/        # slides 15–16, 19–21: PlayerCost + running / terminal terms
│   ├── solver/       # slides 23–26: iLQ + LQ open-loop + LQ feedback
│   ├── horizon/      # (planned) slide 20-style rolling horizon + splicer
│   ├── safety/       # slide 17: GTTC
│   ├── viz/          # matplotlib top-down plots
│   ├── scenarios/    # registry of pre-baked CCAT test cases
│   ├── ui/           # interactive matplotlib UI (python -m src.ui)
│   └── examples/     # three_player_intersection.py (CCAT scene)
└── tests/            # parity / smoke tests
```

## Status

| Component                                                       | Status |
|            --------------------------------------               |:------:|
| `Unicycle4D` dynamics (slide 15-16)                             |   ✅    |
| `ConcatenatedSystem` (multi-player wrapper)                     |   ✅    |
| `PlayerCost` with auto-quadraticization (slide 24)              |   ✅    |
| Quadratic / semiquadratic / goal costs                          |   ⏳    |
| Exponential pairwise proximity (slide 16)                       |   ⏳    |
| LQ **feedback** Nash solver (Basar–Olsder 6.17)                 |   ✅    |
| LQ **open-loop** Nash solver (M/m/Λ recursion)                  |   ✅    |
| iLQ outer loop with backtracking line search                    |   ✅    |
| Three-player intersection example                               |   ✅    |
| Full $\kappa^2 v^4$ + curb $v^2 e^{-\eta_s d}$ in all scenarios |   ⏳    |
| Bundled scenarios registry (4 test cases)                       |   ✅    |
| Interactive UI with live animation                              |   ✅    |
| Augmented-Lagrangian box constraints                            |   ⏳    |
| Receding-horizon driver + solution splicer                      |   ⏳    |
| Generalized TTC safety metric                                   |   ⏳    |

## Quick start

```bash
cd "CCAT Project"
pip install -e .

# Interactive UI — pick a test case, pick the equilibrium, watch the agents move:
python -m src.ui

# Or run the canonical example (saves PNGs of both equilibria):
python -m src.examples.three_player_intersection

# Run the test suite:
pytest -q
```

### Bundled test cases (selectable in the UI)

| Key                         | What it models                                             |
|-----------------------------|------------------------------------------------------------|
| `three_player_intersection` | Four-leg unsignalized intersection, three converging cars  |
| `two_player_head_on`        | Two cars passing each other on a two-lane road             |
| `t_intersection_turn`       | LT vs through traffic (slide **13 of 36**)                 |
| `pedestrian_crossing`       | Vehicle vs pedestrian conflict                             |

Each test case can be run with **feedback Nash** or **open-loop Nash** from the
same UI — the live animation plays back the iLQ-Nash trajectory at real time.

## Citation

If you use this code or the problem formulation, please cite our publication:

Khakpour et al., *Understanding the interaction of vehicles within an intersection:
an optimal control approach with differential games* — (venue / year).

Other useful sources:

- Fridovich-Keil, Ratner, Peters, Dragan, Tomlin. *Efficient iterative
  linear-quadratic approximations for nonlinear multi-player general-sum
  differential games.* ICRA 2020.
- Başar & Olsder. *Dynamic Noncooperative Game Theory* (2nd ed.). SIAM, 1999.
- Engwerda. *LQ dynamic optimization and differential games.* Wiley, 2005.
