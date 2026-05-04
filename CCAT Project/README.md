# CCAT — ILQ Games for Complete-Corridor Intersections

A Python implementation of the iterative linear-quadratic (iLQ) differential-game
solver used in the CCAT project on multi-user (HV / AV / pedestrian / cyclist)
interactions at complete-corridor intersections.

## Mathematical formulation

Per-agent state and control (slide 16/19):

$$X_i(t) = [p_x^i,\; p_y^i,\; \theta_i,\; v_i]^\top, \qquad U_i(t) = [\kappa_i,\; a_i]^\top$$

with dynamics

$$\dot X_i = \big[\,v_i\cos\theta_i,\; v_i\sin\theta_i,\; v_i\kappa_i,\; a_i\,\big]^\top.$$

Per-agent cost (slide 17, sign-flipped to a minimization):

$$J_i = \tfrac{b}{2}\lVert X_i(T) - X_i^d\rVert^2 + \int_0^T\!\Big[\beta_1 + \tfrac{\beta_2}{2}\kappa_i^2 v_i^4 + \tfrac{\beta_3}{2}a_i^2 + \tfrac{\beta_4}{2}v_i^2 e^{-\eta_s d_{curb,i}(t)} + \sum_{j\neq i}\tfrac{\beta_5}{2}m_i m_j v_{rel,ij}^2 e^{-\eta_m d_{ij}(t)}\Big]\,dt$$

subject to the box constraints $a_{\min}\le a_i\le a_{\max}$,
$\kappa_{\min}\le \kappa_i\le \kappa_{\max}$,
$v_{\min}\le v_i\le v_{\max}$.

Strategies are sought as time-varying state-feedback (or open-loop) Nash
equilibria. The open-loop variant is used for HV–HV interactions inside a
receding-horizon loop.

## Algorithm

1. Roll out current strategies → operating point $\xi^k = (\bar x_k, \bar u_{1:N,k})$.
2. Linearize dynamics about $\xi^k$ → $A_k, B_{i,k}$.
3. Quadraticize each player's running and terminal cost about $\xi^k$ →
   $Q_{i,k}, l_{i,k}, R_{ij,k}, r_{ij,k}$.
4. Solve the resulting finite-horizon **LQ Nash game** in closed form
   (feedback or open-loop).
5. Backtracking line-search on $\eta\in(0,1]$ over the affine update.
6. Repeat until convergence.

## Repository layout

```
CCAT Project/
├── README.md
├── pyproject.toml
├── ccat/
│   ├── utils/        # numerical types, dataclasses, RK4 integrator
│   ├── geometry/     # Point, Polyline, line-segment helpers
│   ├── dynamics/     # base + Unicycle4D + ConcatenatedSystem
│   ├── costs/        # Cost base, PlayerCost, quadratic / semiquadratic / CCAT-specific
│   ├── solver/       # LQ feedback, LQ open-loop, iLQ outer loop
│   ├── horizon/      # (planned) receding-horizon driver, solution splicer
│   ├── safety/       # GTTC safety metric
│   ├── viz/          # matplotlib top-down plots
│   ├── scenarios/    # registry of pre-baked CCAT test cases
│   ├── ui/           # interactive matplotlib UI (python -m ccat.ui)
│   └── examples/     # three_player_intersection.py (CCAT scene)
└── tests/            # parity / smoke tests
```

## Status

| Component                                        | Status |
|--------------------------------------------------|:------:|
| `Unicycle4D` dynamics (CCAT slide form)          |   ✅    |
| `ConcatenatedSystem` (multi-player wrapper)      |   ✅    |
| `PlayerCost` with auto-quadraticization          |   ✅    |
| Quadratic / semiquadratic / proximity costs      |   ✅    |
| Goal terminal cost                               |   ✅    |
| LQ **feedback** Nash solver (Basar–Olsder 6.17)  |   ✅    |
| LQ **open-loop** Nash solver (M/m/Λ recursion)   |   ✅    |
| iLQ outer loop with backtracking line search     |   ✅    |
| Three-player intersection example                |   ✅    |
| Curvature-cost $\kappa^2 v^4$, exp-proximity     |   ✅    |
| Bundled scenarios registry (4 test cases)        |   ✅    |
| Interactive UI with live animation               |   ✅    |
| Augmented-Lagrangian box constraints             |   ⏳    |
| Receding-horizon driver + solution splicer       |   ⏳    |
| Generalized TTC safety metric                    |   ✅    |

## Quick start

```bash
cd "CCAT Project"
pip install -e .

# Interactive UI — pick a test case, pick the equilibrium, watch the agents move:
python -m ccat.ui

# Or run the canonical example (saves PNGs of both equilibria):
python -m ccat.examples.three_player_intersection

# Run the test suite:
pytest -q
```

### Bundled test cases (selectable in the UI)

| Key                         | What it models                                             |
|-----------------------------|------------------------------------------------------------|
| `three_player_intersection` | Four-leg unsignalized intersection, three converging cars  |
| `two_player_head_on`        | Two cars passing each other on a two-lane road             |
| `t_intersection_turn`       | Permitted left turn vs opposing through traffic (slide 13) |
| `pedestrian_crossing`       | Vehicle yielding to a pedestrian at a midblock crossing    |

Each test case can be run with **feedback Nash** or **open-loop Nash** from the
same UI — the live animation plays back the iLQ-Nash trajectory at real time.

## References

- Fridovich-Keil, Ratner, Peters, Dragan, Tomlin. *Efficient iterative
  linear-quadratic approximations for nonlinear multi-player general-sum
  differential games.* ICRA 2020.
- Başar & Olsder. *Dynamic Noncooperative Game Theory* (2nd ed.). SIAM, 1999.
- Engwerda. *LQ dynamic optimization and differential games.* Wiley, 2005.
- Khakpour et al. *Modeling and Understanding Multi-User Interaction in
  Complete Corridor Intersections.* CCAT, 2025 (slides).
