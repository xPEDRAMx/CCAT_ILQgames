"""Time-varying finite-horizon **open-loop** LQ Nash solver.

Direct port of ``ilqgames/src/lq_open_loop_solver.cpp``. The derivation is in
``CCAT/open_loop_lq_nash.pdf``; Basar & Olsder ch. 6 also has the textbook
version (without linear terms in the cost).

For each player ``i`` we maintain a backward recursion in two quantities:

    M_i^k   in R^{n x n}         (state Hessian of the value function)
    m_i^k   in R^n               (state gradient of the value function)

with terminal conditions ``M_i^T = Q_i^T``, ``m_i^T = l_i^T``. The recursion
uses the following intermediate per-time-step terms:

    chol_R_i = Cholesky(R_ii^k)            (one per player)
    warped_B_i = R_ii^{-1} B_i^T           (n x m_i, multiplied by Z_{k+1} below)
    warped_r_i = R_ii^{-1} r_ii            (m_i)
    Lambda_k = I + sum_i B_i warped_B_i^T M_i^{k+1}     (n x n)

then

    M_i^k = Q_i^k + A^T M_i^{k+1} Lambda^{-1} A
    m_i^k = l_i^k + A^T ( m_i^{k+1} + M_i^{k+1} Lambda^{-1} eta_k )
    eta_k  = - sum_i B_i ( warped_B_i^T m_i^{k+1} + warped_r_i )

The forward sweep then produces the affine open-loop control as

    u_i^k = - warped_B_i^T ( M_i^{k+1} x_{k+1}^* + m_i^{k+1} ) - warped_r_i

which is stored in the strategy's ``alphas[k]`` (with no feedback ``P_k``).
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..utils.types import (
    LinearDynamics,
    QuadraticCostApprox,
    Strategy,
)


def solve_lq_open_loop(
    linearization: List[LinearDynamics],
    quadraticization: List[List[QuadraticCostApprox]],
    delta_x0: np.ndarray,
) -> List[Strategy]:
    """Solve a time-varying finite-horizon LQ game in open-loop form.

    Args:
        linearization: list of length T+1 with the joint linearization at
            each time step (the last entry's dynamics are not used).
        quadraticization: ``quadraticization[k][i]`` is player ``i``'s
            quadratic cost approximation at time ``k``. Length T+1.
        delta_x0: initial perturbation of the state from the operating
            point's first state. Often zero in iLQ.

    Returns:
        A list of length ``num_players`` whose ``Ps`` are zero and whose
        ``alphas`` carry the open-loop control offsets.
    """

    horizon = len(linearization) - 1
    if horizon < 1:
        raise ValueError("Horizon must be >= 1.")
    num_players = len(quadraticization[0])
    n = linearization[0].A.shape[0]
    u_dims = [linearization[0].B[i].shape[1] for i in range(num_players)]

    # Initialize M and m for k = T.
    M_next = [quadraticization[horizon][i].Q.copy() for i in range(num_players)]
    m_next = [quadraticization[horizon][i].l.copy() for i in range(num_players)]

    # Cache per-time-step "warped" quantities and Lambda decompositions for
    # use in the forward sweep below.
    warped_Bs_per_k: List[List[np.ndarray]] = []
    warped_rs_per_k: List[List[np.ndarray]] = []
    Lambda_per_k: List[np.ndarray] = []
    eta_per_k: List[np.ndarray] = []
    M_at_kp1: List[List[np.ndarray]] = []  # for use in forward pass
    m_at_kp1: List[List[np.ndarray]] = []

    M_seq = [M_next]  # M_seq[t] for t = horizon... built at the end
    m_seq = [m_next]

    for k in range(horizon - 1, -1, -1):
        lin = linearization[k]
        quad = quadraticization[k]

        warped_Bs: List[np.ndarray] = []
        warped_rs: List[np.ndarray] = []
        Lambda = np.eye(n)
        for i in range(num_players):
            R_ii = quad[i].R[i]
            r_ii = quad[i].r[i]
            B_i = lin.B[i]
            # R_ii is m_i x m_i. Solve once and reuse for both warped_B and warped_r.
            wB_i = np.linalg.solve(R_ii, B_i.T)  # (m_i, n)
            wr_i = np.linalg.solve(R_ii, r_ii)   # (m_i,)
            warped_Bs.append(wB_i)
            warped_rs.append(wr_i)
            Lambda = Lambda + B_i @ wB_i @ M_next[i]

        # Solve Lambda^{-1} A and Lambda^{-1} eta below using a single LU.
        # Compute eta first.
        eta = np.zeros(n)
        for i in range(num_players):
            eta = eta - lin.B[i] @ (warped_Bs[i] @ m_next[i] + warped_rs[i])

        # Solve Lambda x = [A, eta] in one shot.
        rhs = np.column_stack([lin.A, eta])
        sol = np.linalg.solve(Lambda, rhs)
        Lambda_inv_A = sol[:, : lin.A.shape[1]]
        Lambda_inv_eta = sol[:, lin.A.shape[1]]

        # New M and m.
        M_curr: List[np.ndarray] = []
        m_curr: List[np.ndarray] = []
        for i in range(num_players):
            M_i = quad[i].Q + lin.A.T @ M_next[i] @ Lambda_inv_A
            M_i = 0.5 * (M_i + M_i.T)
            m_i = quad[i].l + lin.A.T @ (m_next[i] + M_next[i] @ Lambda_inv_eta)
            M_curr.append(M_i)
            m_curr.append(m_i)

        # Cache for forward pass (note these depend on k+1 quantities).
        warped_Bs_per_k.append(warped_Bs)
        warped_rs_per_k.append(warped_rs)
        Lambda_per_k.append(Lambda)
        eta_per_k.append(eta)
        M_at_kp1.append(M_next)
        m_at_kp1.append(m_next)

        M_next = M_curr
        m_next = m_curr

        M_seq.append(M_next)
        m_seq.append(m_next)

    # Reverse so they're indexed by forward-time ``k``.
    warped_Bs_per_k.reverse()
    warped_rs_per_k.reverse()
    Lambda_per_k.reverse()
    eta_per_k.reverse()
    M_at_kp1.reverse()
    m_at_kp1.reverse()

    # Forward sweep — compute optimal x and u.
    strategies = [
        Strategy.zero(horizon=horizon, x_dim=n, u_dim=u_dims[i])
        for i in range(num_players)
    ]

    x_star = delta_x0.copy()
    for k in range(horizon):
        lin = linearization[k]
        # x_{k+1}^* = Lambda_k^{-1} ( A x_k^* + eta_k )
        x_next = np.linalg.solve(Lambda_per_k[k], lin.A @ x_star + eta_per_k[k])

        for i in range(num_players):
            wB_i = warped_Bs_per_k[k][i]
            wr_i = warped_rs_per_k[k][i]
            inter = M_at_kp1[k][i] @ x_next + m_at_kp1[k][i]
            # Sign convention in iLQ: u_i = u_ref_i - P_i dx - alpha_i, with P_i = 0.
            # Here the "alpha" carries the *positive* control offset from the
            # solver; we flip the sign so that subtracting alpha yields the
            # correct increment. This matches lq_open_loop_solver.cpp.
            strategies[i].alphas[k] = wB_i @ inter + wr_i

        x_star = x_next

    return strategies
