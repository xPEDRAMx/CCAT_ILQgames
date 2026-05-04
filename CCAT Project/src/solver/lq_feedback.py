"""Time-varying finite-horizon **feedback** LQ Nash solver.

Reference: Basar & Olsder, *Dynamic Noncooperative Game Theory* (2nd ed.),
Corollary 6.1, eq. 6.17 (pp. 279). The dynamics are

    dx_{k+1} = A_k dx_k + sum_i B_{i,k} du_{i,k}

and each player's running cost is

    L_{i,k}(dx, du) = 0.5 dx^T Q_{i,k} dx + l_{i,k}^T dx
                    + sum_j ( 0.5 du_j^T R_{ij,k} du_j + r_{ij,k}^T du_j ).

Returns affine state-error feedback strategies

    u_{i,k} = u_ref_{i,k} - P_{i,k} dx_k - alpha_{i,k}.

This is a numpy port of the deprecated `python/solve_lq_game.py` from the
upstream ``ilqgames`` repo, with explicit handling of the linear (gradient)
terms ``l`` and ``r`` so that an iterative-LQ outer loop sees consistent
affine updates.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..utils.types import (
    LinearDynamics,
    QuadraticCostApprox,
    Strategy,
)


def solve_lq_feedback(
    linearization: List[LinearDynamics],
    quadraticization: List[List[QuadraticCostApprox]],
) -> List[Strategy]:
    """Solve a time-varying finite-horizon LQ game in feedback form.

    Args:
        linearization: time-indexed list of length T+1 of joint
            linearizations (the last entry is unused for dynamics, but its
            quadratic cost is the terminal cost).
        quadraticization: ``quadraticization[k][i]`` is player ``i``'s
            quadratic cost approximation at time ``k``. Length T+1.

    Returns:
        A list of length ``num_players``. Element ``i`` is a ``Strategy``
        whose ``Ps`` and ``alphas`` lists have length ``T``.
    """

    horizon = len(linearization) - 1
    if horizon < 1:
        raise ValueError("Horizon must be >= 1.")
    num_players = len(quadraticization[0])
    n = linearization[0].A.shape[0]
    u_dims = [linearization[0].B[i].shape[1] for i in range(num_players)]

    # Terminal value: Z_i^{T} = Q_i^{T},   zeta_i^{T} = l_i^{T}.
    Zs = [quadraticization[horizon][i].Q.copy() for i in range(num_players)]
    zetas = [quadraticization[horizon][i].l.copy() for i in range(num_players)]

    Ps_per_player: List[List[np.ndarray]] = [[None] * horizon for _ in range(num_players)]  # type: ignore[list-item]
    alphas_per_player: List[List[np.ndarray]] = [
        [None] * horizon for _ in range(num_players)  # type: ignore[list-item]
    ]

    for k in range(horizon - 1, -1, -1):
        A = linearization[k].A
        Bs = linearization[k].B

        # Per-player Q, l (state) and {R[ij]}, {r[ij]} (control) at time k.
        Qs = [quadraticization[k][i].Q for i in range(num_players)]
        ls = [quadraticization[k][i].l for i in range(num_players)]
        Rs = [
            [quadraticization[k][i].R[j] for j in range(num_players)]
            for i in range(num_players)
        ]
        rs = [
            [quadraticization[k][i].r[j] for j in range(num_players)]
            for i in range(num_players)
        ]

        # Block linear system   S [P_1; ...; P_N] = Y   for the gains.
        # Row i is:  ( R_ii + B_i^T Z_i B_i ) P_i + sum_{j!=i} B_i^T Z_i B_j P_j = B_i^T Z_i A
        S_rows = []
        Y_rows_P = []
        for i in range(num_players):
            S_blocks = []
            for j in range(num_players):
                if i == j:
                    S_blocks.append(Rs[i][i] + Bs[i].T @ Zs[i] @ Bs[i])
                else:
                    S_blocks.append(Bs[i].T @ Zs[i] @ Bs[j])
            S_rows.append(np.concatenate(S_blocks, axis=1))
            Y_rows_P.append(Bs[i].T @ Zs[i] @ A)
        S = np.concatenate(S_rows, axis=0)
        Y_P = np.concatenate(Y_rows_P, axis=0)

        # Solve (least-squares for robustness when S is near-singular).
        P_stack, _, _, _ = np.linalg.lstsq(S, Y_P, rcond=None)
        P_split = np.split(P_stack, np.cumsum(u_dims[:-1]), axis=0)
        for i in range(num_players):
            Ps_per_player[i][k] = P_split[i]

        # F_k = A - sum_i B_i P_i        (Basar & Olsder, eq. 6.17c)
        F = A.copy()
        for i in range(num_players):
            F = F - Bs[i] @ P_split[i]

        # Same S matrix; right-hand side is the gradient terms.
        Y_rows_a = []
        for i in range(num_players):
            # B_i^T zeta_i + r_ii
            Y_rows_a.append(Bs[i].T @ zetas[i] + rs[i][i])
        Y_a = np.concatenate(Y_rows_a, axis=0)
        alpha_stack, _, _, _ = np.linalg.lstsq(S, Y_a, rcond=None)
        alpha_split = np.split(alpha_stack, np.cumsum(u_dims[:-1]), axis=0)
        for i in range(num_players):
            alphas_per_player[i][k] = alpha_split[i]

        # beta_k = -sum_i B_i alpha_i
        beta = np.zeros(n)
        for i in range(num_players):
            beta -= Bs[i] @ alpha_split[i]

        # Z_i  <- Q_i + sum_j P_j^T R_ij P_j + F^T Z_i F
        # zeta_i <- l_i + sum_j ( P_j^T R_ij alpha_j - P_j^T r_ij )
        #         + F^T ( zeta_i + Z_i beta )
        new_Zs = []
        new_zetas = []
        for i in range(num_players):
            Zi = Qs[i] + F.T @ Zs[i] @ F
            zi = ls[i] + F.T @ (zetas[i] + Zs[i] @ beta)
            for j in range(num_players):
                Zi = Zi + P_split[j].T @ Rs[i][j] @ P_split[j]
                zi = zi + P_split[j].T @ Rs[i][j] @ alpha_split[j]
                zi = zi - P_split[j].T @ rs[i][j]
            new_Zs.append(Zi)
            new_zetas.append(zi)
        Zs = new_Zs
        zetas = new_zetas

    return [
        Strategy(Ps=Ps_per_player[i], alphas=alphas_per_player[i])
        for i in range(num_players)
    ]
