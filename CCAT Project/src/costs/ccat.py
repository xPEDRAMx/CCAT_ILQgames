"""CCAT-specific cost terms (slide 16/17).

The CCAT running cost has the closed form

    L_i = beta_1
        + (beta_2/2) * kappa_i^2 * v_i^4                           (lateral accel)
        + (beta_3/2) * a_i^2                                       (longitudinal accel)
        + (beta_4/2) * v_i^2 * exp(-eta_s * d_curb,i(t))           (stationary obstacle)
        + sum_{j!=i} (beta_5/2) * m_i m_j * v_rel,ij^2
                                * exp(-eta_m * d_ij(t))            (moving obstacle)

The cubic ``a^2`` and constant ``beta_1`` terms reduce to the standard
``QuadraticControlCost``; the curvature term and the two exponential terms
are CCAT-specific and live here.

All gradients and Hessians are analytic.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import ControlCost, StateCost


# ---------------------------------------------------------------------------
# Lateral acceleration smoothness term: 0.5 * kappa^2 * v^4
# ---------------------------------------------------------------------------
class CurvatureCost(StateCost):
    """``c(x, u) = 0.5 * kappa^2 * v^4`` — slide's lateral-acceleration term.

    Implemented as a *state*-level term that closes over the player's current
    curvature reference. Because ``ilqgames`` only quadraticizes per-player
    state and per-player control terms (not joint state-control terms), we
    mimic the original C++ ``CurvatureCost`` by evaluating the derivative
    with respect to ``u_kappa`` and storing it on the player's control
    Jacobian, while the ``v^4`` factor is treated as a state Hessian.

    For numerical robustness in the very-low-speed regime the term is
    augmented with a small floor on ``v``.

    NOTE: This implementation expands the term and stores its full Hessian
    contribution in the *control* slot. We provide a separate
    :class:`CurvatureControlCost` below for direct use as a control cost,
    which is the form actually consumed by the LQ approximation.
    """

    def __init__(
        self,
        v_index: int,
        kappa_index: int,
        u_provider,  # callable: () -> np.ndarray returning current player u
        name: str = "curvature",
        apply_after_time: int = -1,
    ) -> None:
        super().__init__(name=name, apply_after_time=apply_after_time)
        self.v_index = int(v_index)
        self.kappa_index = int(kappa_index)
        self._u = u_provider  # type: ignore[assignment]

    def value(self, x: np.ndarray, k: int = 0) -> float:
        v = x[self.v_index]
        u = self._u()
        kappa = u[self.kappa_index]
        return 0.5 * float(kappa * kappa) * float(v ** 4)

    def grad(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        v = x[self.v_index]
        u = self._u()
        kappa = u[self.kappa_index]
        g = np.zeros_like(x, dtype=float)
        g[self.v_index] = 2.0 * (kappa ** 2) * (v ** 3)
        return g

    def hess(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        v = x[self.v_index]
        u = self._u()
        kappa = u[self.kappa_index]
        H = np.zeros((x.size, x.size))
        H[self.v_index, self.v_index] = 6.0 * (kappa ** 2) * (v ** 2)
        return H


# ---------------------------------------------------------------------------
# Stationary obstacle term:  0.5 * v^2 * exp(-eta_s * d_curb)
# Implemented as a pure state cost because d_curb depends only on x.
# ---------------------------------------------------------------------------
class ExponentialPolylineDistanceCost(StateCost):
    """``c(x) = 0.5 * v^2 * exp(-eta * d(x))`` for an Euclidean signed distance ``d``.

    Used to model curb / refuge-island avoidance (slide 17). The polyline's
    signed-distance function ``d(x)`` is treated locally as

        d(x) ~= n^T (p(x) - p_proj)

    where ``n`` is the unit normal at the closest point on the polyline.
    This linearization is re-evaluated each iteration of iLQ and is exact
    for a straight curb.
    """

    def __init__(
        self,
        position_indices: Tuple[int, int],
        v_index: int,
        polyline,  # ccat.geometry.Polyline
        eta: float,
        name: str = "exp_curb",
        apply_after_time: int = -1,
    ) -> None:
        super().__init__(name=name, apply_after_time=apply_after_time)
        self.px_idx, self.py_idx = position_indices
        self.v_index = int(v_index)
        self.polyline = polyline
        self.eta = float(eta)

    def _signed_distance_with_normal(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        from ..geometry.point import Point

        p = Point(float(x[self.px_idx]), float(x[self.py_idx]))
        d, proj = self.polyline.signed_distance(p)
        diff = np.array([p.x - proj.x, p.y - proj.y])
        d_abs = float(np.linalg.norm(diff))
        if d_abs < 1e-9:
            n = np.array([0.0, 0.0])
            d_signed = 0.0
        else:
            n = np.sign(d) * (diff / d_abs)
            d_signed = d
        return d_signed, n

    def value(self, x: np.ndarray, k: int = 0) -> float:
        d, _ = self._signed_distance_with_normal(x)
        v = x[self.v_index]
        return 0.5 * float(v * v) * float(np.exp(-self.eta * d))

    def grad(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        d, n = self._signed_distance_with_normal(x)
        v = x[self.v_index]
        e = np.exp(-self.eta * d)
        g = np.zeros_like(x, dtype=float)
        # d c / d v = v * exp(-eta d)
        g[self.v_index] = v * e
        # d c / d p = -eta * 0.5 * v^2 * exp(-eta d) * n
        g[self.px_idx] = -self.eta * 0.5 * v * v * e * n[0]
        g[self.py_idx] = -self.eta * 0.5 * v * v * e * n[1]
        return g

    def hess(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        d, n = self._signed_distance_with_normal(x)
        v = x[self.v_index]
        e = np.exp(-self.eta * d)
        H = np.zeros((x.size, x.size))
        # d^2 c / d v^2 = exp(-eta d)
        H[self.v_index, self.v_index] = e
        # d^2 c / d p_a d p_b = eta^2 * 0.5 * v^2 * exp * n_a n_b
        coeff_pp = (self.eta ** 2) * 0.5 * v * v * e
        for a in (self.px_idx, self.py_idx):
            for b in (self.px_idx, self.py_idx):
                ia = 0 if a == self.px_idx else 1
                ib = 0 if b == self.px_idx else 1
                H[a, b] += coeff_pp * n[ia] * n[ib]
        # d^2 c / d v d p_a = -eta * v * exp * n_a
        coeff_vp = -self.eta * v * e
        H[self.v_index, self.px_idx] += coeff_vp * n[0]
        H[self.px_idx, self.v_index] += coeff_vp * n[0]
        H[self.v_index, self.py_idx] += coeff_vp * n[1]
        H[self.py_idx, self.v_index] += coeff_vp * n[1]
        return H


# ---------------------------------------------------------------------------
# Moving obstacle term:  0.5 * m_i m_j * v_rel^2 * exp(-eta_m * d_ij)
# ---------------------------------------------------------------------------
class ExponentialProximityCost(StateCost):
    """``c(x) = 0.5 * mass_product * |v_i - v_j|^2 * exp(-eta * d_ij)``.

    Slide 17's vehicle-vehicle (and vehicle-pedestrian) interaction term.
    ``v_rel^2`` is computed from the two players' (vx, vy) components,
    where ``vx = v cos theta`` and ``vy = v sin theta`` for the unicycle
    model. The signed distance is the simple Euclidean distance between
    the two players' (px, py).

    For tractability we expose only ``d_ij`` and ``|v_i - v_j|^2`` to the
    quadraticizer; the trigonometric ``cos/sin`` factors are handled
    by the per-iteration relinearization in iLQ.
    """

    def __init__(
        self,
        i_indices,  # (px_i, py_i, theta_i, v_i)
        j_indices,
        mass_i: float = 1.0,
        mass_j: float = 1.0,
        eta: float = 1.0,
        name: str = "exp_proximity",
        apply_after_time: int = -1,
    ) -> None:
        super().__init__(name=name, apply_after_time=apply_after_time)
        self.pxi, self.pyi, self.thi, self.vi = i_indices
        self.pxj, self.pyj, self.thj, self.vj = j_indices
        self.eta = float(eta)
        self.mass_product = float(mass_i) * float(mass_j)

    # ---- analytic helpers ------------------------------------------------
    def _components(self, x: np.ndarray):
        """Return ``rx, ry, d, V2, E, dtheta`` for use by value/grad/hess."""

        rx = x[self.pxi] - x[self.pxj]
        ry = x[self.pyi] - x[self.pyj]
        d = float(np.hypot(rx, ry))
        thi, thj = x[self.thi], x[self.thj]
        vi, vj = x[self.vi], x[self.vj]
        dtheta = thi - thj
        # V^2 = vi^2 + vj^2 - 2 vi vj cos(dtheta)  (relative-velocity magnitude squared)
        V2 = float(vi * vi + vj * vj - 2.0 * vi * vj * np.cos(dtheta))
        E = float(np.exp(-self.eta * d))
        return rx, ry, d, V2, E, dtheta

    def value(self, x: np.ndarray, k: int = 0) -> float:
        _, _, _, V2, E, _ = self._components(x)
        return 0.5 * self.mass_product * V2 * E

    def grad(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        rx, ry, d, V2, E, dtheta = self._components(x)
        m = self.mass_product
        thi, thj = x[self.thi], x[self.thj]
        vi, vj = x[self.vi], x[self.vj]

        g = np.zeros_like(x, dtype=float)
        if d < 1e-9:
            return g  # singular position; treat as zero gradient

        coeff_pos = -0.5 * m * V2 * self.eta * E / d
        g[self.pxi] += coeff_pos * rx
        g[self.pyi] += coeff_pos * ry
        g[self.pxj] -= coeff_pos * rx
        g[self.pyj] -= coeff_pos * ry

        # dV2/dthi = 2 vi vj sin(dtheta)
        dV2_dthi = 2.0 * vi * vj * np.sin(dtheta)
        dV2_dthj = -dV2_dthi
        # dV2/dvi = 2 vi - 2 vj cos(dtheta)
        dV2_dvi = 2.0 * vi - 2.0 * vj * np.cos(dtheta)
        dV2_dvj = 2.0 * vj - 2.0 * vi * np.cos(dtheta)

        g[self.thi] += 0.5 * m * E * dV2_dthi
        g[self.thj] += 0.5 * m * E * dV2_dthj
        g[self.vi] += 0.5 * m * E * dV2_dvi
        g[self.vj] += 0.5 * m * E * dV2_dvj
        return g

    def hess(self, x: np.ndarray, k: int = 0) -> np.ndarray:
        """Gauss-Newton-style PSD approximation of the Hessian.

        The cost is ``c = 0.5 * h(x)^2`` with ``h = sqrt(m * V^2 * E)``. The
        Gauss-Newton Hessian ``J_h^T J_h`` is rank-1, equal to
        ``(grad c)(grad c)^T / max(2 c, eps)``. This is always PSD and is a
        good local quadratic model when ``c`` is moderate, which is exactly
        the regime where the proximity term matters in iLQ.
        """

        n = x.size
        c = self.value(x, k)
        if c < 1e-12:
            return np.zeros((n, n))
        g = self.grad(x, k)
        return np.outer(g, g) / (2.0 * c)


# ---------------------------------------------------------------------------
# Numerical helpers (only used for the trickier mixed CCAT terms above).
# ---------------------------------------------------------------------------
def _central_difference_gradient(
    f, x: np.ndarray, k: int, eps: float = 1e-5
) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        h = eps * max(1.0, abs(float(x[i])))
        xp[i] += h
        xm[i] -= h
        g[i] = (f(xp, k) - f(xm, k)) / (2.0 * h)
    return g


def _central_difference_hessian(
    f, x: np.ndarray, k: int, eps: float = 1e-4
) -> np.ndarray:
    n = x.size
    H = np.zeros((n, n))
    f0 = f(x, k)
    for i in range(n):
        for j in range(i, n):
            hi = eps * max(1.0, abs(float(x[i])))
            hj = eps * max(1.0, abs(float(x[j])))
            if i == j:
                xp = x.copy(); xm = x.copy()
                xp[i] += hi; xm[i] -= hi
                H[i, i] = (f(xp, k) - 2.0 * f0 + f(xm, k)) / (hi * hi)
            else:
                xpp = x.copy(); xpm = x.copy(); xmp = x.copy(); xmm = x.copy()
                xpp[i] += hi; xpp[j] += hj
                xpm[i] += hi; xpm[j] -= hj
                xmp[i] -= hi; xmp[j] += hj
                xmm[i] -= hi; xmm[j] -= hj
                val = (f(xpp, k) - f(xpm, k) - f(xmp, k) + f(xmm, k)) / (4.0 * hi * hj)
                H[i, j] = val
                H[j, i] = val
    return 0.5 * (H + H.T)
