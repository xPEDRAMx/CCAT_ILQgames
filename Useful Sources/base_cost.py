# Definition of the base structure for all of costs functions
# Note that most of the calculations are done within the players' cost this part is just structural check
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class QuadraticApprox:
    """
        Local quadratic approximation of a scalar cost.

        Cost expansion:
            l ≈ 0.5 * xᵀ Q x + qᵀ x
                + Σ_i (0.5 * u_iᵀ R_i u_i + r_iᵀ u_i)
                + Σ_{i≠j} u_iᵀ S_{ij} u_j
                + const
    """

    # State Terms
    Q: np.ndarray   # shape (nx, nx)
    q: np.ndarray   # shape (nx,)

    # Control Terms
    R: Dict[int, np.ndarray]    # R[i] shape (nu_i, nu_i)
    r: Dict[int, np.ndarray]   # r[i] shape (nu_i,)

    # Control Cross Terms
    S: Dict[Tuple[int, int], np.ndarray]    # S[(i,j)] shape (nu_i, nu_j)
    const: float= 0.0

class BaseCost:
    """
        Abstract base class for all cost terms.
    """
    def evaluate(
            self,
            x: np.ndarray,
            us: list[np.ndarray],
            k: Optional[int] = None
    ) -> float:
        """
            Compute scalar cost value.
        """
        raise NotImplementedError

    def quadraticize(
            self,
            x: np.ndarray,
            us: list[np.ndarray],
            k: Optional[int] = None
    ) -> QuadraticApprox:
        """
        Compute local quadratic approximation of the cost.
        """
        raise NotImplementedError
