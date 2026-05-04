import numpy as np
import pytest

from src.dynamics import ConcatenatedSystem, Unicycle4D


@pytest.fixture
def system():
    return Unicycle4D(dt=0.05)


def test_dynamics_zero_input_zero_speed(system):
    x = np.zeros(4)
    u = np.zeros(2)
    xnext = system.integrate(x, u)
    assert np.allclose(xnext, np.zeros(4))


def test_dynamics_pure_acceleration(system):
    """With u = [0, 1] and v0 = 0, after RK4 we should have v ~= a*dt."""
    x = np.zeros(4)
    u = np.array([0.0, 1.0])
    xnext = system.integrate(x, u)
    assert xnext[Unicycle4D.V] == pytest.approx(system.dt, rel=1e-9)


def test_linearization_finite_difference_match(system):
    """Discrete-time linearization should match a numerical Jacobian."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=4)
    u = rng.normal(size=2)

    A_an, B_an = system.linearize_discrete(x, u)

    # Numerical Jacobian of the *RK4* step (vs the Euler-linearized analytic).
    # We expect them to agree to first order in dt; check with a small dt.
    h = 1e-6
    x_next = system.integrate(x, u)

    A_num = np.zeros((4, 4))
    for i in range(4):
        xp = x.copy(); xp[i] += h
        A_num[:, i] = (system.integrate(xp, u) - x_next) / h
    B_num = np.zeros((4, 2))
    for j in range(2):
        up = u.copy(); up[j] += h
        B_num[:, j] = (system.integrate(x, up) - x_next) / h

    # Tolerance is loose because analytic uses an Euler discretization while
    # the rollout uses RK4. They agree as O(dt^2).
    assert np.allclose(A_an, A_num, atol=5e-3)
    assert np.allclose(B_an, B_num, atol=5e-3)


def test_concatenated_block_structure():
    sys = ConcatenatedSystem([Unicycle4D(dt=0.1), Unicycle4D(dt=0.1)], dt=0.1)
    x = np.array([1.0, 0.0, 0.1, 5.0, -2.0, 1.0, -0.2, 4.0])
    us = [np.array([0.0, 0.0]), np.array([0.05, 0.5])]
    A, Bs = sys.linearize_discrete(x, us)
    assert A.shape == (8, 8)
    assert Bs[0].shape == (8, 2)
    assert Bs[1].shape == (8, 2)
    # Block diagonal: top-right and bottom-left of A are zero.
    assert np.allclose(A[:4, 4:], 0.0)
    assert np.allclose(A[4:, :4], 0.0)
    # Each B injects only into its own player's block.
    assert np.allclose(Bs[0][4:, :], 0.0)
    assert np.allclose(Bs[1][:4, :], 0.0)
