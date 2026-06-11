import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse.linalg as spla

from fdx.utils import (
    build_periodic_tridiagonal_matrix,
    solve_periodic_tridiagonal,
    solve_tridiagonal,
)


@pytest.mark.parametrize("n", [3, 5, 16, 64, 256])
def test_cyclic_thomas_matches_sparse_lu(n):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n)
    b = 4.0 + np.abs(rng.standard_normal(n))  # diagonally dominant
    c = rng.standard_normal(n)
    r = rng.standard_normal(n)

    A = build_periodic_tridiagonal_matrix(a, b, c)
    x_ref = spla.spsolve(A, r)
    x = solve_periodic_tridiagonal(a, b, c, r)

    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("n", [3, 8, 32])
def test_standard_thomas_matches_banded(n):
    rng = np.random.default_rng(1)
    a = rng.standard_normal(n)
    b = 4.0 + np.abs(rng.standard_normal(n))
    c = rng.standard_normal(n)
    r = rng.standard_normal(n)

    # scipy banded layout: row 0 = super, row 1 = main, row 2 = sub
    ab = np.zeros((3, n))
    ab[0, 1:] = c[:-1]
    ab[1, :] = b
    ab[2, :-1] = a[1:]
    x_ref = la.solve_banded((1, 1), ab, r)

    x = solve_tridiagonal(a, b, c, r)
    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-12)


def test_cyclic_thomas_known_solution():
    """Hand-built 3×3 system with integer solution."""
    #  2  1  1       4
    #  1  3  1   x = 5   →  x = [1, 1, 1]
    #  1  1  2       4
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 3.0, 2.0])
    c = np.array([1.0, 1.0, 1.0])
    r = np.array([4.0, 5.0, 4.0])

    x = solve_periodic_tridiagonal(a, b, c, r)
    np.testing.assert_allclose(x, [1.0, 1.0, 1.0], atol=1e-14)
