from fractions import Fraction

import numpy as np
import scipy as sp

# ------------------------------------------------------------------ #
#  Rational Float Class                                              #
# ------------------------------------------------------------------ #


class RationalFloat:
    def __init__(self, value):
        self.value = float(value)

    def __format__(self, spec):
        if spec.endswith("r"):
            max_den = 1_000_000

            if spec.startswith(".") and spec[:-1] != ".":
                max_den = int(spec[1:-1])

            return str(Fraction(self.value).limit_denominator(max_den))

        return format(self.value, spec)


# ------------------------------------------------------------------ #
#  Operator builders                                                 #
# ------------------------------------------------------------------ #


def build_square_banded_matrix(
    n: int,
    offsets: list[int],
    weights: list[float],
) -> sp.sparse.lil_matrix:
    """Build a square banded matrix.

    Parameters:
        n (int): Number of grid points.
        offsets (list[int]): Offsets of the banded matrix.
        weights (list[float]): Weights of the banded matrix.

    Returns:
        sp.sparse.lil_matrix: The banded matrix.
    """
    diags = [np.full(n - abs(k), w) for k, w in zip(offsets, weights)]
    oper = sp.sparse.diags_array(diags, offsets=offsets, shape=(n, n), format="lil")
    return oper


def build_periodic_banded_matrix(
    n: int,
    offsets: list[int],
    weights: list[float],
) -> sp.sparse.csr_matrix:
    """Build a square periodic banded matrix.

    Parameters:
        n (int): Number of grid points.
        offsets (list[int]): Offsets of the banded matrix.
        weights (list[float]): Weights of the banded matrix.

    Returns:
        sp.sparse.csr_matrix: The periodic banded matrix.
    """
    rows = np.arange(n)
    row_idx = np.concatenate([rows] * len(offsets))
    col_idx = np.concatenate([(rows + k) % n for k in offsets])
    data = np.concatenate([np.full(n, w, dtype=float) for w in weights])
    return sp.sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n, n)).tocsr()


def build_periodic_tridiagonal_matrix(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> sp.sparse.csr_matrix:
    """Build a periodic tridiagonal matrix.

    Parameters:
        a (numpy 1d array): Sub-diagonal.
        b (numpy 1d array): Main diagonal.
        c (numpy 1d array): Super-diagonal.
    Returns:
        sp.sparse.csr_matrix: The periodic tridiagonal matrix.
    """
    assert len(a) == len(b) == len(c)
    n = len(b)
    main = sp.sparse.diags_array(b, offsets=0, shape=(n, n))
    lower = sp.sparse.diags_array(a[1:], offsets=-1, shape=(n, n))
    upper = sp.sparse.diags_array(c[:-1], offsets=1, shape=(n, n))
    corners = sp.sparse.coo_matrix(
        ([a[0], c[-1]], ([0, n - 1], [n - 1, 0])), shape=(n, n)
    )
    return (main + lower + upper + corners).tocsr()


def build_rectangular_banded_matrix(
    n: int,
    offsets: list[int],
    weights: list[float],
    n_ghost_points: int = 0,
    *,
    l_reset: int = 0,
    r_reset: int = 0,
) -> sp.sparse.csr_matrix:
    """Build a rectangular banded matrix.

    The rectangular operator is obtained from a square banded operator by
    discarding the ghost rows on each side. Optionally, the first and last
    rows of the resulting rectangular block can be cleared to support one-
    sided boundary stencils.

    Parameters:
        n (int): Number of grid points.
        offsets (list[int]): Offsets of the banded matrix.
        weights (list[float]): Weights of the banded matrix.
        n_ghost_points (int): Number of ghost points on each side.
        l_reset (int): Number of rows to reset on the left side.
        r_reset (int): Number of rows to reset on the right side.

    Returns:
        sp.sparse.csr_matrix: The rectangular banded matrix.
    """
    n_total = n + 2 * n_ghost_points
    square_oper = build_square_banded_matrix(n_total, offsets, weights)
    oper_rectangular = square_oper[n_ghost_points : n_total - n_ghost_points, :]

    boundary_width = 2 * len(offsets)
    for i in range(l_reset):
        oper_rectangular[i, :boundary_width] = 0
    for i in range(r_reset):
        oper_rectangular[-(i + 1), -boundary_width:] = 0

    return oper_rectangular.tocsr()


def build_rectangular_tridiagonal_matrix(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, n_ghost_points: int = 0
) -> sp.sparse.csr_matrix:
    """Build a square tridiagonal matrix.

    Parameters:
        a (numpy 1d array): Sub-diagonal.
        b (numpy 1d array): Main diagonal.
        c (numpy 1d array): Super-diagonal.
    Returns:
        sp.sparse.csr_matrix: The rectangular tridiagonal matrix.
    """
    assert len(a) == len(b) == len(c)
    m, offsets = len(a), [-1, 0, 1]
    if n_ghost_points > 0:
        # Build a rectangular tridiagonal matrix with ghost points
        n = m + 2 * n_ghost_points
        return sp.sparse.diags_array(
            [a, b, c], offsets=[n_ghost_points + i for i in offsets], shape=(m, n)
        ).tocsr()
    else:
        # Build a square tridiagonal matrix
        return sp.sparse.diags_array(
            [a[1:], b, c[:-1]], offsets=offsets, shape=(m, m)
        ).tocsr()


def build_lerp_boundaries_matrix(n: int, r: int) -> sp.sparse.csr_matrix:
    """Build a matrix operator that performs linear interpolation

    Use the interior data to construct ghost point values at the
    boundaries. This is used for dirichlet boundary conditions.

    Parameters:
        n (int): Number of grid points.
    Returns:
        sp.sparse.csr_matrix: The interpolation matrix.
    """
    oper = sp.sparse.lil_matrix((n + 2 * r, n))
    interp_left = np.array([[3, -2], [2, -1]])
    interp_right = np.rot90(interp_left, 2)
    oper[:r, :r] = interp_left
    oper[r:-r, :] = sp.sparse.eye(n)
    oper[-r:, -r:] = interp_right
    return oper.tocsr()


def build_mirror_symmetric_operator(mat: sp.sparse.csr_matrix) -> sp.sparse.csr_matrix:
    """Build a mirror symmetric operator.

    The mirror symmetric operator is obtained by flipping the input operator
    left-to-right and top-to-bottom.

    Parameters:
        mat (sp.sparse.csr_matrix): The input operator.

    Returns:
        sp.sparse.csr_matrix: The mirror symmetric operator.
    """
    return sp.sparse.csr_matrix(sp.sparse.lil_matrix(mat).tolil()[::-1, :][:, ::-1])


# ------------------------------------------------------------------ #
#  Operator helper                                                   #
# ------------------------------------------------------------------ #


def ensure_sparse(D):
    """Ensure CSR format: wrap dense np.ndarray results in sparse."""
    if isinstance(D, np.ndarray):
        return sp.sparse.csr_matrix(D)
    if sp.sparse.isspmatrix_csr(D):
        return D
    return D.tocsr()


# ------------------------------------------------------------------ #
# Tridiagonal Matrix Algorithm  (Thomas Algorithm)                   #
# ------------------------------------------------------------------ #


def solve_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    *,
    x: np.ndarray | None = None,
    s: int = 0,
    e: int | None = None,
) -> np.ndarray:
    """Thomas algorithm for A x = r on rows s..e (inclusive).
    a, b, c are length-n diagonals with A[i, i-1]=a[i], A[i,i]=b[i], A[i,i+1]=c[i].
    a[0] and c[n-1] are ignored (non-periodic tridiagonal).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    r = np.asarray(r, dtype=float)
    n = b.size
    if e is None:
        e = n - 1
    if x is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    gam = np.empty(e + 1, dtype=float)
    bet = b[s]
    x[s] = r[s] / bet
    for i in range(s + 1, e + 1):
        gam[i] = c[i - 1] / bet
        bet = b[i] - a[i] * gam[i]
        x[i] = (r[i] - a[i] * x[i - 1]) / bet
    for i in range(e - 1, s - 1, -1):
        x[i] -= gam[i + 1] * x[i + 1]
    return x


def solve_cyclic_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    alpha: float,
    beta: float,
    r: np.ndarray,
    *,
    x: np.ndarray | None = None,
    s: int = 0,
    e: int | None = None,
) -> np.ndarray:
    """Cyclic Thomas algorithm (Sherman–Morrison).
    alpha = A[s, e]   (upper wrap-around corner)
    beta  = A[e, s]   (lower wrap-around corner)
    """
    b = np.asarray(b, dtype=float)
    n = b.size
    if e is None:
        e = n - 1
    if x is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    gamma = -b[s]
    bb = b.copy()
    bb[s] = b[s] - gamma
    bb[e] = b[e] - alpha * beta / gamma
    solve_tridiagonal(a, bb, c, r, x=x, s=s, e=e)
    u = np.zeros(n, dtype=float)
    u[s] = gamma
    u[e] = beta
    z = np.zeros(n, dtype=float)
    solve_tridiagonal(a, bb, c, u, x=z, s=s, e=e)
    fact = (x[s] + alpha * x[e] / gamma) / (1.0 + z[s] + alpha * z[e] / gamma)
    x[s : e + 1] -= fact * z[s : e + 1]
    return x


def solve_periodic_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper matching build_periodic_tridiagonal_matrix."""
    return solve_cyclic_tridiagonal(a, b, c, a[0], c[-1], r)


# ------------------------------------------------------------------ #
# Utility functions                                                  #
# ------------------------------------------------------------------ #


def compute_order_of_accuracy(h: np.ndarray, err: np.ndarray) -> np.ndarray:
    """
    Compute the order of accuracy of a numerical method.

    Parameters:
        h (numpy 1d array): Grid spacing.
        err (numpy 1d array): L1 norm of the error.

    Returns:
        p (numpy 1d array): Order of accuracy.
    """
    p = np.zeros(len(h))
    for i in range(1, len(h)):
        p[i] = np.log(err[i - 1] / err[i]) / np.log(h[i - 1] / h[i])
    return p
