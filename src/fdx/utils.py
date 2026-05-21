import numpy as np
import scipy as sp

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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
    oper = sp.sparse.lil_matrix((n, n))
    for row in range(n):
        for k, w in zip(offsets, weights):
            oper[row, (row + k) % n] = w
    return oper.tocsr()


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
