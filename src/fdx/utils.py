import numpy as np
import scipy as sp

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def build_banded_matrix(
    n: int,
    offsets: list[int],
    weights: list[float],
    l_gap: int = 0,
    r_gap: int = 0,
) -> sp.sparse.lil_matrix:
    """Build a banded matrix.

    Parameters:
        n (int): Number of grid points.
        offsets (list[int]): Offsets of the banded matrix.
        weights (list[float]): Weights of the banded matrix.
        l_gap (int): Gap on the left side of the matrix.
        r_gap (int): Gap on the right side of the matrix.

    Returns:
        sp.sparse.lil_matrix: The banded matrix.
    """
    diags = [np.full(n - abs(k), w) for k, w in zip(offsets, weights)]
    oper = sp.sparse.diags_array(diags, offsets=offsets, shape=(n, n), format="lil")
    stencil_size = 2 * len(offsets)
    if l_gap > 0:
        for i in range(l_gap):
            oper[i, :stencil_size] = 0
            oper[i, i] = 1
    if r_gap > 0:
        for i in range(r_gap):
            oper[-(i + 1), -stencil_size:] = 0
            oper[-(i + 1), -(i + 1)] = 1
    return oper


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
