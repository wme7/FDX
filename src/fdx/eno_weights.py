import numpy as np

# ---------------------------------------------------------------------------
# ENO/WENO stencil coefficients: c_{r,j} for r=0..k, j=0..k, k=1..r_order
# ---------------------------------------------------------------------------


def c_rj(k: int, r: int, j: int) -> float:
    """
    Compute ENO/WENO stencil coefficient c_{r,j} for given k under uniform grid.

    Reference:
    - Eq. (2.12) in Shu, C.-W. (1998). Essentially Non-Oscillatory and Weighted
        Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws.
        NASA/CR-97-206253 ICASE Report No. 97-65.

    Parameters
    ----------
    k (int): Accuracy order k : 1, 2, ...
    r (int): substencil of the ENO reconstruction (-1 <= r <= k - 1)
    j (int): Relative position of the stencil (0 <= j <= k - 1)

    Returns
    -------
    float: Coefficient c_{r,j} for the ENO/WENO reconstruction
    """
    sum2 = 0.0
    for m in range(j + 1, k + 1):
        sum_ = 0.0
        for n in range(0, k + 1):
            if n != m:
                prod = 1.0
                for q in range(0, k + 1):
                    if q != n and q != m:
                        prod *= r - q + 1
                sum_ += prod

        prod2 = 1.0
        for n in range(0, k + 1):
            if n != m:
                prod2 *= m - n

        sum2 += sum_ / prod2

    return sum2


def fd_eno_substencil_weights(r_order: int, n_stencil: int) -> np.ndarray:
    """
    Get single ENO/WENO substencil coefficients.
    """
    assert n_stencil >= -1 and n_stencil <= r_order - 1, "Invalid n_stencil value"
    stencil = np.zeros(r_order, dtype=float)
    for j in range(0, r_order):
        stencil[j] = c_rj(r_order, n_stencil, j)
    return stencil


def fd_eno_weights(r_order: int) -> np.ndarray:
    """
    Generate table of ENO/WENO substencil coefficients.
    """
    table = np.zeros((r_order + 1, r_order), dtype=float)
    for r in np.arange(-1, r_order):
        for j in range(0, r_order):
            table[r + 1, j] = c_rj(r_order, r, j)
    return np.fliplr(table)


def fd_smooth_indicator_weights(r_order: int) -> list[np.ndarray]:
    """
    Table of ENO/WENO substencil smoothness indicator weights.
    """
    if r_order == 3:
        scale = np.array([np.sqrt(13 / 12), 0.5], dtype=float)[:, None]
        coeffs = np.array(
            [
                [[1.0, -2.0, 1.0], [1.0, -4.0, 3.0]],
                [[1.0, -2.0, 1.0], [1.0, 0.0, -1.0]],
                [[1.0, -2.0, 1.0], [3.0, -4.0, 1.0]],
            ],
            dtype=float,
        )
        weights = coeffs * scale
        return [weights[0], weights[1], weights[2]]

    raise NotImplementedError(
        "Smooth indicator weights are only implemented for r_order == 3."
    )
