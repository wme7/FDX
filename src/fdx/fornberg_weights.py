import numpy as np

# ---------------------------------------------------------------------------
# Fornberg's weights algorithm: return table of FD coeficients
# ---------------------------------------------------------------------------


def fornberg_weights(m: int, x: float, α: list[int] | list[float]) -> np.ndarray:
    """
    A clean, faithful implementation of Fornberg (1998) Algorithm 1.

    This version mirrors the paper's pseudocode variable names exactly
    and is the recommended implementation.

    Parameters
    ----------
    m (int), highest derivative order needed
    x (float), evaluation point
    α (list[float]), stencil grid points (n+1 points, 0-indexed)

    Returns
    -------
    c : np.ndarray, shape (n+1, m+1)
        c[m, j] = weight for alpha[j] in the m-th derivative approximation.
    """
    n = len(α)
    if m >= n:
        raise ValueError("fornberg_weights: length(α) must be larger than m")

    c1 = 1.0
    c4 = α[0] - x
    C = np.zeros((n, m + 1), dtype=float)
    C[0, 0] = 1.0

    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = α[i] - x
        for j in range(i):
            c3 = α[i] - α[j]
            c2 = c2 * c3
            if j == i - 1:
                for s in range(mn, 0, -1):
                    C[i, s] = c1 * (s * C[i - 1, s - 1] - c5 * C[i - 1, s]) / c2
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2

            for s in range(mn, 0, -1):
                C[j, s] = (c4 * C[j, s] - s * C[j, s - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2

    return C  # shape: (n+1, m+1)


# ---------------------------------------------------------------------------
# Convenience wrapper: return only the weights for derivative order m
# ---------------------------------------------------------------------------


def fd_explicit_weights(
    m: int, x: float | None, alpha: list[int] | list[float] | None
) -> np.ndarray:
    """
    Return only the weights for the m-th derivative.

    Parameters
    ----------
    m     : derivative order
    xbar  : evaluation point. If omitted, defaults to x = 0.
    α     : stencil points. If omitted, defaults to [-m, 0, m].

    Returns
    -------
    coefs : np.ndarray, shape (len(alpha),) stentil coefs or weights
    """
    # Set defaults if not provided
    if x is None:
        x = 0.0
    if alpha is None:
        alpha = [i for i in range(-m, m + 1)]

    # Compute coefs
    coefs = fornberg_weights(m, x, alpha)

    # Set extremely small values to zero
    coefs[np.abs(coefs) < np.finfo(coefs.dtype).eps * 10] = 0

    return coefs[:, -1]
