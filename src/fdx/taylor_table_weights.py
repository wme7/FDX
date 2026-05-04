import numpy as np
from scipy.special import factorial

# ---------------------------------------------------------------------------
# Taylor Table Algorithm: return explicit or implicit FD coeficients
# ---------------------------------------------------------------------------


def fd_central_weights(
    m, alpha: list[int] | list[float] | None, beta: list[int] | list[float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves for the coefficients α and β using the Taylor Table Algorithm.

    Parameters:
        m (int): Order of the derivative.
        α (list[float], optional): Array of α-indexes. If omitted, defaults to [0].
        β (list[float], optional): Array of β-indexes. If omitted, defaults to [-m:m].

    Returns:
        α_coefs (np.array): Array of α coefs.
        β_coefs (np.array): Array of β coefs.
    """
    # Set defaults if not provided
    if alpha is None:
        alpha = [0]  # self-reference
    else:
        alpha = list(alpha)  # shallow copy - prevent modifying original argument
    if beta is None:
        beta = [i for i in range(-m, m + 1)]

    # Remove self-reference (i=0) from α-list
    self_reference = None
    if 0 not in alpha:
        raise ValueError("fd_central_weights: No self-reference in α-indexes list")
    else:
        self_reference = alpha.index(0)
        alpha.remove(0)

    # Convert α and β to NumPy arrays if they are lists
    α = np.asarray(alpha, dtype=float)
    β = np.asarray(beta, dtype=float)

    # Number of coefficients
    p = len(β)
    n = len(α) + p

    # Build Matrix A
    A_ij = np.zeros((n, n))

    # β coefficients block (left part of matrix)
    powers = np.arange(n)  # np.array([0, 1, 2, ..., n])
    A_ij[:n, :p] = (1 / factorial(powers))[:, None] * β ** powers[:, None]

    # α coefficients block (right part of matrix, shifted by m)
    if len(α) > 0:
        powers_α = np.arange(n - m)
        A_ij[m : m + len(powers_α), p:] = (1 / factorial(powers_α))[
            :, None
        ] * α ** powers_α[:, None]

    # Right-hand side vector b
    b_j = np.zeros(n)
    b_j[m] = 1

    # Solve the linear system A * coefs = b
    coefs = np.linalg.solve(A_ij, b_j)

    # Extract α and β coefficients
    β_coefs = coefs[:p]  # RHS coeficients
    α_coefs = -coefs[p:]  # LHS coeficients
    if self_reference is not None:
        α_coefs = np.insert(α_coefs, self_reference, 1.0)

    # Set extremely small values to zero
    β_coefs[np.abs(β_coefs) < np.finfo(β_coefs.dtype).eps * 10] = 0
    α_coefs[np.abs(α_coefs) < np.finfo(α_coefs.dtype).eps * 10] = 0

    return α_coefs, β_coefs
