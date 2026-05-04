import numpy as np

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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
