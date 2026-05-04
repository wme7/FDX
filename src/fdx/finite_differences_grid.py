"""
Finite Differences Grid Module.

This module provides a class for building finite difference grids and operators.

The class `Grid1d` is a 1D grid with uniform spacing.
The class `Grid2d` is a 2D grid with uniform spacing.

The class `Grid1d` provides the following operators:
- `Dx`: 1st derivative operator for x-axis
- `Dx2`: 2nd derivative operator for x-axis
- `inv_h`: Inverse grid spacing

The class `Grid2d` provides the following operators:
- `Dx`: 1st derivative operator for x-axis
- `Dy`: 1st derivative operator for y-axis
- `Dxy`: Mixed derivative operator for x-axis and y-axis
- `Dyx`: Mixed derivative operator for x-axis and y-axis
- `grad`: Gradient operator
- `div`: Divergence operator
- `curl`: Curl operator
- `laplacian`: Laplacian operator
- `inv_hx`: Inverse grid spacing for x-axis
- `inv_hy`: Inverse grid spacing for y-axis

The class `Grid2d` also provide methods to perform operations on the grid.
- `Derivative`: Derivative operator
- `Grad`: Gradient operator
- `Div`: Divergence operator
- `Curl`: Curl operator
- `Laplacian`: Laplacian operator
"""

from enum import Enum, auto
from functools import cached_property

import numpy as np
import scipy as sp

from .fornberg_weights import fd_explicit_weights
from .taylor_table_weights import fd_central_weights


# ------------------------------------------------------------------ #
#  FD Operators Parameters                                           #
# ------------------------------------------------------------------ #
class BoundaryCondition(Enum):
    PERIODIC = auto()
    DIRICHLET = auto()
    GHOST_POINTS = auto()
    # BRADY_LIVESCU_CONSERVATIVE = auto()


class FiniteDifferenceScheme(Enum):
    EXPLICIT = auto()
    TRIDIAGONAL = auto()
    COMPACT = auto()
    # WENO5 = auto()
    # CRWENO5 = auto()


# ------------------------------------------------------------------ #
#  Operator builders                                                 #
# ------------------------------------------------------------------ #
def build_explicit_fd_matrix(
    n: int,
    m_derivative: int,
    r_width: int,
    h: float,
    bc: BoundaryCondition = BoundaryCondition.DIRICHLET,
    verbose: bool = False,
) -> sp.sparse.csr_matrix:
    """
    Build a 1D explicit finite difference differentiation matrix on a uniform grid.

    Constructs a sparse banded matrix whose rows encode centred finite difference
    stencils of order ``m_derivative``. Near boundaries the stencil is shifted
    one-sided (Dirichlet) or wrapped around (periodic).

    Parameters
    ----------
    n : int
        Number of grid points.
    m_derivative : int
        Order of the derivative to approximate.
    r_width : int
        Half-width of the centred stencil (stencil spans ``2*r_width + 1`` points).
    h : float
        Uniform grid spacing.
    bc : BoundaryCondition, optional
        Boundary condition type. ``PERIODIC`` applies wrap-around corner entries;
        ``DIRICHLET`` replaces boundary rows with one-sided stencils.
        Default is ``BoundaryCondition.DIRICHLET``.
    verbose : bool, optional
        If ``True``, prints the dense matrix before scaling. Default is ``False``.

    Returns
    -------
    D : scipy.sparse.csr_matrix, shape (n, n)
        Sparse differentiation matrix scaled by ``h**(-m_derivative)``.
    """
    scaling = pow(h, -m_derivative)
    offsets = list(range(-r_width, r_width + 1))
    weights = fd_explicit_weights(m=m_derivative, x=0, alpha=offsets)

    diags = [np.full(n - abs(k), w) for k, w in zip(offsets, weights)]
    D = sp.sparse.diags_array(diags, offsets=offsets, shape=(n, n), format="lil")

    match bc:
        case BoundaryCondition.PERIODIC:
            _apply_periodic_corners(D, n, offsets, weights)

        case BoundaryCondition.DIRICHLET:
            _apply_explicit_dirichlet_onesided(D, n, m_derivative, r_width)

        case BoundaryCondition.GHOST_POINTS:
            _apply_tridiagonal_ghost_points(D, n, r_width)

    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print(D.toarray())

    return D.tocsr() * scaling


def build_tridiagonal_fd_matrix(
    n: int,
    m_derivative: int,
    r_width: int,
    h: float,
    bc: BoundaryCondition = BoundaryCondition.DIRICHLET,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build a 1D implicit finite difference differentiation matrix on a uniform grid.

    Solves the banded implicit system ``A @ D = B * h**(-m_derivative)``, where ``A``
    encodes the tridiagonal LHS stencil and ``B`` the explicit RHS stencil. The
    system is factorised once via sparse LU and solved for all columns of ``B``
    simultaneously, returning a dense differentiation operator.

    Parameters
    ----------
    n : int
        Number of grid points.
    m_derivative : int
        Order of the derivative to approximate.
    r_width : int
        Half-width of the RHS explicit stencil (stencil spans ``2*r_width + 1`` points).
    h : float
        Uniform grid spacing.
    bc : BoundaryCondition, optional
        Boundary condition type. ``PERIODIC`` applies wrap-around corner entries to
        both ``A`` and ``B``; ``DIRICHLET`` replaces boundary rows with one-sided
        compact stencils. Default is ``BoundaryCondition.DIRICHLET``.
    verbose : bool, optional
        If ``True``, prints the dense ``A`` and ``B`` matrices before solving.
        Default is ``False``.

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Dense differentiation operator scaled by ``h**(-m_derivative)``.
    """
    scale = pow(h, -m_derivative)
    a_offsets = [-1, 0, 1]
    b_offsets = list(range(-r_width, r_width + 1))
    a_weights, b_weights = fd_central_weights(
        m=m_derivative, alpha=a_offsets, beta=b_offsets
    )

    # Build sparse banded matrices in LIL (efficient for row-wise assembly)
    a_diags = [np.full(n - abs(k), w) for k, w in zip(a_offsets, a_weights)]
    b_diags = [np.full(n - abs(k), w) for k, w in zip(b_offsets, b_weights)]
    A = sp.sparse.diags_array(a_diags, offsets=a_offsets, shape=(n, n), format="lil")
    B = sp.sparse.diags_array(b_diags, offsets=b_offsets, shape=(n, n), format="lil")

    match bc:
        case BoundaryCondition.PERIODIC:
            _apply_periodic_corners(A, n, a_offsets, a_weights)
            _apply_periodic_corners(B, n, b_offsets, b_weights)

        case BoundaryCondition.DIRICHLET:
            _apply_tridiagonal_dirichlet_onesided(A, B, n, m_derivative, r_width)

        case BoundaryCondition.GHOST_POINTS:
            raise ValueError("Ghost points BC only available for explicit schemes.")

    # Convert once to CSC — optimal for column-wise LU factorization
    A_csc = A.tocsc()
    B_csc = B.tocsc()

    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print(A.toarray())
            print(B.toarray())

    # Factorize A once, solve against all columns of B in one shot
    lu = sp.sparse.linalg.splu(A_csc)
    D = lu.solve(B_csc.toarray())  # shape: (n, n), dtype: float64

    return D * scale


# ------------------------------------------------------------------ #
#  Boundary helpers                                                  #
# ------------------------------------------------------------------ #
def _apply_periodic_corners(D, n, offsets, weights):
    """
    Fill the wrap-around corner entries of a sparse matrix for a periodic stencil.

    For each non-zero diagonal offset ``k``, the ``k`` entries that fall outside
    the matrix bounds are placed in the opposite corner to enforce periodicity:

    - ``k > 0`` : missing entries go in the bottom-left corner,
                rows ``[n-k, n)``, cols ``[0, k)``.
    - ``k < 0`` : missing entries go in the top-right corner,
                rows ``[0, |k|)``, cols ``[n-|k|, n)``.

    Parameters
    ----------
    D : scipy.sparse.lil_matrix, shape (n, n)
        Differentiation matrix to modify in place.
    n : int
        Number of grid points (matrix dimension).
    offsets : sequence of int
        Diagonal offsets of the stencil.
    weights : sequence of float
        Finite difference weights corresponding to each offset.
    """
    for k, w in zip(offsets, weights):
        if k == 0:
            continue
        if k > 0:
            rows = np.arange(n - k, n)
            cols = np.arange(0, k)
        else:
            abs_k = abs(k)
            rows = np.arange(0, abs_k)
            cols = np.arange(n - abs_k, n)
        D[rows, cols] = w


def _apply_tridiagonal_ghost_points(D, n, r_width):
    """
    Replace near-boundary rows with an identity matrix.
    """
    stencil_size = 2 * r_width + 1
    for r in range(r_width):
        # -- top boundary:
        D[n - 1 - r, n - stencil_size :] = 0  # reset values
        D[n - 1 - r, n - 1 - r] = 1

        # -- bottom boundary:
        D[r, :stencil_size] = 0  # reset values
        D[r, r] = 1


def _apply_explicit_dirichlet_onesided(D, n, m_derivative, r_width):
    """
    Replace near-boundary rows with one-sided finite difference stencils.

    For the ``r_width`` rows closest to each boundary, the centred stencil window
    is shifted so that it stays within ``[0, n)``. Row ``r`` (0-indexed from each
    end) uses a stencil whose window spans ``2*r_width + 1`` columns, clamped to
    the first (top boundary) or last (bottom boundary) columns of the matrix.

    Parameters
    ----------
    D : scipy.sparse.lil_matrix, shape (n, n)
        Differentiation matrix to modify in place.
    n : int
        Number of grid points (matrix dimension).
    m_derivative : int
        Order of the derivative to approximate.
    r_width : int
        Number of boundary rows to replace on each side.
    """
    stencil_size = 2 * r_width + 1
    for r in range(r_width):
        # -- top boundary:
        a_top = list(range(-r, stencil_size - r))
        w_top = fd_explicit_weights(m=m_derivative, x=0, alpha=a_top)
        D[r, :stencil_size] = w_top

        # -- bottom boundary:
        a_bot = list(range(-(stencil_size - r - 1), r + 1))
        w_bot = fd_explicit_weights(m=m_derivative, x=0, alpha=a_bot)
        D[n - 1 - r, n - stencil_size :] = w_bot


def _apply_tridiagonal_dirichlet_onesided(A, B, n, m_derivative, r_width):
    """
    Replace near-boundary rows of a compact scheme with one-sided stencils.

    Clears the ``r_width`` rows nearest each boundary in both the LHS matrix
    ``A`` and the RHS matrix ``B``, then fills them with one-sided compact
    stencils. The RHS stencil width is fixed at ``m_derivative + 2`` points to
    recover standard schemes such as Lele-6 and Padé-4. The first and last rows
    use a reduced LHS stencil (two-point instead of tridiagonal) to avoid
    referencing phantom points outside the domain.

    Parameters
    ----------
    A : scipy.sparse.lil_matrix, shape (n, n)
        Tridiagonal LHS matrix of the compact scheme, modified in place.
    B : scipy.sparse.lil_matrix, shape (n, n)
        Explicit RHS matrix of the compact scheme, modified in place.
    n : int
        Number of grid points (matrix dimension).
    m_derivative : int
        Order of the derivative to approximate.
    r_width : int
        Number of boundary rows to replace on each side.
    """
    stencil_size = 2 * r_width + 1
    b_stencil_size = m_derivative + 2  # to recover Lele-6 and Padé-4

    for r in range(r_width):
        # -- reset values near boundaries
        A[r, :stencil_size] = 0
        B[r, :stencil_size] = 0
        A[n - 1 - r, n - stencil_size :] = 0
        B[n - 1 - r, n - stencil_size :] = 0

    for r in range(r_width):
        # -- top boundary:
        a_top = [-1, 0, 1] if r != 0 else [0, 1]
        b_top = list(range(-r, b_stencil_size - r))
        a_w, b_w = fd_central_weights(m=m_derivative, alpha=a_top, beta=b_top)
        A[r, : len(a_top)] = a_w
        B[r, : len(b_top)] = b_w

        # -- bottom boundary:
        a_bot = [-1, 0, 1] if r != 0 else [-1, 0]
        b_bot = list(range(-(b_stencil_size - r - 1), r + 1))
        a_w, b_w = fd_central_weights(m=m_derivative, alpha=a_bot, beta=b_bot)
        A[n - 1 - r, n - len(a_bot) :] = a_w
        B[n - 1 - r, n - len(b_bot) :] = b_w


# ------------------------------------------------------------------ #
#  Operator helper                                                   #
# ------------------------------------------------------------------ #
def _build_1d_operator(n, m, r, h, bc, scheme, verbose) -> sp.sparse.csr_matrix:
    match scheme:
        case FiniteDifferenceScheme.EXPLICIT:
            return build_explicit_fd_matrix(n, m, r, h, bc, verbose)
        case FiniteDifferenceScheme.TRIDIAGONAL:
            return build_tridiagonal_fd_matrix(n, m, r, h, bc, verbose)
        case FiniteDifferenceScheme.COMPACT:
            return build_tridiagonal_fd_matrix(n, m, 1, h, bc, verbose)


# ------------------------------------------------------------------ #
#  Grid Classes                                                      #
# ------------------------------------------------------------------ #
def _uniform_1d_grid_axis(
    a: float,
    b: float,
    n: int,
    bc: BoundaryCondition,
    r: int,
) -> tuple[float, float, int, bool, float]:
    """
    Uniform 1D axis parameters consistent with ``Grid1d``.

    Returns ``(a_grid, b_grid, n_grid, endpoint, h)`` for ``np.linspace``.
    """
    if n < 2:
        raise ValueError(f"Axis grid requires n >= 2, got n={n}.")
    if not (b > a):
        raise ValueError(f"Axis grid requires b > a, got a={a}, b={b}.")
    if r < 0:
        raise ValueError(f"Stencil half-width r must be >= 0, got r={r}.")

    a0, b0, n0 = float(a), float(b), int(n)

    match bc:
        case BoundaryCondition.PERIODIC:
            return a0, b0, n0, False, (b0 - a0) / n0
        case BoundaryCondition.DIRICHLET:
            return a0, b0, n0, True, (b0 - a0) / (n0 - 1)
        case BoundaryCondition.GHOST_POINTS:
            if r == 0:
                raise ValueError("BoundaryCondition.GHOST_POINTS requires r >= 1.")
            h = (b0 - a0) / (n0 - 1)
            n_grid = n0 + 2 * r
            a_grid = a0 - r * h
            b_grid = b0 + r * h
            return a_grid, b_grid, n_grid, True, h
        case _:
            raise ValueError(f"Unsupported BoundaryCondition for axis grid: {bc!r}")


class Grid1d:
    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        n: int = 100,
        r: int = 1,
        bc: BoundaryCondition = BoundaryCondition.DIRICHLET,
        scheme: FiniteDifferenceScheme = FiniteDifferenceScheme.EXPLICIT,
        verbose: bool = False,
    ):

        self.r = r  # stencil width
        self.bc = bc
        self.scheme = scheme
        self.verbose = verbose

        a_grid, b_grid, n_grid, endpoint, h = _uniform_1d_grid_axis(a, b, n, bc, r)

        self.a = a_grid
        self.b = b_grid
        self.n = n_grid

        self.x = np.linspace(
            start=a_grid, stop=b_grid, num=n_grid, endpoint=endpoint, dtype=float
        )
        self.h = h  # grid spacing

    @cached_property
    def Dx(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n, 1, self.r, self.h, self.bc, self.scheme, self.verbose
        )

    @cached_property
    def Dx2(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n, 2, self.r, self.h, self.bc, self.scheme, self.verbose
        )

    @cached_property
    def inv_h(self) -> float:
        return 1.0 / self.h


class Grid2d:
    def __init__(
        self,
        xa: float = 0.0,
        xb: float = 1.0,
        nx: int = 100,
        rx: int = 1,
        ya: float = 0.0,
        yb: float = 1.0,
        ny: int = 100,
        ry: int = 1,
        bcx: BoundaryCondition = BoundaryCondition.DIRICHLET,
        bcy: BoundaryCondition = BoundaryCondition.DIRICHLET,
        scheme: FiniteDifferenceScheme = FiniteDifferenceScheme.EXPLICIT,
        verbose: bool = False,
    ):

        self.rx = rx  # stencil width for x-axis
        self.ry = ry  # stencil width for y-axis
        self.bcx = bcx
        self.bcy = bcy
        self.scheme = scheme
        self.verbose = verbose

        xa_g, xb_g, nx_g, endpoint_x, hx = _uniform_1d_grid_axis(xa, xb, nx, bcx, rx)
        ya_g, yb_g, ny_g, endpoint_y, hy = _uniform_1d_grid_axis(ya, yb, ny, bcy, ry)

        self.xa = xa_g
        self.xb = xb_g
        self.nx = nx_g

        self.x = np.linspace(
            start=xa_g, stop=xb_g, num=nx_g, endpoint=endpoint_x, dtype=float
        )
        self.hx = hx  # grid spacing for x-axis

        self.ya = ya_g
        self.yb = yb_g
        self.ny = ny_g

        self.y = np.linspace(
            start=ya_g, stop=yb_g, num=ny_g, endpoint=endpoint_y, dtype=float
        )
        self.hy = hy  # grid spacing for y-axis

    @cached_property
    def Ix(self):
        return sp.sparse.eye(self.nx, format="csr")

    @cached_property
    def Iy(self):
        return sp.sparse.eye(self.ny, format="csr")

    @cached_property
    def Dx_1d(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.nx, 1, self.rx, self.hx, self.bcx, self.scheme, self.verbose
        )

    @cached_property
    def Dy_1d(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.ny, 1, self.ry, self.hy, self.bcy, self.scheme, self.verbose
        )

    @cached_property
    def Dx2_1d(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.nx, 2, self.rx, self.hx, self.bcx, self.scheme, self.verbose
        )

    @cached_property
    def Dy2_1d(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.ny, 2, self.ry, self.hy, self.bcy, self.scheme, self.verbose
        )

    @cached_property
    def Dx2d(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.Iy, self.Dx2_1d, format="csr")

    @cached_property
    def Dy2d(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.Dy2_1d, self.Ix, format="csr")

    @cached_property
    def Dx(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.Iy, self.Dx_1d, format="csr")

    @cached_property
    def Dy(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.Dy_1d, self.Ix, format="csr")

    @cached_property
    def Dxy(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.Dy_1d, self.Dx_1d, format="csr")

    @cached_property
    def Dyx(self) -> sp.sparse.csr_matrix:
        return self.Dxy

    @cached_property
    def grad(self):
        return sp.sparse.vstack([self.Dx, self.Dy], format="csr")

    @cached_property
    def div(self):
        return sp.sparse.hstack([self.Dx, self.Dy], format="csr")

    @cached_property
    def curl(self):
        return sp.sparse.hstack([-self.Dy, self.Dx], format="csr")

    @cached_property
    def laplacian(self):
        return self.Dx2d + self.Dy2d

    @cached_property
    def inv_hx(self) -> float:
        return 1.0 / self.hx

    @cached_property
    def inv_hy(self) -> float:
        return 1.0 / self.hy

    # Shortcuts to perform operations on the grid
    def Derivative(self, u: np.ndarray, direction: str) -> np.ndarray:
        u_flat = u.ravel()  # row-major flatten: index k = j*nx + i
        if direction == "x":
            du_flat = self.Dx @ u_flat
        elif direction == "y":
            du_flat = self.Dy @ u_flat
        elif direction in ["yx", "xy"]:
            du_flat = self.Dxy @ u_flat
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return du_flat.reshape(self.ny, self.nx)

    def Grad(self, sField: np.ndarray) -> list[np.ndarray]:
        grad_flat = self.grad @ sField.ravel()
        grad_array = grad_flat.reshape(2, self.ny, self.nx)
        return [grad_array[0, :, :], grad_array[1, :, :]]

    def Div(self, vField: list[np.ndarray]) -> np.ndarray:
        div_flat = self.div @ np.concatenate([vField[0].ravel(), vField[1].ravel()])
        return div_flat.reshape(self.ny, self.nx)

    def Curl(self, vField: list[np.ndarray]) -> np.ndarray:
        curl_flat = self.curl @ np.concatenate([vField[0].ravel(), vField[1].ravel()])
        return curl_flat.reshape(self.ny, self.nx)

    def Laplacian(self, sField: np.ndarray) -> np.ndarray:
        Laplacian_flat = self.laplacian @ sField.ravel()
        return Laplacian_flat.reshape(self.ny, self.nx)
