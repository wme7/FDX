"""
Finite Differences Grid Module.

This module provides a class for building finite difference grids and operators.

The class `Grid1d` is a 1D grid with uniform spacing.
It provides the following operators:
- `Dx`: 1st derivative operator for x-axis
- `Dx2`: 2nd derivative operator for x-axis

The class `Grid2d` is a 2D grid with uniform spacing.
It provides the following operators:
- `Dx`: 1st derivative operator for x-axis
- `Dy`: 1st derivative operator for y-axis
- `Dxy`: Mixed derivative operator for x-axis and y-axis
- `Dyx`: Mixed derivative operator for x-axis and y-axis
- `grad`: Gradient operator
- `div`: Divergence operator
- `curl`: Curl operator
- `laplacian`: Laplacian operator

The class `Grid2d` also provide methods to perform operations on the grid.
- `Derivative(sField: np.ndarray, axis: str)`: Derivative operator
- `Grad(sField: np.ndarray)`: Gradient operator
- `Div(vField: list[np.ndarray])`: Divergence operator
- `Curl(vField: list[np.ndarray])`: Curl operator
- `Laplacian(sField: np.ndarray)`: Laplacian operator
"""

from enum import Enum, auto
from functools import cached_property

import numpy as np
import scipy as sp

from .fornberg_weights import fd_explicit_weights
from .taylor_table_weights import fd_central_weights
from .utils import build_square_banded_matrix


# ------------------------------------------------------------------ #
#  FD Operators Parameters                                           #
# ------------------------------------------------------------------ #
class BoundaryCondition(Enum):
    PERIODIC = auto()
    DIRICHLET = auto()
    GHOST_POINTS = auto()
    # CONSERVATIVE = auto()


class FiniteDifferenceScheme(Enum):
    CENTRAL = auto()
    PADE = auto()
    COMPACT = auto()
    UPWIND = auto()
    DOWNWIND = auto()
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
    bc: BoundaryCondition,
    bias: str,
    verbose: bool = False,
) -> sp.sparse.csr_matrix:
    """
    Build a 1D explicit finite-difference differentiation matrix on a uniform grid.

    Constructs a sparse banded matrix whose rows encode finite-difference
    stencils of order ``m_derivative``. The optional ``bias`` argument allows
    choosing centred or one-sided/upwind stencils; near boundaries the
    stencil is either replaced by one-sided formulas (Dirichlet/ghost-point
    treatment) or wrapped around for periodic domains.

    Parameters
    ----------
    n : int
        Number of grid points.
    m_derivative : int
        Order of the derivative to approximate.
    r_width : int
        Half-width of the centred stencil (stencil spans ``2*r_width + 1`` points)
        for centred schemes, or the upwind stencil half-width for biased schemes.
    h : float
        Uniform grid spacing.
    bc : BoundaryCondition
        Boundary condition type. ``PERIODIC`` applies wrap-around corner entries;
        ``DIRICHLET`` replaces boundary rows with one-sided stencils; ``GHOST_POINTS``
        introduces ghost-point treatment for explicit schemes.
    bias : str
        Stencil bias (case-insensitive): ``"central"``/``"both"`` (centred
        stencil), ``"left"``/``"upwind"`` (left-biased / upwind), or
        ``"right"``/``"downwind"`` (right-biased / downwind). For biased
        schemes the implementation currently supports first-derivative upwind
        stencils only.
    verbose : bool, optional
        If ``True``, prints the dense matrix before scaling. Default is ``False``.

    Returns
    -------
    D : scipy.sparse.csr_matrix, shape (n, n)
        Sparse differentiation matrix scaled by ``h**(-m_derivative)``.
    """
    scaling = pow(h, -m_derivative)

    # determine stencil offsets based on requested bias
    if bias in ["both", "central"]:
        offsets = list(range(-r_width, r_width + 1))
    elif bias in ["left", "upwind"]:
        if m_derivative != 1:
            raise ValueError(
                "Upwind/Downwind scheme implemented for 1st derivative only."
            )
        offsets = list(range(-r_width, 1))
    elif bias in ["right", "downwind"]:
        if m_derivative != 1:
            raise ValueError(
                "Upwind/Downwind scheme implemented for 1st derivative only."
            )
        offsets = list(range(0, r_width + 1))
    else:
        raise ValueError(f"Unknown bias: {bias!r}")
    weights = fd_explicit_weights(m=m_derivative, x=0, alpha=offsets) * scaling

    D = build_square_banded_matrix(n, offsets, weights.tolist())

    match bc:
        case BoundaryCondition.PERIODIC:
            _apply_periodic_corners(D, n, offsets, weights)

        case BoundaryCondition.DIRICHLET:
            _apply_scheme_onesided(D, m_derivative, r_width, scaling, bias)

        case BoundaryCondition.GHOST_POINTS:
            _apply_ghost_points(D, r_width, bias)

    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print(D.toarray())

    return D.tocsr()


def build_pade_fd_matrix(
    n: int,
    m_derivative: int,
    r_width: int,
    h: float,
    bc: BoundaryCondition,
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
    A = build_square_banded_matrix(n, a_offsets, a_weights.tolist())
    B = build_square_banded_matrix(n, b_offsets, b_weights.tolist())

    match bc:
        case BoundaryCondition.PERIODIC:
            _apply_periodic_corners(A, n, a_offsets, a_weights)
            _apply_periodic_corners(B, n, b_offsets, b_weights)

        case BoundaryCondition.DIRICHLET:
            _apply_pade_onesided(A, B, m_derivative, r_width)

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


def _apply_ghost_points(D, r_width, bias):
    """
    Replace rows near the selected boundary with an identity matrix.

    Depending on ``bias``, this writes identity rows at the left boundary
    (``"left"``/``"upwind"``), right boundary (``"right"``/``"downwind"``),
    or both boundaries (``"both"``).
    """
    stencil_size = 2 * r_width + 1
    for r in range(r_width):
        if bias in ["left", "upwind", "both"]:
            # -- left boundary:
            D[r, :stencil_size] = 0  # reset values
            D[r, r] = 1
        elif bias in ["right", "downwind", "both"]:
            # -- right boundary:
            D[-(1 + r), -stencil_size:] = 0  # reset values
            D[-(1 + r), -(1 + r)] = 1


def _apply_scheme_onesided(D, m_derivative, r_width, scale, bias="both"):
    """
    Replace near-boundary rows with one-sided finite-difference stencils.

    This function modifies the supplied LIL sparse matrix `D` in-place. The
    number of grid points ``n`` is inferred from ``D.shape[0]``. For the
    ``r_width`` rows nearest a boundary the centred stencil window is shifted
    (clamped) so it remains inside the domain and a one-sided finite-difference
    stencil is applied. The argument ``scale`` is multiplied into the computed
    weights before they are written into the matrix.

    The ``bias`` argument controls which boundaries are modified:
    - ``"both"`` or ``"central"``: replace rows on both left (top) and right
        (bottom) boundaries using shifted centred stencils of width
        ``2*r_width + 1``.
    - ``"left"`` or ``"upwind"``: only replace the first ``r_width`` rows
        (top boundary) with one-sided/upwind stencils.
    - ``"right"`` or ``"downwind"``: only replace the last ``r_width`` rows
        (bottom boundary) with one-sided/downwind stencils.

    Parameters
    ----------
    D : scipy.sparse.lil_matrix, shape (n, n)
        Differentiation matrix to modify in place (rows near boundaries are
        overwritten with one-sided stencils).
    m_derivative : int
        Order of the derivative to approximate (passed to the weight routine).
    r_width : int
        Number of boundary rows to replace on each side (stencil half-width).
    scale : float
        Multiplicative scaling applied to the computed finite-difference
        weights (typically `h**(-m_derivative)`).
    bias : str, optional
        Boundary side to modify: ``"both"`` (default), ``"left"``,
        ``"right"``, ``"upwind"`` or ``"downwind"``.
    """
    if bias in ["left", "upwind"]:
        for r in range(r_width):
            # -- left boundary:
            if r == 0:
                D[r, r] = 1
            else:
                α = list(range(-r, 1))
                ω = fd_explicit_weights(m=m_derivative, x=0, alpha=α)
                D[r, : (r + 1)] = ω * scale

    elif bias in ["right", "downwind"]:
        for r in range(r_width):
            # -- right boundary:
            if r == 0:
                D[-(1 + r), -(1 + r)] = 1
            else:
                α = list(range(0, r + 1))
                ω = fd_explicit_weights(m=m_derivative, x=0, alpha=α)
                D[-(1 + r), -(r + 1) :] = ω * scale

    elif bias in ["both", "central"]:
        stencil_size = 2 * r_width + 1
        for r in range(r_width):
            # -- left boundary:
            α = list(range(-r, stencil_size - r))
            ω = fd_explicit_weights(m=m_derivative, x=0, alpha=α)
            D[r, :stencil_size] = ω * scale

            # -- right boundary:
            α = list(range(-(stencil_size - r - 1), r + 1))
            ω = fd_explicit_weights(m=m_derivative, x=0, alpha=α)
            D[-(1 + r), -stencil_size:] = ω * scale


def _apply_pade_onesided(A, B, m_derivative, r_width):
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
        A[-(1 + r), -stencil_size:] = 0
        B[-(1 + r), -stencil_size:] = 0

    for r in range(r_width):
        # -- top boundary:
        α = [-1, 0, 1] if r != 0 else [0, 1]
        β = list(range(-r, b_stencil_size - r))
        a_w, b_w = fd_central_weights(m=m_derivative, alpha=α, beta=β)
        A[r, : len(α)] = a_w
        B[r, : len(β)] = b_w

        # -- bottom boundary:
        α = [-1, 0, 1] if r != 0 else [-1, 0]
        β = list(range(-(b_stencil_size - r - 1), r + 1))
        a_w, b_w = fd_central_weights(m=m_derivative, alpha=α, beta=β)
        A[-(1 + r), -len(α) :] = a_w
        B[-(1 + r), -len(β) :] = b_w


# ------------------------------------------------------------------ #
#  Operator helper                                                   #
# ------------------------------------------------------------------ #
def _build_1d_operator(n, m, r, h, bc, scheme, verbose) -> sp.sparse.csr_matrix:
    match scheme:
        case FiniteDifferenceScheme.CENTRAL:
            return build_explicit_fd_matrix(n, m, r, h, bc, "central", verbose)
        case FiniteDifferenceScheme.UPWIND:
            return build_explicit_fd_matrix(n, m, r, h, bc, "upwind", verbose)
        case FiniteDifferenceScheme.DOWNWIND:
            return build_explicit_fd_matrix(n, m, r, h, bc, "downwind", verbose)
        case FiniteDifferenceScheme.PADE:
            return build_pade_fd_matrix(n, m, r, h, bc, verbose)
        case FiniteDifferenceScheme.COMPACT:
            return build_pade_fd_matrix(n, m, 1, h, bc, verbose)


# ------------------------------------------------------------------ #
#  Grid Classes                                                      #
# ------------------------------------------------------------------ #
def _uniform_1d_grid_axis(
    a: float,
    b: float,
    n: int,
    bc: BoundaryCondition,
    scheme: FiniteDifferenceScheme,
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
            if r < 1:
                raise ValueError("BoundaryCondition.GHOST_POINTS requires r >= 1.")
            h = (b0 - a0) / (n0 - 1)
            match scheme:
                case FiniteDifferenceScheme.UPWIND:
                    n_grid = n0 + r
                    a_grid = a0 - r * h
                    b_grid = b0
                    return a_grid, b_grid, n_grid, True, h
                case FiniteDifferenceScheme.DOWNWIND:
                    n_grid = n0 + r
                    a_grid = a0
                    b_grid = b0 + r * h
                    return a_grid, b_grid, n_grid, True, h
                case _:
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
        scheme: FiniteDifferenceScheme = FiniteDifferenceScheme.CENTRAL,
        verbose: bool = False,
    ):

        self.r = r  # stencil width
        self.bc = bc
        self.scheme = scheme
        self.verbose = verbose

        a_grid, b_grid, n_grid, endpoint, h = _uniform_1d_grid_axis(
            a, b, n, bc, scheme, r
        )

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
        scheme: FiniteDifferenceScheme = FiniteDifferenceScheme.CENTRAL,
        verbose: bool = False,
    ):

        self.rx = rx  # stencil width for x-axis
        self.ry = ry  # stencil width for y-axis
        self.bcx = bcx
        self.bcy = bcy
        self.scheme = scheme
        self.verbose = verbose

        xa_g, xb_g, nx_g, endpoint_x, hx = _uniform_1d_grid_axis(
            xa, xb, nx, bcx, scheme, rx
        )
        ya_g, yb_g, ny_g, endpoint_y, hy = _uniform_1d_grid_axis(
            ya, yb, ny, bcy, scheme, ry
        )

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
    def Dx_1d_T(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.nx, 1, self.rx, self.hx, self.bcx, self.scheme, self.verbose
        ).T

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
    def Derivative(self, u: np.ndarray, axis: str) -> np.ndarray:
        if axis == "x":
            du_flat = u @ self.Dx_1d_T
        elif axis == "y":
            du_flat = self.Dy_1d @ u
        elif axis in ["yx", "xy"]:
            du_flat = self.Dy_1d @ (u @ self.Dx_1d_T)
        else:
            raise ValueError(f"Invalid axis: {axis}")
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
