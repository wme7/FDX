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


# ------------------------------------------------------------------ #
#  Operator builders                                                 #
# ------------------------------------------------------------------ #
def build_explicit_fd_matrix(
    n: int,
    m_derivative: int,
    r_width: int,
    n_ghost_points: int,
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
            _apply_ghost_points(D, n_ghost_points, bias)

    if verbose:
        with np.printoptions(precision=2, suppress=True, linewidth=120):
            print(f"D ({D.shape}):\n{D.toarray()}")

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
            print(f"A ({A.shape}):\n{A.toarray()}")
            print(f"B ({B.shape}):\n{B.toarray()}")

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


def _apply_ghost_points(D, n_ghost_points, bias):
    """
    Replace rows near the selected boundary with an identity matrix.

    Depending on ``bias``, this writes identity rows at the left boundary
    (``"left"``/``"upwind"``), right boundary (``"right"``/``"downwind"``),
    or both boundaries (``"central"``/``"both"``).
    """
    # stencil_size = 2 * r_width + 1
    for r in range(n_ghost_points):
        if bias in ["left", "upwind"]:
            # -- left boundary:
            D[r, :] = 0  # reset values
            D[r, r] = 1
        elif bias in ["right", "downwind"]:
            # -- right boundary:
            D[-r - 1, :] = 0  # reset values
            D[-r - 1, -r - 1] = 1
        elif bias in ["both", "central"]:
            # -- left boundary:
            D[r, :] = 0  # reset values
            D[r, r] = 1
            # -- right boundary:
            D[-r - 1, :] = 0  # reset values
            D[-r - 1, -r - 1] = 1


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
                D[-r - 1, -r - 1] = 1
            else:
                α = list(range(0, r + 1))
                ω = fd_explicit_weights(m=m_derivative, x=0, alpha=α)
                D[-r - 1, -(r + 1) :] = ω * scale

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
            D[-r - 1, -stencil_size:] = ω * scale


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
        A[-r - 1, -stencil_size:] = 0
        B[-r - 1, -stencil_size:] = 0

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
        A[-r - 1, -len(α) :] = a_w
        B[-r - 1, -len(β) :] = b_w


# ------------------------------------------------------------------ #
#  Operator helper                                                   #
# ------------------------------------------------------------------ #
def _build_1d_operator(n, m, r, n_gp, h, bc, scheme, verbose) -> sp.sparse.csr_matrix:
    match scheme:
        case FiniteDifferenceScheme.CENTRAL:
            return build_explicit_fd_matrix(n, m, r, n_gp, h, bc, "central", verbose)
        case FiniteDifferenceScheme.UPWIND:
            return build_explicit_fd_matrix(n, m, r, n_gp, h, bc, "upwind", verbose)
        case FiniteDifferenceScheme.DOWNWIND:
            return build_explicit_fd_matrix(n, m, r, n_gp, h, bc, "downwind", verbose)
        case FiniteDifferenceScheme.PADE:
            return build_pade_fd_matrix(n, m, r, h, bc, verbose)
        case FiniteDifferenceScheme.COMPACT:
            return build_pade_fd_matrix(n, m, 1, h, bc, verbose)


# ------------------------------------------------------------------ #
#  Grid Helper Functions                                             #
# ------------------------------------------------------------------ #
def _uniform_1d_grid_axis(
    a: float,
    b: float,
    n: int,
    bc: BoundaryCondition,
    n_gps: int,
) -> tuple[float, float, int, bool, float, int]:
    """
    Uniform 1D axis parameters consistent with ``Grid1d``.

    Returns ``(a_grid, b_grid, n_grid, endpoint, h)`` for ``np.linspace``.
    """
    if n < 2:
        raise ValueError(f"Axis grid requires n >= 2, got n={n}.")
    if not (b > a):
        raise ValueError(f"Axis grid requires b > a, got a={a}, b={b}.")
    if bc == BoundaryCondition.GHOST_POINTS and n_gps < 1:
        raise ValueError(f"Number of ghost points must be >= 1, got n_gps={n_gps}.")

    a0, b0, n0 = float(a), float(b), int(n)

    match bc:
        case BoundaryCondition.PERIODIC:
            return a0, b0, n0, False, (b0 - a0) / n0, 0
        case BoundaryCondition.DIRICHLET:
            return a0, b0, n0, True, (b0 - a0) / (n0 - 1), 0
        case BoundaryCondition.GHOST_POINTS:
            h = (b0 - a0) / (n0 - 1)
            n_grid = n0 + 2 * n_gps
            a_grid = a0 - n_gps * h
            b_grid = b0 + n_gps * h
            return a_grid, b_grid, n_grid, True, h, n_gps


# ------------------------------------------------------------------ #
#  Grid Classes                                                      #
# ------------------------------------------------------------------ #
class Grid1d:
    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        n: int = 100,
        *,
        bc: BoundaryCondition = BoundaryCondition.DIRICHLET,
        n_ghost_points: int = 0,
        scheme: FiniteDifferenceScheme = FiniteDifferenceScheme.CENTRAL,
        r_width: int = 1,
        verbose: bool = False,
    ):

        self.r = r_width  # stencil width
        self.bc = bc
        self.scheme = scheme
        self.verbose = verbose

        a_g, b_g, n_g, endpoint, h_g, n_gps = _uniform_1d_grid_axis(
            a, b, n, bc, n_ghost_points
        )

        self.a = a_g
        self.b = b_g
        self.n = n_g
        self.n_gps = n_gps
        self.h = h_g  # grid spacing
        self.inv_h = 1.0 / h_g

        self.x = np.linspace(
            start=a_g, stop=b_g, num=n_g, endpoint=endpoint, dtype=float
        )

        # Print short summary of the grid
        fields = {
            "x": f"[{self.a}, {self.b}]",
            "n": self.n,
            "h": f"{self.h:.6f}",
            "bc": self.bc.name,
            "[default]scheme": self.scheme.name,
            "r": self.r,
            "n_gps": self.n_gps,
        }
        body = ", ".join(f"{k}={v}" for k, v in fields.items())
        print(f"{type(self).__name__}({body})")

    # -------------------------------- #
    # First-order derivative operators #
    # -------------------------------- #
    @cached_property
    def Dx_upwind(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            1,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.UPWIND,
            self.verbose,
        )

    @cached_property
    def Dx_downwind(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            1,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.DOWNWIND,
            self.verbose,
        )

    @cached_property
    def Dx_central(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            1,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.CENTRAL,
            self.verbose,
        )

    @cached_property
    def Dx_pade(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            1,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.PADE,
            self.verbose,
        )

    @cached_property
    def Dx_compact(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            1,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.COMPACT,
            self.verbose,
        )

    # --------------------------------- #
    # Second-order derivative operators #
    # --------------------------------- #
    @cached_property
    def Dx2_central(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            2,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.CENTRAL,
            self.verbose,
        )

    @cached_property
    def Dx2_pade(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            2,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.PADE,
            self.verbose,
        )

    @cached_property
    def Dx2_compact(self) -> sp.sparse.csr_matrix:
        return _build_1d_operator(
            self.n,
            2,
            self.r,
            self.n_gps,
            self.h,
            self.bc,
            FiniteDifferenceScheme.COMPACT,
            self.verbose,
        )

    # ------------------------------------- #
    # Shortcuts to default scheme operators #
    # ------------------------------------- #
    @property
    def Dx(self) -> sp.sparse.csr_matrix:
        match self.scheme:
            case FiniteDifferenceScheme.UPWIND:
                return self.Dx_upwind
            case FiniteDifferenceScheme.DOWNWIND:
                return self.Dx_downwind
            case FiniteDifferenceScheme.CENTRAL:
                return self.Dx_central
            case FiniteDifferenceScheme.PADE:
                return self.Dx_pade
            case FiniteDifferenceScheme.COMPACT:
                return self.Dx_compact

    @property
    def Dx2(self) -> sp.sparse.csr_matrix:
        match self.scheme:
            case FiniteDifferenceScheme.CENTRAL:
                return self.Dx2_central
            case FiniteDifferenceScheme.PADE:
                return self.Dx2_pade
            case FiniteDifferenceScheme.COMPACT:
                return self.Dx2_compact

    # -------------------------------------------------- #
    # Shortcut to build a specific derivative operator   #
    # -------------------------------------------------- #
    def Derivative(
        self, scheme: FiniteDifferenceScheme, k_order: int, r_width: int
    ) -> sp.sparse.csr_matrix:
        if self.bc == BoundaryCondition.GHOST_POINTS and r_width > self.n_gps:
            raise ValueError(
                "Stencil width is incompatible with number of ghost points."
            )
        return _build_1d_operator(
            self.n,
            k_order,
            r_width,
            self.n_gps,
            self.h,
            self.bc,
            scheme,
            self.verbose,
        )


class Grid2d:
    """Uniform 2D tensor-product grid with finite-difference operators.

    Two independent ``Grid1d`` axes (``gx``, ``gy``) handle boundary
    conditions and stencils per direction.  Full 2D operators are assembled
    via Kronecker products of the 1D blocks.
    """

    def __init__(
        self,
        xa: float = 0.0,
        xb: float = 1.0,
        nx: int = 100,
        ya: float = 0.0,
        yb: float = 1.0,
        ny: int = 100,
        *,
        bc_x: BoundaryCondition = BoundaryCondition.DIRICHLET,
        bc_y: BoundaryCondition = BoundaryCondition.DIRICHLET,
        n_ghost_points_x: int = 0,
        n_ghost_points_y: int = 0,
        scheme: FiniteDifferenceScheme = FiniteDifferenceScheme.CENTRAL,
        r_width_x: int = 1,
        r_width_y: int = 1,
        verbose: bool = False,
    ):
        self.bcx = bc_x
        self.bcy = bc_y
        self.scheme = scheme
        self.verbose = verbose

        self.gx = Grid1d(
            a=xa,
            b=xb,
            n=nx,
            bc=bc_x,
            n_ghost_points=n_ghost_points_x,
            scheme=scheme,
            r_width=r_width_x,
            verbose=verbose,
        )
        self.gy = Grid1d(
            a=ya,
            b=yb,
            n=ny,
            bc=bc_y,
            n_ghost_points=n_ghost_points_y,
            scheme=scheme,
            r_width=r_width_y,
            verbose=verbose,
        )

        self.xa = self.gx.a
        self.xb = self.gx.b
        self.nx = self.gx.n
        self.n_gps_x = self.gx.n_gps
        self.hx = self.gx.h
        self.inv_hx = self.gx.inv_h
        self.x = self.gx.x
        self.rx = self.gx.r

        self.ya = self.gy.a
        self.yb = self.gy.b
        self.ny = self.gy.n
        self.n_gps_y = self.gy.n_gps
        self.hy = self.gy.h
        self.inv_hy = self.gy.inv_h
        self.y = self.gy.x
        self.ry = self.gy.r

    # ------------------------------------------- #
    # Methods for building the discrete operators #
    # ------------------------------------------- #
    @cached_property
    def Ix(self):
        return sp.sparse.eye(self.nx, format="csr")

    @cached_property
    def Iy(self):
        return sp.sparse.eye(self.ny, format="csr")

    @cached_property
    def Dx_operator(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.Iy, self.gx.Dx, format="csr")

    @cached_property
    def Dy_operator(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.gy.Dx, self.Ix, format="csr")

    @cached_property
    def Dxy_operator(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.gy.Dx, self.gx.Dx, format="csr")

    @cached_property
    def Dyx_operator(self) -> sp.sparse.csr_matrix:
        return self.Dxy_operator

    @cached_property
    def Dx2_operator(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.Iy, self.gx.Dx2, format="csr")

    @cached_property
    def Dy2_operator(self) -> sp.sparse.csr_matrix:
        return sp.sparse.kron(self.gy.Dx2, self.Ix, format="csr")

    @cached_property
    def grad(self):
        return sp.sparse.vstack([self.Dx_operator, self.Dy_operator], format="csr")

    @cached_property
    def div(self):
        return sp.sparse.hstack([self.Dx_operator, self.Dy_operator], format="csr")

    @cached_property
    def curl(self):
        return sp.sparse.hstack([-self.Dy_operator, self.Dx_operator], format="csr")

    @cached_property
    def laplacian(self):
        return self.Dx2_operator + self.Dy2_operator

    # -------------------------------------------------------- #
    # Shortcuts to perform operations using the default scheme #
    # -------------------------------------------------------- #
    @cached_property
    def Dx_1d_T(self) -> sp.sparse.csr_matrix:
        return self.gx.Dx.T

    @cached_property
    def Dy_1d(self) -> sp.sparse.csr_matrix:
        return self.gy.Dx

    @cached_property
    def Dx2_1d_T(self) -> sp.sparse.csr_matrix:
        return self.gx.Dx2.T

    @cached_property
    def Dy2_1d(self) -> sp.sparse.csr_matrix:
        return self.gy.Dx2

    def Dx(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_1d_T

    def Dy(self, u: np.ndarray) -> np.ndarray:
        return self.Dy_1d @ u

    def Dx2(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx2_1d_T

    def Dy2(self, u: np.ndarray) -> np.ndarray:
        return self.Dy2_1d @ u

    def Dxy(self, u: np.ndarray) -> np.ndarray:
        return self.Dy_1d @ (u @ self.Dx_1d_T)

    def Grad(self, sField: np.ndarray) -> list[np.ndarray]:
        return [self.Dx(sField), self.Dy(sField)]

    def Div(self, vField: list[np.ndarray]) -> np.ndarray:
        return self.Dx(vField[0]) + self.Dy(vField[1])

    def Curl(self, vField: list[np.ndarray]) -> np.ndarray:
        return self.Dy(vField[0]) - self.Dx(vField[1])

    def Laplacian(self, sField: np.ndarray) -> np.ndarray:
        return self.Dx2(sField) + self.Dy2(sField)

    # ------------------------------------------------------- #
    # Shortcuts to perform operations using a specific scheme #
    # ------------------------------------------------------- #
    @cached_property
    def Dx_upwind_1d_T(self) -> sp.sparse.csr_matrix:
        return self.gx.Dx_upwind.T

    def Dx_upwind(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_upwind_1d_T

    def Dy_upwind(self, u: np.ndarray) -> np.ndarray:
        return self.gy.Dx_upwind @ u

    def Grad_upwind(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_upwind(u), self.Dy_upwind(u)]

    def Div_upwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_upwind(v[0]) + self.Dy_upwind(v[1])

    def Curl_upwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_upwind(v[0]) - self.Dx_upwind(v[1])

    @cached_property
    def Dx_downwind_1d_T(self) -> sp.sparse.csr_matrix:
        return self.gx.Dx_downwind.T

    def Dx_downwind(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_downwind_1d_T

    def Dy_downwind(self, u: np.ndarray) -> np.ndarray:
        return self.gy.Dx_downwind @ u

    def Grad_downwind(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_downwind(u), self.Dy_downwind(u)]

    def Div_downwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_downwind(v[0]) + self.Dy_downwind(v[1])

    def Curl_downwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_downwind(v[0]) - self.Dx_downwind(v[1])

    @cached_property
    def Dx_central_1d_T(self) -> sp.sparse.csr_matrix:
        return self.gx.Dx_central.T

    def Dx_central(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_central_1d_T

    def Dy_central(self, u: np.ndarray) -> np.ndarray:
        return self.gy.Dx_central @ u

    def Grad_central(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_central(u), self.Dy_central(u)]

    def Div_central(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_central(v[0]) + self.Dy_central(v[1])

    def Curl_central(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_central(v[0]) - self.Dx_central(v[1])

    def Laplacian_central(self, u: np.ndarray) -> np.ndarray:
        return self.Dx2_central(u) + self.Dy2_central(u)

    @cached_property
    def Dx_pade_1d_T(self) -> sp.sparse.csr_matrix:
        return self.gx.Dx_pade.T

    def Dx_pade(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_pade_1d_T

    def Dy_pade(self, u: np.ndarray) -> np.ndarray:
        return self.gy.Dx_pade @ u

    def Grad_pade(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_pade(u), self.Dy_pade(u)]

    def Div_pade(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_pade(v[0]) + self.Dy_pade(v[1])

    def Curl_pade(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_pade(v[0]) - self.Dx_pade(v[1])

    def Laplacian_pade(self, u: np.ndarray) -> np.ndarray:
        return self.Dx2_pade(u) + self.Dy2_pade(u)

    @cached_property
    def Dx_compact_1d_T(self) -> sp.sparse.csr_matrix:
        return self.gx.Dx_compact.T

    def Dx_compact(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_compact_1d_T

    def Dy_compact(self, u: np.ndarray) -> np.ndarray:
        return self.gy.Dx_compact @ u

    def Grad_compact(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_compact(u), self.Dy_compact(u)]

    def Div_compact(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_compact(v[0]) + self.Dy_compact(v[1])

    def Curl_compact(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_compact(v[0]) - self.Dx_compact(v[1])

    def Laplacian_compact(self, u: np.ndarray) -> np.ndarray:
        return self.Dx2_compact(u) + self.Dy2_compact(u)
