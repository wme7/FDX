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
from .taylor_table_weights import fd_pade_weights
from .utils import build_square_banded_matrix, ensure_sparse


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
    a_weights, b_weights = fd_pade_weights(
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
        a_w, b_w = fd_pade_weights(m=m_derivative, alpha=α, beta=β)
        A[r, : len(α)] = a_w
        B[r, : len(β)] = b_w

        # -- bottom boundary:
        α = [-1, 0, 1] if r != 0 else [-1, 0]
        β = list(range(-(b_stencil_size - r - 1), r + 1))
        a_w, b_w = fd_pade_weights(m=m_derivative, alpha=α, beta=β)
        A[-r - 1, -len(α) :] = a_w
        B[-r - 1, -len(β) :] = b_w


# ------------------------------------------------------------------ #
#  Grid Helper Functions                                             #
# ------------------------------------------------------------------ #
def _axis_params(
    a: float,
    b: float,
    n: int,
    bc: BoundaryCondition,
    n_gps: int,
) -> tuple[float, float, int, float, int]:
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

    match bc:
        case BoundaryCondition.PERIODIC:
            return a, b, n, (b - a) / n, 0
        case BoundaryCondition.DIRICHLET:
            return a, b, n, (b - a) / (n - 1), 0
        case BoundaryCondition.GHOST_POINTS:
            h = (b - a) / (n - 1)
            n_new = n + 2 * n_gps
            a_new = a - n_gps * h
            b_new = b + n_gps * h
            return a_new, b_new, n_new, h, n_gps


def _make_nodes(a: float, b: float, n: int, bc: BoundaryCondition) -> np.ndarray:
    endpoint = bc == (BoundaryCondition.DIRICHLET or BoundaryCondition.GHOST_POINTS)
    return np.linspace(start=a, stop=b, num=n, endpoint=endpoint, dtype=float)


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
        r_width: int = 1,
        bc: BoundaryCondition = BoundaryCondition.DIRICHLET,
        n_ghost_points: int = 0,
        verbose: bool = False,
        axis_name: str = "x",
    ):

        self.a = a  # physical domain start
        self.b = b  # physical domain end
        self.r = r_width  # stencil width
        self.bc = bc
        self.verbose = verbose

        self.min, self.max, self.n, self.h, self.n_gps = _axis_params(
            a, b, n, bc, n_ghost_points
        )
        self.nodes = _make_nodes(self.min, self.max, self.n, bc)
        self.inv_h = 1.0 / self.h

        # Print short summary of the grid
        fields = {
            axis_name: f"[{self.a}, {self.b}]",
            "n": self.n,
            "h": f"{self.h:.6f}",
            "r_width": self.r,
            "bc": self.bc.name,
            "n_gps": self.n_gps,
        }
        body = ", ".join(f"{k}={v}" for k, v in fields.items())
        print(f"{type(self).__name__}({body})")

    # --------------------- #
    # 1-d Operator Builder  #
    # --------------------- #
    def _build_1d_operator(
        self,
        k_order: int,
        scheme: FiniteDifferenceScheme,
        r_width: int | None = None,
        bc: BoundaryCondition | None = None,
        n_gps: int | None = None,
        verbose: bool | None = None,
    ) -> sp.sparse.csr_matrix:
        r = r_width or self.r
        bc = bc or self.bc
        n_gps = n_gps or self.n_gps
        verbose = verbose or self.verbose
        match scheme:
            case FiniteDifferenceScheme.UPWIND:
                return build_explicit_fd_matrix(
                    self.n,
                    k_order,
                    r,
                    n_gps,
                    self.h,
                    bc,
                    "upwind",
                    verbose,
                )
            case FiniteDifferenceScheme.DOWNWIND:
                return build_explicit_fd_matrix(
                    self.n,
                    k_order,
                    r,
                    n_gps,
                    self.h,
                    bc,
                    "downwind",
                    verbose,
                )
            case FiniteDifferenceScheme.CENTRAL:
                return build_explicit_fd_matrix(
                    self.n,
                    k_order,
                    r,
                    n_gps,
                    self.h,
                    bc,
                    "central",
                    verbose,
                )
            case FiniteDifferenceScheme.PADE:
                return build_pade_fd_matrix(self.n, k_order, r, self.h, bc, verbose)
            case FiniteDifferenceScheme.COMPACT:
                return build_pade_fd_matrix(self.n, k_order, 1, self.h, bc, verbose)

    # ---------------------------- #
    # Cached derivative operators  #
    # ---------------------------- #
    @cached_property
    def Dx_upwind_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(1, FiniteDifferenceScheme.UPWIND)

    @cached_property
    def Dx_downwind_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(1, FiniteDifferenceScheme.DOWNWIND)

    @cached_property
    def Dx_central_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(1, FiniteDifferenceScheme.CENTRAL)

    @cached_property
    def Dx_pade_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(1, FiniteDifferenceScheme.PADE)

    @cached_property
    def Dx_compact_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(1, FiniteDifferenceScheme.COMPACT)

    @cached_property
    def Dx2_central_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(2, FiniteDifferenceScheme.CENTRAL)

    @cached_property
    def Dx2_pade_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(2, FiniteDifferenceScheme.PADE)

    @cached_property
    def Dx2_compact_operator(self) -> sp.sparse.csr_matrix:
        return self._build_1d_operator(2, FiniteDifferenceScheme.COMPACT)

    # --------------------------------- #
    # Shortcuts to perform derivatives  #
    # --------------------------------- #
    def Dx_upwind(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_upwind_operator @ u

    def Dx_downwind(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_downwind_operator @ u

    def Dx_central(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_central_operator @ u

    def Dx_pade(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_pade_operator @ u

    def Dx_compact(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_compact_operator @ u

    def Dx2_central(self, u: np.ndarray) -> np.ndarray:
        return self.Dx2_central_operator @ u

    def Dx2_pade(self, u: np.ndarray) -> np.ndarray:
        return self.Dx2_pade_operator @ u

    def Dx2_compact(self, u: np.ndarray) -> np.ndarray:
        return self.Dx2_compact_operator @ u

    # -------------------------------------------------- #
    # Shortcut to build a specific derivative operator   #
    # -------------------------------------------------- #
    def apply_derivative(
        self,
        u: np.ndarray,
        *,
        k_order: int = 1,
        scheme: FiniteDifferenceScheme = FiniteDifferenceScheme.CENTRAL,
        r_width: int = 1,
    ) -> np.ndarray:
        return self._build_1d_operator(k_order, scheme, r_width) @ u


class Grid2d:
    """Uniform 2D tensor-product grid with finite-difference operators.

    Two independent ``Grid1d`` axes (``x``, ``y``) handle boundary
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
        r_width_x: int = 1,
        r_width_y: int = 1,
        bc_x: BoundaryCondition = BoundaryCondition.DIRICHLET,
        bc_y: BoundaryCondition = BoundaryCondition.DIRICHLET,
        n_ghost_points_x: int = 0,
        n_ghost_points_y: int = 0,
        verbose: bool = False,
    ):
        self.x = Grid1d(
            a=xa,
            b=xb,
            n=nx,
            r_width=r_width_x,
            bc=bc_x,
            n_ghost_points=n_ghost_points_x,
            verbose=verbose,
            axis_name="x",
        )
        self.y = Grid1d(
            a=ya,
            b=yb,
            n=ny,
            r_width=r_width_y,
            bc=bc_y,
            n_ghost_points=n_ghost_points_y,
            verbose=verbose,
            axis_name="y",
        )

    @cached_property
    def X(self) -> np.ndarray:
        """Node coordinates, shape ``(ny, nx)``."""
        return self.x.nodes[np.newaxis, :].repeat(self.y.n, axis=0)

    @cached_property
    def Y(self) -> np.ndarray:
        """Node coordinates, shape ``(ny, nx)``."""
        return self.y.nodes[:, np.newaxis].repeat(self.x.n, axis=1)

    # ------------------------------------------- #
    # Methods for building 2-D discrete operators #
    # ------------------------------------------- #
    @cached_property
    def Ix(self):
        return sp.sparse.eye(self.x.n, format="csr")

    @cached_property
    def Iy(self):
        return sp.sparse.eye(self.y.n, format="csr")

    def Dx_operator(self, scheme: FiniteDifferenceScheme) -> sp.sparse.csr_matrix:
        Dx_1d = self.x._build_1d_operator(k_order=1, scheme=scheme)
        return sp.sparse.kron(self.Iy, Dx_1d, format="csr")

    def Dy_operator(self, scheme: FiniteDifferenceScheme) -> sp.sparse.csr_matrix:
        Dy_1d = self.y._build_1d_operator(k_order=1, scheme=scheme)
        return sp.sparse.kron(Dy_1d, self.Ix, format="csr")

    def laplacian(self, scheme: FiniteDifferenceScheme) -> sp.sparse.csr_matrix:
        Dx2_1d = self.x._build_1d_operator(k_order=2, scheme=scheme)
        Dy2_1d = self.y._build_1d_operator(k_order=2, scheme=scheme)
        return sp.sparse.kron(Dy2_1d, self.Ix, format="csr") + sp.sparse.kron(
            self.Iy, Dx2_1d, format="csr"
        )

    # ------------------------------------------------------- #
    # Shortcuts to perform operations using a specific scheme #
    # ------------------------------------------------------- #
    @cached_property
    def Dx_upwind_1d_T(self) -> sp.sparse.csr_matrix:
        return self.x.Dx_upwind_operator.T

    def Dx_upwind(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_upwind_1d_T

    def Dy_upwind(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx_upwind_operator @ u

    def Grad_upwind(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_upwind(u), self.Dy_upwind(u)]

    def Div_upwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_upwind(v[0]) + self.Dy_upwind(v[1])

    def Curl_upwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_upwind(v[0]) - self.Dx_upwind(v[1])

    @cached_property
    def Dx_downwind_1d_T(self) -> sp.sparse.csr_matrix:
        return self.x.Dx_downwind_operator.T

    def Dx_downwind(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_downwind_1d_T

    def Dy_downwind(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx_downwind_operator @ u

    def Grad_downwind(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_downwind(u), self.Dy_downwind(u)]

    def Div_downwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_downwind(v[0]) + self.Dy_downwind(v[1])

    def Curl_downwind(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_downwind(v[0]) - self.Dx_downwind(v[1])

    @cached_property
    def Dx_central_1d_T(self) -> sp.sparse.csr_matrix:
        return self.x.Dx_central_operator.T

    def Dx_central(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_central_1d_T

    def Dy_central(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx_central_operator @ u

    def Dxy_central(self, u: np.ndarray) -> np.ndarray:
        return self.Dy_central(self.Dx_central(u))

    def Dyx_central(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_central(self.Dy_central(u))

    @cached_property
    def Dx2_central_1d_T(self) -> sp.sparse.csr_matrix:
        return self.x.Dx2_central_operator.T

    def Dx2_central(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx2_central_1d_T

    def Dy2_central(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx2_central_operator @ u

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
        return self.x.Dx_pade_operator.T

    def Dx_pade(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_pade_1d_T

    def Dy_pade(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx_pade_operator @ u

    def Dxy_pade(self, u: np.ndarray) -> np.ndarray:
        return self.Dy_pade(self.Dx_pade(u))

    def Dyx_pade(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_pade(self.Dy_pade(u))

    @cached_property
    def Dx2_pade_1d_T(self) -> sp.sparse.csr_matrix:
        return self.x.Dx2_pade_operator.T

    def Dx2_pade(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx2_pade_1d_T

    def Dy2_pade(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx2_pade_operator @ u

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
        return self.x.Dx_compact_operator.T

    def Dx_compact(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx_compact_1d_T

    def Dy_compact(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx_compact_operator @ u

    def Dxy_compact(self, u: np.ndarray) -> np.ndarray:
        return self.Dy_compact(self.Dx_compact(u))

    def Dyx_compact(self, u: np.ndarray) -> np.ndarray:
        return self.Dx_compact(self.Dy_compact(u))

    @cached_property
    def Dx2_compact_1d_T(self) -> sp.sparse.csr_matrix:
        return self.x.Dx2_compact_operator.T

    def Dx2_compact(self, u: np.ndarray) -> np.ndarray:
        return u @ self.Dx2_compact_1d_T

    def Dy2_compact(self, u: np.ndarray) -> np.ndarray:
        return self.y.Dx2_compact_operator @ u

    def Grad_compact(self, u: np.ndarray) -> list[np.ndarray]:
        return [self.Dx_compact(u), self.Dy_compact(u)]

    def Div_compact(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dx_compact(v[0]) + self.Dy_compact(v[1])

    def Curl_compact(self, v: list[np.ndarray]) -> np.ndarray:
        return self.Dy_compact(v[0]) - self.Dx_compact(v[1])

    def Laplacian_compact(self, u: np.ndarray) -> np.ndarray:
        return self.Dx2_compact(u) + self.Dy2_compact(u)


def _kron_3d_axis_linop(D_1d, *, nz, ny, nx, axis):
    """3D analogue of _kron_axis_linop.

    Flat layout is row-major over (nz, ny, nx).  The operator is applied
    along the chosen axis via a single tensordot / einsum, never
    forming the full (nx ny nz) x (nx ny nz) matrix.
    """
    import scipy.sparse.linalg as spla

    N = nz * ny * nx

    if axis == "x":  # I_z ⊗ I_y ⊗ D_x  (axis=2 of (nz, ny, nx))

        def matvec(v):
            U = np.asarray(v).reshape(nz, ny, nx)
            # Apply D_1d to the last axis: reshape to (nz*ny, nx), use D_1d @ U.T,
            # then transpose back.  Saves one explicit Python loop.
            U_flat = U.reshape(nz * ny, nx)
            return (D_1d @ U_flat.T).T.reshape(nz, ny, nx).ravel()

    elif axis == "y":  # I_z ⊗ D_y ⊗ I_x  (axis=1)

        def matvec(v):
            U = np.asarray(v).reshape(nz, ny, nx)
            # Apply D_1d to the middle axis: collapse (nz, nx) -> rows of D_1d @ Y.
            # Easiest: transpose to put y last, apply, then transpose back.
            Up = U.transpose(0, 2, 1).reshape(nz * nx, ny)
            Vp = (D_1d @ Up.T).T.reshape(nz, nx, ny)
            return Vp.transpose(0, 2, 1).ravel()

    elif axis == "z":  # D_z ⊗ I_y ⊗ I_x  (axis=0)

        def matvec(v):
            U = np.asarray(v).reshape(nz, ny, nx)
            # Apply D_1d to the first axis: collapse (ny, nx) -> D_1d @ Z.
            Up = U.reshape(nz, ny * nx)
            return (D_1d @ Up).ravel()

    else:
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")

    def matmat(V):
        k = V.shape[1]
        out = np.empty_like(V)
        for c in range(k):
            out[:, c] = matvec(V[:, c])
        return out

    return spla.LinearOperator(
        (N, N),
        matvec=matvec,
        matmat=matmat,
        dtype=float,
    )


class Grid3d:
    """Uniform tensor-product Cartesian grid in 3D.

    Mirrors Grid2d: separate axes, schemes.
    We do not perform any expensive Kronecker products.
    Instead, we use matrix-free versions.
    """

    def __init__(
        self,
        xa: float = 0.0,
        xb: float = 1.0,
        nx: int = 50,
        ya: float = 0.0,
        yb: float = 1.0,
        ny: int = 50,
        za: float = 0.0,
        zb: float = 1.0,
        nz: int = 50,
        r_width_x: int = 1,
        r_width_y: int = 1,
        r_width_z: int = 1,
        bc_x: BoundaryCondition = BoundaryCondition.DIRICHLET,
        bc_y: BoundaryCondition = BoundaryCondition.DIRICHLET,
        bc_z: BoundaryCondition = BoundaryCondition.DIRICHLET,
        n_ghost_points_x: int = 0,
        n_ghost_points_y: int = 0,
        n_ghost_points_z: int = 0,
        verbose: bool = False,
    ):
        self.x = Grid1d(
            a=xa,
            b=xb,
            n=nx,
            r_width=r_width_x,
            bc=bc_x,
            n_ghost_points=n_ghost_points_x,
            verbose=verbose,
        )
        self.y = Grid1d(
            a=ya,
            b=yb,
            n=ny,
            r_width=r_width_y,
            bc=bc_y,
            n_ghost_points=n_ghost_points_y,
            verbose=verbose,
        )
        self.z = Grid1d(
            a=za,
            b=zb,
            n=nz,
            r_width=r_width_z,
            bc=bc_z,
            n_ghost_points=n_ghost_points_z,
            verbose=verbose,
        )

    # ---------------------------- #
    # Cached derivative operators  #
    # ---------------------------- #

    @cached_property
    def Dx_central_linop_1d(self):
        D = self.x.Dx_central_operator
        return sp.sparse.linalg.aslinearoperator(ensure_sparse(D))

    @cached_property
    def Dy_central_linop_1d(self):
        D = self.y.Dx_central_operator
        return sp.sparse.linalg.aslinearoperator(ensure_sparse(D))

    @cached_property
    def Dz_central_linop_1d(self):
        D = self.z.Dx_central_operator
        return sp.sparse.linalg.aslinearoperator(ensure_sparse(D))

    @cached_property
    def Dx_linop_3d(self):
        return _kron_3d_axis_linop(
            self.Dx_linop_1d,
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
            axis="x",
        )

    @cached_property
    def Dy_linop_3d(self):
        return _kron_3d_axis_linop(
            self.Dy_linop_1d,
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
            axis="y",
        )

    @cached_property
    def Dz_linop_3d(self):
        return _kron_3d_axis_linop(
            self.Dz_linop_1d,
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
            axis="z",
        )

    # ------- convenience: per-axis derivative on (nz, ny, nx) arrays ------- #
    def Derivative(self, u: np.ndarray, axis: str) -> np.ndarray:
        """Apply ∂_x / ∂_y / ∂_z to a (nz, ny, nx) field via the linop path.

        Equivalent to the dense Kronecker matvec but without the O(n³)
        memory footprint of the sparse Kronecker form.
        """
        if axis == "x":
            return (self.Dx_linop_3d @ u.ravel()).reshape(self.nz, self.ny, self.nx)
        if axis == "y":
            return (self.Dy_linop_3d @ u.ravel()).reshape(self.nz, self.ny, self.nx)
        if axis == "z":
            return (self.Dz_linop_3d @ u.ravel()).reshape(self.nz, self.ny, self.nx)
        raise ValueError(f"Invalid axis: {axis!r}")
