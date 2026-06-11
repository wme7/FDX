"""
Weighted Essentially Non-Oscillatory Grid Module.

This module provides classes for building uniform Cartesian grids and
nonlinear ENO/WENO spatial operators.

The class ``Grid1d`` is a 1D grid with uniform spacing.  It provides
upwind-biased first-derivative operators via WENO5 or CRWENO5 reconstruction
at cell interfaces:

- ``Dx_upwind`` / ``Dx_downwind``: left- and right-biased ∂/∂x

The class ``Grid2d`` is a 2D tensor-product grid built from two independent
``Grid1d`` axes.  Because WENO/CRWENO operators are state-dependent and
nonlinear, 2D derivatives are applied by dimension splitting (row/column
slices), not by Kronecker products of fixed sparse matrices:

- ``Dx_upwind`` / ``Dx_downwind``: ∂/∂x along each row (axis ``x``)
- ``Dy_upwind`` / ``Dy_downwind``: ∂/∂y along each column (axis ``y``)
- ``Derivative(u, axis, bias)``: convenience dispatcher for the above

Field layout matches ``finite_differences_grid.Grid2d``: arrays are
``(ny, nx)`` with row-major flattening ``index = j * nx + i``.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from functools import cached_property

import numpy as np
import scipy as sp

from .eno_weights import (
    fd_eno_substencil_weights,
    fd_eno_weights,
    fd_smooth_indicator_weights,
)
from .utils import build_lerp_boundaries_matrix as lerp_oper
from .utils import build_mirror_symmetric_operator as mirror_symmetric
from .utils import build_periodic_banded_matrix as periodic_oper
from .utils import build_periodic_tridiagonal_matrix as periodic_tridiag_oper
from .utils import build_rectangular_banded_matrix as rect_oper
from .utils import build_rectangular_tridiagonal_matrix as rect_tridiag_oper


# ------------------------------------------------------------------ #
#  ENO/WENO Operators Parameters                                     #
# ------------------------------------------------------------------ #
class BoundaryCondition(Enum):
    PERIODIC = auto()
    DIRICHLET = auto()
    DIRICHLET_HOMOGENEOUS = auto()
    GHOST_POINTS = auto()


class NonOscillatoryScheme(Enum):
    WENO5 = auto()
    CRWENO5 = auto()


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
    """Uniform 1D grid axis parameters consistent with `Grid1d`.

    Returns `(a_grid, b_grid, n_grid, endpoint, h, n_ghost_points)` for `np.linspace`.
    """
    if n < 2:
        raise ValueError(f"Axis grid requires n >= 2, got n={n}.")
    if not (b > a):
        raise ValueError(f"Axis grid requires b > a, got a={a}, b={b}.")
    if bc == BoundaryCondition.GHOST_POINTS and n_gps < 3:
        raise ValueError(f"Number of ghost points must be >= 3, got n_gps={n_gps}.")

    a0, b0, n0 = float(a), float(b), int(n)

    match bc:
        case BoundaryCondition.PERIODIC:
            return a0, b0, n0, False, (b0 - a0) / n0, 0
        case BoundaryCondition.DIRICHLET:
            return a0, b0, n0, True, (b0 - a0) / (n0 - 1), 2  # 2 virtual ghost points
        case BoundaryCondition.GHOST_POINTS:
            h = (b0 - a0) / (n0 - 1)
            n_grid = n0 + 2 * n_gps
            a_grid = a0 - n_gps * h
            b_grid = b0 + n_gps * h
            return a_grid, b_grid, n_grid, True, h, 2


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
        scheme: NonOscillatoryScheme = NonOscillatoryScheme.WENO5,
        r_width: int = 3,
        bc: BoundaryCondition = BoundaryCondition.DIRICHLET,
        n_ghost_points: int = 0,
        verbose: bool = False,
        axis_name: str = "x",
    ):

        self.r = 3  # stencil width, fix for the time being
        self.bc = bc
        self.verbose = verbose

        xmin, xmax, n_new, endpoint, h, n_gps = _uniform_1d_grid_axis(
            a,
            b,
            n,
            bc,
            n_gps=3,  # fix for the time being
        )

        self.a = a
        self.b = b
        self.n = n_new
        self.min = xmin
        self.max = xmax
        self.n_gps = n_gps
        self.h = h  # grid spacing
        self.inv_h = 1.0 / h
        self.scheme = scheme

        self.nodes = np.linspace(
            start=xmin, stop=xmax, num=n_new, endpoint=endpoint, dtype=float
        )

        # Default constants
        self.ϵ = 1e-6

        # print short summary of the grid
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

    # ----------------------------------------- #
    # Cached properties to build 1-d operators  #
    # ----------------------------------------- #
    @cached_property
    def linearly_interpolate_boundaries(self) -> sp.sparse.csr_matrix:
        return lerp_oper(self.n, self.n_gps)

    @cached_property
    def weno5_weights(self) -> np.ndarray:
        return fd_eno_weights(self.r)

    @cached_property
    def weno5_smooth_indicators_weights(self) -> list[np.ndarray]:
        return fd_smooth_indicator_weights(self.r)

    @cached_property
    def crweno5_boundary_weights(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        outer_L = fd_eno_substencil_weights(self.r, 0)
        inner_L = fd_eno_substencil_weights(self.r + 1, 1)
        inner_R = fd_eno_substencil_weights(self.r + 1, 2)
        outer_R = fd_eno_substencil_weights(self.r, 2)

        if self.verbose:
            np.set_printoptions(precision=3, linewidth=120)
            print(f"outer_L:\t{outer_L}")
            print(f"inner_L:\t{inner_L}")
            print(f"inner_R:\t{inner_R}")
            print(f"outer_R:\t{outer_R}")

        return outer_L, inner_L, inner_R, outer_R

    @cached_property
    def smoothness_indicator_stack(self) -> sp.sparse.csr_matrix:
        """Stack of smoothness indicator matrices.

        Returns a stack of smoothness indicator matrices of
        shape (6 * m, n_grid), where m is the output length.
        """
        n_grid = self.n
        n_gps = self.n_gps
        bc = self.bc
        α = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        I0, I1, I2 = fd_smooth_indicator_weights(self.r)

        # banded matrices for the smoothness indicators
        if bc == BoundaryCondition.PERIODIC:
            β00 = periodic_oper(n_grid, α[0], I0[0])
            β10 = periodic_oper(n_grid, α[1], I1[0])
            β20 = periodic_oper(n_grid, α[2], I2[0])
            β01 = periodic_oper(n_grid, α[0], I0[1])
            β11 = periodic_oper(n_grid, α[1], I1[1])
            β21 = periodic_oper(n_grid, α[2], I2[1])

        elif bc == BoundaryCondition.DIRICHLET:
            β00 = rect_oper(n_grid, α[0], I0[0], n_gps)
            β10 = rect_oper(n_grid, α[1], I1[0], n_gps)
            β20 = rect_oper(n_grid, α[2], I2[0], n_gps)
            β01 = rect_oper(n_grid, α[0], I0[1], n_gps)
            β11 = rect_oper(n_grid, α[1], I1[1], n_gps)
            β21 = rect_oper(n_grid, α[2], I2[1], n_gps)

        elif bc == BoundaryCondition.GHOST_POINTS:
            n_interior = n_grid - 2 * n_gps
            β00 = rect_oper(n_interior, α[0], I0[0], n_gps)
            β10 = rect_oper(n_interior, α[1], I1[0], n_gps)
            β20 = rect_oper(n_interior, α[2], I2[0], n_gps)
            β01 = rect_oper(n_interior, α[0], I0[1], n_gps)
            β11 = rect_oper(n_interior, α[1], I1[1], n_gps)
            β21 = rect_oper(n_interior, α[2], I2[1], n_gps)

        else:
            raise ValueError(f"Unsupported BoundaryCondition: {bc!r}")

        if self.verbose:
            np.set_printoptions(precision=2, linewidth=120)
            print(f"β00 ({β00.shape}):\n{β00.toarray()}")
            print(f"β01 ({β01.shape}):\n{β01.toarray()}")
            print(f"β10 ({β10.shape}):\n{β10.toarray()}")
            print(f"β11 ({β11.shape}):\n{β11.toarray()}")
            print(f"β20 ({β20.shape}):\n{β20.toarray()}")
            print(f"β21 ({β21.shape}):\n{β21.toarray()}")

        return sp.sparse.vstack([β00, β10, β20, β01, β11, β21], format="csr")

    def smoothness_indicators_js(self, u: np.ndarray, side: str) -> list[np.ndarray]:
        match side:
            case "left" | "L":
                d0, d1, d2 = 0.1, 0.6, 0.3
            case "right" | "R":
                d0, d1, d2 = 0.3, 0.6, 0.1
            case _:
                raise ValueError(f"Invalid side: {side!r}")

        betas = (self.smoothness_indicator_stack @ u).reshape(6, -1)
        β0 = betas[0] ** 2 + betas[3] ** 2 + self.ϵ
        β1 = betas[1] ** 2 + betas[4] ** 2 + self.ϵ
        β2 = betas[2] ** 2 + betas[5] ** 2 + self.ϵ
        α0, α1, α2 = d0 / (β0 * β0), d1 / (β1 * β1), d2 / (β2 * β2)
        inv_total = 1.0 / (α0 + α1 + α2)
        w0, w1, w2 = α0 * inv_total, α1 * inv_total, α2 * inv_total
        return [w0, w1, w2]

    @cached_property
    def weno5_left_substencils_stack(self) -> sp.sparse.csr_matrix:

        if self.scheme != NonOscillatoryScheme.WENO5:
            raise AttributeError(
                f"weno5_left_substencils is only defined for WENO5, got {self.scheme!r}"
            )

        n_grid, n_gps, bc = self.n, self.n_gps, self.bc
        α = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        ωL = fd_eno_weights(self.r)[:3]

        # banded matrices for the WENO5 substencils
        match bc:
            case BoundaryCondition.PERIODIC:
                sL0 = periodic_oper(n_grid, α[0], ωL[0])
                sL1 = periodic_oper(n_grid, α[1], ωL[1])
                sL2 = periodic_oper(n_grid, α[2], ωL[2])
            case BoundaryCondition.DIRICHLET | BoundaryCondition.DIRICHLET_HOMOGENEOUS:
                sL0 = rect_oper(n_grid, α[0], ωL[0], n_gps, r_reset=1)
                sL1 = rect_oper(n_grid, α[1], ωL[1], n_gps, r_reset=1)
                sL2 = rect_oper(n_grid, α[2], ωL[2], n_gps, r_reset=1)
            case BoundaryCondition.GHOST_POINTS:
                n_interior = n_grid - 2 * n_gps
                sL0 = rect_oper(n_interior, α[0], ωL[0], n_gps, r_reset=1)
                sL1 = rect_oper(n_interior, α[1], ωL[1], n_gps, r_reset=1)
                sL2 = rect_oper(n_interior, α[2], ωL[2], n_gps, r_reset=1)

        if self.verbose:
            np.set_printoptions(precision=2, linewidth=120)
            print(f"sL0 ({sL0.shape}):\n{sL0.toarray()}")
            print(f"sL1 ({sL1.shape}):\n{sL1.toarray()}")
            print(f"sL2 ({sL2.shape}):\n{sL2.toarray()}")

        return sp.sparse.vstack([sL0, sL1, sL2], format="csr")

    @cached_property
    def weno5_right_substencils_stack(self) -> sp.sparse.csr_matrix:
        # Mirror left substencils: (sR0←sL2, sR1←sL1, sR2←sL0).
        return mirror_symmetric(self.weno5_left_substencils_stack)

    def pointwise_eval_weno5_substencil(self, u: np.ndarray, side: str) -> float:
        """Pointwise evaluation of the WENO5 substencil.

        Args:
            u: A length-5 local stencil (indices j-2…j+2 relative to j-point).
            side: The side of the stencil to evaluate.

        Returns:
            The pointwise evaluation of the WENO5 substencil.

        Example:
            >>> u_stencil = u[j-2:j+3]
            >>> uL_j_half = grid.pointwise_eval_weno5_substencil(u_stencil, "L")
            >>> uR_j_half = grid.pointwise_eval_weno5_substencil(u_stencil, "R")
        """
        match side:
            case "left" | "L":
                d0, d1, d2 = 0.1, 0.6, 0.3
                s0, s1, s2 = self.weno5_weights[:3]
            case "right" | "R":
                d0, d1, d2 = 0.3, 0.6, 0.1
                s0, s1, s2 = self.weno5_weights[1:]
            case _:
                raise ValueError(f"Invalid side: {side!r}")

        u0, u1, u2 = u[0:3], u[1:4], u[2:5]
        I0, I1, I2 = self.weno5_smooth_indicators_weights
        beta00, beta01 = np.dot(u0, I0[0]), np.dot(u0, I0[1])
        beta10, beta11 = np.dot(u1, I1[0]), np.dot(u1, I1[1])
        beta20, beta21 = np.dot(u2, I2[0]), np.dot(u2, I2[1])
        β0 = beta00 * beta00 + beta01 * beta01 + self.ϵ
        β1 = beta10 * beta10 + beta11 * beta11 + self.ϵ
        β2 = beta20 * beta20 + beta21 * beta21 + self.ϵ
        α0, α1, α2 = d0 / (β0 * β0), d1 / (β1 * β1), d2 / (β2 * β2)
        inv_total = 1.0 / (α0 + α1 + α2)
        w0, w1, w2 = α0 * inv_total, α1 * inv_total, α2 * inv_total

        return w0 * np.dot(u0, s0) + w1 * np.dot(u1, s1) + w2 * np.dot(u2, s2)

    def _ghost_weno5_stencil(self, v: np.ndarray, interior_row: int) -> np.ndarray:
        """Length-5 stencil ``v[j-2:j+3]`` for an interior-system row index.

        Interior row ``r`` maps to the full-grid cell index ``j = n_gps + r``.
        Negative indices follow the usual Python convention on the interior
        block (length ``n - 2*n_gps``).
        """
        n_interior = self.n - 2 * self.n_gps
        j = self.n_gps + (
            interior_row if interior_row >= 0 else n_interior + interior_row
        )
        return v[j - 2 : j + 3]

    def smoothness_indicators_gb(self, u: np.ndarray, side: str) -> list[np.ndarray]:
        match side:
            case "left" | "L":
                d0, d1, d2 = 0.2, 0.5, 0.3
            case "right" | "R":
                d0, d1, d2 = 0.3, 0.5, 0.2
            case _:
                raise ValueError(f"Invalid side: {side!r}")

        betas = (self.smoothness_indicator_stack @ u).reshape(6, -1)
        β0 = betas[0] ** 2 + betas[3] ** 2 + self.ϵ
        β1 = betas[1] ** 2 + betas[4] ** 2 + self.ϵ
        β2 = betas[2] ** 2 + betas[5] ** 2 + self.ϵ

        α0, α1, α2 = d0 / (β0 * β0), d1 / (β1 * β1), d2 / (β2 * β2)
        inv_total = 1.0 / (α0 + α1 + α2)
        w0, w1, w2 = α0 * inv_total, α1 * inv_total, α2 * inv_total

        match side:
            case "left" | "L":
                a0 = (2.0 * w0 + w1) / 3.0
                a1 = (w0 + 2.0 * w1 + 2.0 * w2) / 3.0
                a2 = w2 / 3.0
                b0 = w0 / 6.0
                b1 = (5.0 * w0 + 5.0 * w1 + w2) / 6.0
                b2 = (w1 + 5.0 * w2) / 6.0
            case "right" | "R":
                a0 = w0 / 3.0
                a1 = (w2 + 2.0 * w1 + 2.0 * w0) / 3.0
                a2 = (2.0 * w2 + w1) / 3.0
                b0 = (w1 + 5.0 * w0) / 6.0
                b1 = (5.0 * w2 + 5.0 * w1 + w0) / 6.0
                b2 = w2 / 6.0

        return [a0, a1, a2, b0, b1, b2]

    def left_WENO5_reconstruction(self, u: np.ndarray) -> np.ndarray:
        # Interpolate the solution to the left and right interfaces
        if self.bc == BoundaryCondition.DIRICHLET:
            v = self.linearly_interpolate_boundaries @ u
        else:
            v = u

        # Evaluate the left-biased WENO reconstruction
        w0L, w1L, w2L = self.smoothness_indicators_js(v, "L")
        sL = (self.weno5_left_substencils_stack @ v).reshape(3, -1)
        return w0L * sL[0] + w1L * sL[1] + w2L * sL[2]

    def right_WENO5_reconstruction(self, u: np.ndarray) -> np.ndarray:
        # Interpolate the solution to the left and right interfaces
        if self.bc == BoundaryCondition.DIRICHLET:
            v = self.linearly_interpolate_boundaries @ u
        else:
            v = u

        # Evaluate the right-biased WENO reconstruction
        w0R, w1R, w2R = self.smoothness_indicators_js(v, "R")
        sR = (self.weno5_right_substencils_stack @ v).reshape(3, -1)
        return w0R * sR[0] + w1R * sR[1] + w2R * sR[2]

    def left_CRWENO5_reconstruction(self, u: np.ndarray) -> np.ndarray:
        # Interpolate the solution to the left and right interfaces
        if self.bc == BoundaryCondition.DIRICHLET:
            v = self.linearly_interpolate_boundaries @ u
        else:
            v = u

        # Compute the smoothness indicators
        a0, a1, a2, b0, b1, b2 = self.smoothness_indicators_gb(v, "L")

        # Build A_ij and B_ij matrices
        match self.bc:
            case BoundaryCondition.PERIODIC:
                A = periodic_tridiag_oper(a0, a1, a2)
                rhs = b0 * np.roll(v, 1) + b1 * v + b2 * np.roll(v, -1)
            case BoundaryCondition.DIRICHLET:
                outer_L, inner_L, inner_R, outer_R = self.crweno5_boundary_weights
                a0[0], a1[0], a2[0] = 0, 1, 0
                a0[1], a1[1], a2[1] = 0, 1, 0
                a0[-2], a1[-2], a2[-2] = 0, 1, 0
                a0[-1], a1[-1], a2[-1] = 0, 1, 0
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=2)
                rhs = B @ v
                rhs[0] = np.dot(outer_L, u[:3])
                rhs[1] = np.dot(inner_L, u[:4])
                rhs[-2] = np.dot(inner_R, u[-4:])
                rhs[-1] = np.dot(outer_R, u[-3:])
            case BoundaryCondition.DIRICHLET_HOMOGENEOUS:
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=2)
                rhs = B @ v
            case BoundaryCondition.GHOST_POINTS:
                ng = self.n_gps
                a0[0], a1[0], a2[0] = 0, 1, 0
                a0[1], a1[1], a2[1] = 0, 1, 0
                a0[-2], a1[-2], a2[-2] = 0, 1, 0
                a0[-1], a1[-1], a2[-1] = 0, 1, 0
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=ng)
                rhs = B @ v
                for row in (0, 1, -2, -1):
                    rhs[row] = self.pointwise_eval_weno5_substencil(
                        self._ghost_weno5_stencil(v, row), "L"
                    )

        # if DEBUG:
        #     np.set_printoptions(precision=2, linewidth=120)
        #     print(f"A ({A.shape}):\n{A.toarray()}")
        #     print(f"B ({B.shape}):\n{B.toarray()}")

        return sp.sparse.linalg.splu(A.tocsc()).solve(rhs)

    def right_CRWENO5_reconstruction(self, u: np.ndarray) -> np.ndarray:
        # Interpolate the solution to the left and right interfaces
        if self.bc == BoundaryCondition.DIRICHLET:
            v = self.linearly_interpolate_boundaries @ u
        else:
            v = u

        # Compute the smoothness indicators
        a0, a1, a2, b0, b1, b2 = self.smoothness_indicators_gb(v, "R")

        # Build A_ij and B_ij matrices
        match self.bc:
            case BoundaryCondition.PERIODIC:
                A = periodic_tridiag_oper(a0, a1, a2)
                rhs = b0 * np.roll(v, 1) + b1 * v + b2 * np.roll(v, -1)
            case BoundaryCondition.DIRICHLET:
                outer_L, inner_L, inner_R, _ = self.crweno5_boundary_weights
                a0[1], a1[1], a2[1] = 0, 1, 0
                a0[2], a1[2], a2[2] = 0, 1, 0
                a0[-2], a1[-2], a2[-2] = 0, 1, 0
                a0[-1], a1[-1], a2[-1] = 0, 1, 0
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=2)
                rhs = B @ v
                rhs[1] = np.dot(outer_L, u[:3])
                rhs[2] = np.dot(inner_L, u[:4])
                rhs[-2] = np.dot(inner_L, u[-4:])
                rhs[-1] = np.dot(inner_R, u[-4:])
            case BoundaryCondition.DIRICHLET_HOMOGENEOUS:
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=2)
                rhs = B @ v
            case BoundaryCondition.GHOST_POINTS:
                ng = self.n_gps
                a0[1], a1[1], a2[1] = 0, 1, 0
                a0[2], a1[2], a2[2] = 0, 1, 0
                a0[-2], a1[-2], a2[-2] = 0, 1, 0
                a0[-1], a1[-1], a2[-1] = 0, 1, 0
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=ng)
                rhs = B @ v
                for row in (1, 2, -2, -1):
                    rhs[row] = self.pointwise_eval_weno5_substencil(
                        self._ghost_weno5_stencil(v, row), "R"
                    )

        # if DEBUG
        #     np.set_printoptions(precision=2, linewidth=120)
        #     print(f"A ({A.shape}):\n{A.toarray()}")
        #     print(f"B ({B.shape}):\n{B.toarray()}")

        return sp.sparse.linalg.splu(A.tocsc()).solve(rhs)

    def Dx_rusanov(
        self, u: np.ndarray, flux_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        match self.scheme:
            case NonOscillatoryScheme.WENO5:
                uL = self.left_WENO5_reconstruction(u)
                uR = self.right_WENO5_reconstruction(u)
            case NonOscillatoryScheme.CRWENO5:
                uL = self.left_CRWENO5_reconstruction(u)
                uR = self.right_CRWENO5_reconstruction(u)

        # uL[j] = u^L_{j+1/2}; uR[j] = u^R_{j-1/2}.
        # Rusanov flux at j+1/2 pairs uL[j] with uR[j+1].
        match self.bc:
            case BoundaryCondition.PERIODIC:
                uR_if = np.roll(uR, shift=-1)
            case _:
                uR_if = np.empty_like(uR)
                uR_if[:-1] = uR[1:]
                uR_if[-1] = uR[-1]

        fL = flux_fn(uL)
        fR = flux_fn(uR_if)

        match self.bc:
            case BoundaryCondition.PERIODIC:
                lambda_max = np.maximum(np.abs(uL), np.abs(uR_if))
                f = 0.5 * (fL + fR) - 0.5 * lambda_max * (uR_if - uL)
                return (f - np.roll(f, shift=1)) * self.inv_h
            case BoundaryCondition.DIRICHLET | BoundaryCondition.DIRICHLET_HOMOGENEOUS:
                lambda_max = np.maximum(np.abs(uL), np.abs(uR_if))
                f = 0.5 * (fL + fR) - 0.5 * lambda_max * (uR_if - uL)
                return np.pad(f[1:-1] - f[:-2], (1, 1), mode="constant") * self.inv_h
            case BoundaryCondition.GHOST_POINTS:
                lambda_max = np.maximum(np.abs(uL), np.abs(uR_if))
                f = 0.5 * (fL + fR) - 0.5 * lambda_max * (uR_if - uL)
                return (
                    np.pad(f[1:-1] - f[:-2], (self.r, self.r), mode="constant")
                    * self.inv_h
                )

    def Dx_upwind(self, u: np.ndarray) -> np.ndarray:
        match self.scheme:
            case NonOscillatoryScheme.WENO5:
                uL = self.left_WENO5_reconstruction(u)
            case NonOscillatoryScheme.CRWENO5:
                uL = self.left_CRWENO5_reconstruction(u)

        # Compute the derivative for the left-biased stencil
        match self.bc:
            case BoundaryCondition.PERIODIC:
                return (uL - np.roll(uL, shift=1)) * self.inv_h
            case BoundaryCondition.DIRICHLET | BoundaryCondition.DIRICHLET_HOMOGENEOUS:
                return np.pad(uL[1:-1] - uL[:-2], (1, 1), mode="constant") * self.inv_h
            case BoundaryCondition.GHOST_POINTS:
                return (
                    np.pad(uL[1:-1] - uL[:-2], (self.r, self.r), mode="constant")
                    * self.inv_h
                )

    def Dx_downwind(self, u: np.ndarray) -> np.ndarray:
        match self.scheme:
            case NonOscillatoryScheme.WENO5:
                uR = self.right_WENO5_reconstruction(u)
            case NonOscillatoryScheme.CRWENO5:
                uR = self.right_CRWENO5_reconstruction(u)

        # Compute the derivative for the right-biased stencil
        match self.bc:
            case BoundaryCondition.PERIODIC:
                return (np.roll(uR, shift=-1) - uR) * self.inv_h
            case BoundaryCondition.DIRICHLET | BoundaryCondition.DIRICHLET_HOMOGENEOUS:
                return np.pad(uR[2:] - uR[1:-1], (1, 1), mode="constant") * self.inv_h
            case BoundaryCondition.GHOST_POINTS:
                return (
                    np.pad(uR[2:] - uR[1:-1], (self.r, self.r), mode="constant")
                    * self.inv_h
                )


def _apply_1d_derivative(
    grid_1d: Grid1d,
    method: str,
    u: np.ndarray,
    axis: str,
    *,
    workers: int | None = None,
) -> np.ndarray:
    """Apply a 1D ``Grid1d`` derivative method slice-wise on a 2D field.

    Parameters
    ----------
    grid_1d : Grid1d
        1D grid whose ``method`` (e.g. ``Dx_upwind``) is applied to each slice.
    method : str
        Name of the ``Grid1d`` derivative method.
    u : (ny, nx) array
        2D field in row-major layout.
    axis : {"x", "y"}
        ``"x"`` differentiates along columns (last axis); ``"y"`` along rows.
    workers : int or None, optional
        If ``None`` or ``<= 1``, slices are processed sequentially.  Otherwise
        a :class:`~concurrent.futures.ThreadPoolExecutor` runs up to ``workers``
        slices in parallel (independent rows/columns).
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")

    fn = getattr(grid_1d, method)
    out = np.empty_like(u, dtype=float)

    if axis == "x":

        def work(k: int) -> None:
            out[k, :] = fn(u[k, :])

        n_slices = u.shape[0]
    elif axis == "y":

        def work(j: int) -> None:
            out[:, j] = fn(u[:, j])

        n_slices = u.shape[1]
    else:
        raise ValueError(f"Invalid axis: {axis!r}")

    if workers is None or workers <= 1:
        for i in range(n_slices):
            work(i)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(work, range(n_slices)))

    return out


class Grid2d:
    """Uniform 2D tensor-product grid with WENO/CRWENO upwind derivatives.

    Two independent ``Grid1d`` axes (``gx``, ``gy``) handle boundary
    conditions and reconstruction per direction.  Nonlinear operators are
    applied by looping over rows (x) or columns (y).  Set ``workers`` to
    parallelize independent slices with a thread pool (default: sequential).
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
        r_width_x: int = 3,
        r_width_y: int = 3,
        bc_x: BoundaryCondition = BoundaryCondition.DIRICHLET,
        bc_y: BoundaryCondition = BoundaryCondition.DIRICHLET,
        n_ghost_points_x: int = 0,
        n_ghost_points_y: int = 0,
        verbose: bool = False,
        workers: int | None = None,
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
        self.workers = workers if workers is not None else 1

    # ------------------------------------------- #
    # Methods for building the discrete operators #
    # ------------------------------------------- #
    @cached_property
    def X(self) -> np.ndarray:
        """Node coordinates, shape ``(ny, nx)``."""
        return self.x.nodes[np.newaxis, :].repeat(self.y.n, axis=0)

    @cached_property
    def Y(self) -> np.ndarray:
        """Node coordinates, shape ``(ny, nx)``."""
        return self.y.nodes[:, np.newaxis].repeat(self.x.n, axis=1)

    def _check_shape(self, u: np.ndarray) -> None:
        if u.shape != (self.x.n, self.y.n):
            raise ValueError(
                f"Expected field shape ({self.x.n}, {self.y.n}), got {u.shape}."
            )

    def _derivative_workers(self, workers: int | None) -> int | None:
        return self.workers if workers is None else workers

    def Dx_upwind(self, u: np.ndarray, *, workers: int | None = None) -> np.ndarray:
        """Left-biased d/dx applied row-wise along the x-axis."""
        self._check_shape(u)
        return _apply_1d_derivative(
            self.x, "Dx_upwind", u, axis="x", workers=self._derivative_workers(workers)
        )

    def Dx_downwind(self, u: np.ndarray, *, workers: int | None = None) -> np.ndarray:
        """Right-biased d/dx applied row-wise along the x-axis."""
        self._check_shape(u)
        return _apply_1d_derivative(
            self.x,
            "Dx_downwind",
            u,
            axis="x",
            workers=self._derivative_workers(workers),
        )

    def Dy_upwind(self, u: np.ndarray, *, workers: int | None = None) -> np.ndarray:
        """Left-biased d/dy applied column-wise along the y-axis."""
        self._check_shape(u)
        return _apply_1d_derivative(
            self.y, "Dx_upwind", u, axis="y", workers=self._derivative_workers(workers)
        )

    def Dy_downwind(self, u: np.ndarray, *, workers: int | None = None) -> np.ndarray:
        """Right-biased d/dy applied column-wise along the y-axis."""
        self._check_shape(u)
        return _apply_1d_derivative(
            self.y,
            "Dx_downwind",
            u,
            axis="y",
            workers=self._derivative_workers(workers),
        )
