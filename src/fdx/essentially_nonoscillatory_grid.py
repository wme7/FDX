"""
Weighted Essentially Non-Oscillatory Grid Module.

This module provides a class for building finite difference grids and operators.
"""

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
    GHOST_POINTS = auto()


class NonOscillatoryScheme(Enum):
    WENO5 = auto()
    CRWENO5 = auto()


# ------------------------------------------------------------------ #
#  Grid Classes                                                      #
# ------------------------------------------------------------------ #
def _uniform_1d_grid_axis(
    a: float,
    b: float,
    n: int,
    bc: BoundaryCondition,
    scheme: NonOscillatoryScheme,
    r: int,
) -> tuple[float, float, int, bool, float, int]:
    """Uniform 1D grid axis parameters consistent with `Grid1d`.

    Returns `(a_grid, b_grid, n_grid, endpoint, h, n_ghost_points)` for `np.linspace`.
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
            return a0, b0, n0, False, (b0 - a0) / n0, 0
        case BoundaryCondition.DIRICHLET:
            return a0, b0, n0, True, (b0 - a0) / (n0 - 1), r - 1
        case BoundaryCondition.GHOST_POINTS:
            # build a uniform grid with three ghost points on each side
            h = (b0 - a0) / (n0 - 1)
            n_grid = n0 + 2 * r
            a_grid = a0 - r * h
            b_grid = b0 + r * h
            return a_grid, b_grid, n_grid, True, h, r - 1
        case _:
            raise ValueError(f"Unsupported BoundaryCondition for axis grid: {bc!r}")


class Grid1d:
    def __init__(
        self,
        a: float = 0.0,
        b: float = 1.0,
        n: int = 100,
        *,
        bc: BoundaryCondition = BoundaryCondition.GHOST_POINTS,
        scheme: NonOscillatoryScheme = NonOscillatoryScheme.WENO5,
        verbose: bool = False,
    ):
        match scheme:
            case NonOscillatoryScheme.WENO5:
                r = 3
            case NonOscillatoryScheme.CRWENO5:
                r = 3
            case _:
                raise ValueError(f"Unsupported NonOscillatoryScheme: {scheme!r}")

        self.r = r  # stencil width
        self.bc = bc
        self.scheme = scheme
        self.verbose = verbose

        a_grid, b_grid, n_grid, endpoint, h, n_gps = _uniform_1d_grid_axis(
            a, b, n, bc, scheme, r
        )

        self.a = a_grid
        self.b = b_grid
        self.n = n_grid
        self.n_gps = n_gps

        self.x = np.linspace(
            start=a_grid, stop=b_grid, num=n_grid, endpoint=endpoint, dtype=float
        )
        self.h = h  # grid spacing
        self.inv_h = 1 / self.h

        # Constants
        self.ϵ = 1e-6

    @cached_property
    def linearly_interpolate_boundaries(self) -> sp.sparse.csr_matrix:
        return lerp_oper(self.n, self.r - 1)

    @cached_property
    def crweno5_boundary_weights(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        outer_L = fd_eno_substencil_weights(self.r, 0) * self.inv_h
        inner_L = fd_eno_substencil_weights(self.r + 1, 1) * self.inv_h
        inner_R = fd_eno_substencil_weights(self.r + 1, 2) * self.inv_h
        outer_R = fd_eno_substencil_weights(self.r, 2) * self.inv_h

        if self.verbose:
            np.set_printoptions(precision=3, linewidth=120)
            print(f"outer_L:\n{outer_L.toarray()}")
            print(f"inner_L:\n{inner_L.toarray()}")
            print(f"inner_R:\n{inner_R.toarray()}")
            print(f"outer_R:\n{outer_R.toarray()}")

        return outer_L, inner_L, inner_R, outer_R

    @cached_property
    def weno5_left_substencils(
        self,
    ) -> tuple[sp.sparse.csr_matrix, sp.sparse.csr_matrix, sp.sparse.csr_matrix]:

        if self.scheme != NonOscillatoryScheme.WENO5:
            raise AttributeError(
                f"weno5_left_substencils is only defined for WENO5, got {self.scheme!r}"
            )

        n_grid = self.n
        n_gps = self.n_gps
        bc = self.bc
        α = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        ωL = fd_eno_weights(self.r)[:3] * self.inv_h

        # banded matrices for the WENO5 substencils
        if bc == BoundaryCondition.PERIODIC:
            sL0 = periodic_oper(n_grid, α[0], ωL[0])
            sL1 = periodic_oper(n_grid, α[1], ωL[1])
            sL2 = periodic_oper(n_grid, α[2], ωL[2])

        elif bc == BoundaryCondition.DIRICHLET:
            sL0 = rect_oper(n_grid, α[0], ωL[0], n_gps, r_reset=1)
            sL1 = rect_oper(n_grid, α[1], ωL[1], n_gps, r_reset=1)
            sL2 = rect_oper(n_grid, α[2], ωL[2], n_gps, r_reset=1)

        elif bc == BoundaryCondition.GHOST_POINTS:
            n_interior = n_grid - 2 * n_gps
            sL0 = rect_oper(n_interior, α[0], ωL[0], n_gps, r_reset=1)
            sL1 = rect_oper(n_interior, α[1], ωL[1], n_gps, r_reset=1)
            sL2 = rect_oper(n_interior, α[2], ωL[2], n_gps, r_reset=1)

        else:
            raise ValueError(f"Unsupported BoundaryCondition: {bc!r}")

        if self.verbose:
            np.set_printoptions(precision=2, linewidth=120)
            print(f"sL0 ({sL0.shape}):\n{sL0.toarray()}")
            print(f"sL1 ({sL1.shape}):\n{sL1.toarray()}")
            print(f"sL2 ({sL2.shape}):\n{sL2.toarray()}")

        return sL0, sL1, sL2

    def pointwise_eval_weno5_left_substencils(self, u: np.ndarray) -> float:

        α = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        sL = fd_eno_weights(self.r)[:3] * self.inv_h
        wL = self.pointwise_eval_smoothness_indicators_js(u, "L")

        return (
            wL[0] * np.dot(u[α[0]], sL[0])
            + wL[1] * np.dot(u[α[1]], sL[1])
            + wL[2] * np.dot(u[α[2]], sL[2])
        )

    def pointwise_eval_weno5_right_substencils(self, u: np.ndarray) -> float:

        α = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        sR = fd_eno_weights(self.r)[1:] * self.inv_h
        wR = self.pointwise_eval_smoothness_indicators_js(u, "R")

        return (
            wR[0] * np.dot(u[α[0]], sR[0])
            + wR[1] * np.dot(u[α[1]], sR[1])
            + wR[2] * np.dot(u[α[2]], sR[2])
        )

    @cached_property
    def weno5_right_substencils(
        self,
    ) -> tuple[sp.sparse.csr_matrix, sp.sparse.csr_matrix, sp.sparse.csr_matrix]:

        # Mirror left substencils: (sR0←sL2, sR1←sL1, sR2←sL0).
        sL0, sL1, sL2 = self.weno5_left_substencils
        sR0 = mirror_symmetric(sL2)
        sR1 = mirror_symmetric(sL1)
        sR2 = mirror_symmetric(sL0)

        if self.verbose:
            np.set_printoptions(precision=2, linewidth=120)
            print(f"sR0 ({sR0.shape}):\n{sR0.toarray()}")
            print(f"sR1 ({sR1.shape}):\n{sR1.toarray()}")
            print(f"sR2 ({sR2.shape}):\n{sR2.toarray()}")

        return sR0, sR1, sR2

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

        α0 = d0 / (β0 * β0)
        α1 = d1 / (β1 * β1)
        α2 = d2 / (β2 * β2)

        inv_total = 1.0 / (α0 + α1 + α2)

        w0 = α0 * inv_total
        w1 = α1 * inv_total
        w2 = α2 * inv_total

        return [w0, w1, w2]

    def pointwise_eval_smoothness_indicators_js(
        self, u: np.ndarray, side: str
    ) -> tuple[float, float, float]:
        match side:
            case "left" | "L":
                d0, d1, d2 = 0.1, 0.6, 0.3
            case "right" | "R":
                d0, d1, d2 = 0.3, 0.6, 0.1
            case _:
                raise ValueError(f"Invalid side: {side!r}")

        α = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        I0, I1, I2 = fd_smooth_indicator_weights(self.r)
        beta0 = np.dot(u[α[0]], I0[0]) + np.dot(u[α[0]], I0[1])
        beta1 = np.dot(u[α[1]], I1[0]) + np.dot(u[α[1]], I1[1])
        beta2 = np.dot(u[α[2]], I2[0]) + np.dot(u[α[2]], I2[1])

        β0 = beta0**2 + beta1**2 + self.ϵ
        β1 = beta1**2 + beta2**2 + self.ϵ
        β2 = beta2**2 + beta0**2 + self.ϵ

        α0 = d0 / (β0 * β0)
        α1 = d1 / (β1 * β1)
        α2 = d2 / (β2 * β2)

        inv_total = 1.0 / (α0 + α1 + α2)

        w0 = α0 * inv_total
        w1 = α1 * inv_total
        w2 = α2 * inv_total

        return w0, w1, w2

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

        α0 = d0 / (β0 * β0)
        α1 = d1 / (β1 * β1)
        α2 = d2 / (β2 * β2)

        inv_total = 1 / (α0 + α1 + α2)

        w0 = α0 * inv_total
        w1 = α1 * inv_total
        w2 = α2 * inv_total

        match side:
            case "left" | "L":
                a0 = (2.0 * w0 + w1) / 3.0
                a1 = (w0 + 2.0 * w1 + 2.0 * w2) / 3.0
                a2 = w2 / 3.0
                b0 = self.inv_h * w0 / 6.0
                b1 = self.inv_h * (5.0 * w0 + 5.0 * w1 + w2) / 6.0
                b2 = self.inv_h * (w1 + 5.0 * w2) / 6.0
            case "right" | "R":
                a0 = w0 / 3.0
                a1 = (w2 + 2.0 * w1 + 2.0 * w0) / 3.0
                a2 = (2.0 * w2 + w1) / 3.0
                b0 = self.inv_h * (w1 + 5.0 * w0) / 6.0
                b1 = self.inv_h * (5.0 * w2 + 5.0 * w1 + w0) / 6.0
                b2 = self.inv_h * w2 / 6.0

        return [a0, a1, a2, b0, b1, b2]

    def Dx_upwind(self, u: np.ndarray) -> np.ndarray:
        # Interpolate the solution to the left and right interfaces
        if self.bc == BoundaryCondition.DIRICHLET:
            v = self.linearly_interpolate_boundaries @ u
        else:
            v = u

        if self.scheme == NonOscillatoryScheme.WENO5:
            # Compute the smoothness indicators
            w0L, w1L, w2L = self.smoothness_indicators_js(v, "L")

            # Compute the WENO reconstruction
            sL0, sL1, sL2 = self.weno5_left_substencils
            uL = w0L * (sL0 @ v) + w1L * (sL1 @ v) + w2L * (sL2 @ v)

        elif self.scheme == NonOscillatoryScheme.CRWENO5:
            # Compute the smoothness indicators
            a0, a1, a2, b0, b1, b2 = self.smoothness_indicators_gb(v, "L")

            # Build A_ij and B_ij matrices
            if self.bc == BoundaryCondition.PERIODIC:
                A = periodic_tridiag_oper(a0, a1, a2)
                rhs = b0 * np.roll(v, 1) + b1 * v + b2 * np.roll(v, -1)

            elif self.bc == BoundaryCondition.DIRICHLET:
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

            elif self.bc == BoundaryCondition.GHOST_POINTS:
                outer_L, inner_L, inner_R, outer_R = self.crweno5_boundary_weights
                ng = self.n_gps
                a0[0], a1[0], a2[0] = 0, 1, 0
                a0[1], a1[1], a2[1] = 0, 1, 0
                a0[-2], a1[-2], a2[-2] = 0, 1, 0
                a0[-1], a1[-1], a2[-1] = 0, 1, 0
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=ng)
                rhs = B @ v
                # MD : should I use a WENO5 reconstructions here?
                rhs[0] = np.dot(outer_L, v[ng : ng + 3])
                rhs[1] = np.dot(inner_L, v[ng : ng + 4])
                rhs[-2] = np.dot(inner_R, v[-ng - 4 : -ng])
                rhs[-1] = np.dot(outer_R, v[-ng - 3 : -ng])

            else:
                raise ValueError(f"Unsupported BoundaryCondition: {self.bc!r}")

            # Compute uL: u_j+1/2
            lu = sp.sparse.linalg.splu(A.tocsc())
            uL = lu.solve(rhs)

            if self.verbose:
                np.set_printoptions(precision=2, linewidth=120)
                print(f"A ({A.shape}):\n{A.toarray()}")
                print(f"B ({B.shape}):\n{B.toarray()}")

        # Compute the derivative for the left-biased stencil
        if self.bc == BoundaryCondition.PERIODIC:
            return uL - np.roll(uL, shift=1)
        elif self.bc == BoundaryCondition.DIRICHLET:
            return np.pad(uL[1:-1] - uL[:-2], (1, 1), mode="constant")
        else:  # self.bc == BoundaryCondition.GHOST_POINTS
            return np.pad(uL[1:-1] - uL[:-2], (self.r, self.r), mode="constant")

    def Dx_downwind(self, u: np.ndarray) -> np.ndarray:
        # Interpolate the solution to the left and right interfaces
        if self.bc == BoundaryCondition.DIRICHLET:
            v = self.linearly_interpolate_boundaries @ u
        else:
            v = u

        if self.scheme == NonOscillatoryScheme.WENO5:
            # Compute the smoothness indicators
            w0R, w1R, w2R = self.smoothness_indicators_js(v, "R")

            # Compute the WENO reconstruction
            sR0, sR1, sR2 = self.weno5_right_substencils
            uR = w0R * (sR0 @ v) + w1R * (sR1 @ v) + w2R * (sR2 @ v)

        elif self.scheme == NonOscillatoryScheme.CRWENO5:
            # Compute the smoothness indicators
            a0, a1, a2, b0, b1, b2 = self.smoothness_indicators_gb(v, "R")

            # Build A_ij and B_ij matrices
            if self.bc == BoundaryCondition.PERIODIC:
                A = periodic_tridiag_oper(a0, a1, a2)
                rhs = b0 * np.roll(v, 1) + b1 * v + b2 * np.roll(v, -1)

            elif self.bc == BoundaryCondition.DIRICHLET:
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

            elif self.bc == BoundaryCondition.GHOST_POINTS:
                outer_L, inner_L, inner_R, _ = self.crweno5_boundary_weights
                ng = self.n_gps
                a0[1], a1[1], a2[1] = 0, 1, 0
                a0[2], a1[2], a2[2] = 0, 1, 0
                a0[-2], a1[-2], a2[-2] = 0, 1, 0
                a0[-1], a1[-1], a2[-1] = 0, 1, 0
                A = rect_tridiag_oper(a0, a1, a2)
                B = rect_tridiag_oper(b0, b1, b2, n_ghost_points=ng)
                rhs = B @ v
                # MD : should I use a WENO5 reconstructions here?
                rhs[1] = np.dot(outer_L, v[ng : ng + 3])
                rhs[2] = np.dot(inner_L, v[ng : ng + 4])
                rhs[-2] = np.dot(inner_L, v[-ng - 4 : -ng])
                rhs[-1] = np.dot(inner_R, v[-ng - 4 : -ng])

            else:
                raise ValueError(f"Unsupported BoundaryCondition: {self.bc!r}")

            # Compute uR: u_j+1/2
            lu = sp.sparse.linalg.splu(A.tocsc())
            uR = lu.solve(rhs)

            if self.verbose:
                np.set_printoptions(precision=2, linewidth=120)
                print(f"A ({A.shape}):\n{A.toarray()}")
                print(f"B ({B.shape}):\n{B.toarray()}")

        # Compute the derivative for the right-biased stencil
        if self.bc == BoundaryCondition.PERIODIC:
            return np.roll(uR, shift=-1) - uR
        elif self.bc == BoundaryCondition.DIRICHLET:
            return np.pad(uR[2:] - uR[1:-1], (1, 1), mode="constant")
        else:  # self.bc == BoundaryCondition.GHOST_POINTS
            return np.pad(uR[2:] - uR[1:-1], (self.r, self.r), mode="constant")
