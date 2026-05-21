"""
Weighted Essentially Non-Oscillatory Grid Module.

This module provides a class for building finite difference grids and operators.
"""

from enum import Enum, auto
from functools import cached_property

import numpy as np
import scipy as sp

from .eno_weights import fd_eno_weights, fd_smooth_indicator_weights
from .utils import build_lerp_boundaries_matrix as lerp_oper
from .utils import build_mirror_symmetric_operator as mirror_symmetric
from .utils import build_periodic_banded_matrix as periodic_oper
from .utils import build_rectangular_banded_matrix as rect_oper


# ------------------------------------------------------------------ #
#  ENO/WENO Operators Parameters                                     #
# ------------------------------------------------------------------ #
class BoundaryCondition(Enum):
    PERIODIC = auto()
    DIRICHLET = auto()
    GHOST_POINTS = auto()


class NonOscillatoryScheme(Enum):
    WENO5 = auto()
    # CRWENO5 = auto()


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

        self.x = np.linspace(
            start=a_grid, stop=b_grid, num=n_grid, endpoint=endpoint, dtype=float
        )
        self.h = h  # grid spacing

        # offsets and weights
        inv_h = 1 / self.h
        α = [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]
        ωL = fd_eno_weights(r)[:3] * inv_h
        I0, I1, I2 = fd_smooth_indicator_weights(r)

        # Build banded operatora depending on the boundary condition
        match self.bc:
            case BoundaryCondition.PERIODIC:
                # build the banded matrices for the smoothness indicators
                self.β00 = periodic_oper(n_grid, α[0], I0[0])
                self.β10 = periodic_oper(n_grid, α[1], I1[0])
                self.β20 = periodic_oper(n_grid, α[2], I2[0])
                self.β01 = periodic_oper(n_grid, α[0], I0[1])
                self.β11 = periodic_oper(n_grid, α[1], I1[1])
                self.β21 = periodic_oper(n_grid, α[2], I2[1])

                # build the banded matrices for the WENO reconstruction
                self.sL0 = periodic_oper(n_grid, α[0], ωL[0])
                self.sL1 = periodic_oper(n_grid, α[1], ωL[1])
                self.sL2 = periodic_oper(n_grid, α[2], ωL[2])
                self.sR2 = mirror_symmetric(self.sL0)
                self.sR1 = mirror_symmetric(self.sL1)
                self.sR0 = mirror_symmetric(self.sL2)

            case BoundaryCondition.DIRICHLET:
                # build the banded matrices for the smoothness indicators
                self.β00 = rect_oper(n_grid, α[0], I0[0], n_gps)
                self.β10 = rect_oper(n_grid, α[1], I1[0], n_gps)
                self.β20 = rect_oper(n_grid, α[2], I2[0], n_gps)
                self.β01 = rect_oper(n_grid, α[0], I0[1], n_gps)
                self.β11 = rect_oper(n_grid, α[1], I1[1], n_gps)
                self.β21 = rect_oper(n_grid, α[2], I2[1], n_gps)

                # build the banded matrices for the WENO reconstruction
                self.sL0 = rect_oper(n_grid, α[0], ωL[0], n_gps, r_reset=1)
                self.sL1 = rect_oper(n_grid, α[1], ωL[1], n_gps, r_reset=1)
                self.sL2 = rect_oper(n_grid, α[2], ωL[2], n_gps, r_reset=1)
                self.sR2 = mirror_symmetric(self.sL0)
                self.sR1 = mirror_symmetric(self.sL1)
                self.sR0 = mirror_symmetric(self.sL2)

            case BoundaryCondition.GHOST_POINTS:
                n_interior = n_grid - 2 * n_gps
                # build the banded matrices for the smoothness indicators
                self.β00 = rect_oper(n_interior, α[0], I0[0], n_gps)
                self.β10 = rect_oper(n_interior, α[1], I1[0], n_gps)
                self.β20 = rect_oper(n_interior, α[2], I2[0], n_gps)
                self.β01 = rect_oper(n_interior, α[0], I0[1], n_gps)
                self.β11 = rect_oper(n_interior, α[1], I1[1], n_gps)
                self.β21 = rect_oper(n_interior, α[2], I2[1], n_gps)

                # build the banded matrices for the WENO reconstruction
                self.sL0 = rect_oper(n_interior, α[0], ωL[0], n_gps, r_reset=1)
                self.sL1 = rect_oper(n_interior, α[1], ωL[1], n_gps, r_reset=1)
                self.sL2 = rect_oper(n_interior, α[2], ωL[2], n_gps, r_reset=1)
                self.sR2 = mirror_symmetric(self.sL0)
                self.sR1 = mirror_symmetric(self.sL1)
                self.sR0 = mirror_symmetric(self.sL2)

        if verbose:
            np.set_printoptions(precision=2, linewidth=120)
            for name in [
                "sL0",
                "sL1",
                "sL2",
            ]:
                print(f"{name}:\n{getattr(self, name).toarray()}")

        # Constants
        self.ϵ = 1e-6

    @cached_property
    def linearly_interpolate_boundaries(self) -> sp.sparse.csr_matrix:
        return lerp_oper(self.n, self.r - 1)

    def smoothness_indicators_js(self, u: np.ndarray, side: str) -> list[np.ndarray]:
        match side:
            case "left" | "L":
                d0, d1, d2 = 0.1, 0.6, 0.3
            case "right" | "R":
                d0, d1, d2 = 0.3, 0.6, 0.1
            case _:
                raise ValueError(f"Invalid side: {side!r}")

        b00 = self.β00 @ u
        b10 = self.β10 @ u
        b20 = self.β20 @ u
        b01 = self.β01 @ u
        b11 = self.β11 @ u
        b21 = self.β21 @ u

        β0 = b00 * b00 + b01 * b01 + self.ϵ
        β1 = b10 * b10 + b11 * b11 + self.ϵ
        β2 = b20 * b20 + b21 * b21 + self.ϵ

        α0 = d0 / (β0 * β0)
        α1 = d1 / (β1 * β1)
        α2 = d2 / (β2 * β2)

        inv_total = 1 / (α0 + α1 + α2)

        w0L = α0 * inv_total
        w1L = α1 * inv_total
        w2L = α2 * inv_total

        return [w0L, w1L, w2L]

    def Dx_upwind(self, u: np.ndarray) -> np.ndarray:
        # Interpolate the solution to the left and right interfaces
        if self.bc == BoundaryCondition.DIRICHLET:
            v = self.linearly_interpolate_boundaries @ u
        else:
            v = u

        # Compute the smoothness indicators
        w0L, w1L, w2L = self.smoothness_indicators_js(v, "L")

        # Compute the WENO reconstruction
        uL = w0L * (self.sL0 @ v) + w1L * (self.sL1 @ v) + w2L * (self.sL2 @ v)

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

        # Compute the smoothness indicators
        w0R, w1R, w2R = self.smoothness_indicators_js(v, "R")

        # Compute the WENO reconstruction
        uR = w0R * (self.sR0 @ v) + w1R * (self.sR1 @ v) + w2R * (self.sR2 @ v)

        # Compute the derivative for the right-biased stencil
        if self.bc == BoundaryCondition.PERIODIC:
            return np.roll(uR, shift=-1) - uR
        elif self.bc == BoundaryCondition.DIRICHLET:
            return np.pad(uR[2:] - uR[1:-1], (1, 1), mode="constant")
        else:  # self.bc == BoundaryCondition.GHOST_POINTS
            return np.pad(uR[2:] - uR[1:-1], (self.r, self.r), mode="constant")
