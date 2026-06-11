"""
Solve 2D incompressible channel flow (Lorena Barba Step 12 style).

Problem setup:
  - Periodic in x
  - No-slip walls in y
  - Pressure-Poisson solved with explicit Jacobi iterations
  - Momentum advanced with RK1 (Forward Euler)

FDX operators are used for spatial derivatives:
  - Convection: CENTRAL or UPWIND
  - Diffusion:  CENTRAL
"""

from pathlib import Path

import numpy as np

from fdx import finite_differences_grid as Ω
from fdx import viz
from fdx.finite_differences_grid import BoundaryCondition as dΩ
from fdx.time_integrators import RungeKutta as rk

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "channel_flow2d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------- Parameters --------------------------- #

# Physical parameters
RHO = 1.0
NU = 0.1
F = 1.0  # constant streamwise forcing

# Numerical parameters
LX, LY = 2.0, 2.0
NX, NY = 40, 40
NIT = 50
DT = 0.005
MAX_STEPS = 5_000
STEADY_TOL = 1e-3

# Spatial/time schemes
CONVECTION_SCHEME = "CENTRAL"  # "CENTRAL" or "UPWIND"
TIME_SCHEME = "RK1"


# ----------------------------- Setup ------------------------------ #

# fmt: off
g = Ω.Grid2d(
    xa=0.0, xb=LX, nx=NX, bc_x=dΩ.PERIODIC, r_width_x=1,
    ya=0.0, yb=LY, ny=NY, bc_y=dΩ.DIRICHLET, r_width_y=1,
    verbose=False,
)
# fmt: on

X, Y, dx, dy = g.X, g.Y, g.x.h, g.y.h

u = np.zeros_like(X)
v = np.zeros_like(X)
p = np.zeros_like(X)

rk1 = rk.from_name(TIME_SCHEME)


def apply_velocity_bc(u_field: np.ndarray, v_field: np.ndarray) -> None:
    """No-slip at top and bottom walls."""
    u_field[0, :] = 0.0
    u_field[-1, :] = 0.0
    v_field[0, :] = 0.0
    v_field[-1, :] = 0.0


def build_pressure_rhs(u_field: np.ndarray, v_field: np.ndarray) -> np.ndarray:
    """Build RHS b of Poisson equation: ∇²p = b."""
    ux = g.Dx_central(u_field)
    uy = g.Dy_central(u_field)
    vx = g.Dx_central(v_field)
    vy = g.Dy_central(v_field)
    return RHO * ((ux + vy) / DT - ux**2 - 2.0 * uy * vx - vy**2)


def solve_pressure_poisson_explicit(
    p_field: np.ndarray, b_field: np.ndarray, nit: int
) -> np.ndarray:
    """Explicit Jacobi iterations with periodic-x treatment."""
    pn = np.empty_like(p_field)
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    for _ in range(nit):
        pn = p_field.copy()

        # interior (excluding periodic-x edge columns for now)
        p_field[1:-1, 1:-1] = (
            (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2
            + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2
            - b_field[1:-1, 1:-1] * dx2 * dy2
        ) / denom

        # periodic-x: right boundary (j=-1 uses j=0 and j=-2)
        p_field[1:-1, -1] = (
            (pn[1:-1, 0] + pn[1:-1, -2]) * dy2
            + (pn[2:, -1] + pn[:-2, -1]) * dx2
            - b_field[1:-1, -1] * dx2 * dy2
        ) / denom

        # periodic-x: left boundary (j=0 uses j=1 and j=-1)
        p_field[1:-1, 0] = (
            (pn[1:-1, 1] + pn[1:-1, -1]) * dy2
            + (pn[2:, 0] + pn[:-2, 0]) * dx2
            - b_field[1:-1, 0] * dx2 * dy2
        ) / denom

        # Neumann boundary conditions in y
        p_field[-1, :] = p_field[-2, :]  # bottom wall
        p_field[0, :] = p_field[1, :]  # top wall

    return p_field


def momentum_rhs(state: np.ndarray, p: np.ndarray) -> np.ndarray:
    """RHS for incompressible momentum equations."""
    u, v = state[0], state[1]

    scheme = CONVECTION_SCHEME.upper().strip()
    if scheme == "CENTRAL":
        ux = g.Dx_central(u)
        uy = g.Dy_central(u)
        vx = g.Dx_central(v)
        vy = g.Dy_central(v)
    elif scheme == "UPWIND":
        ux = np.where(u >= 0.0, g.Dx_upwind(u), g.Dx_downwind(u))
        uy = np.where(v >= 0.0, g.Dy_upwind(u), g.Dy_downwind(u))
        vx = np.where(u >= 0.0, g.Dx_upwind(v), g.Dx_downwind(v))
        vy = np.where(v >= 0.0, g.Dy_upwind(v), g.Dy_downwind(v))
    else:
        raise ValueError(
            f"Unknown CONVECTION_SCHEME={CONVECTION_SCHEME!r}. Use CENTRAL or UPWIND."
        )
    # Compute (u·∇u, u·∇v)
    conv_u = u * ux + v * uy
    conv_v = u * vx + v * vy

    rhs_u = -conv_u - (1.0 / RHO) * g.Dx_central(p) + NU * g.Laplacian_central(u) + F
    rhs_v = -conv_v - (1.0 / RHO) * g.Dy_central(p) + NU * g.Laplacian_central(v)
    return np.stack([rhs_u, rhs_v], axis=0)


# ----------------------------- Solve ------------------------------ #

kinetic_energy_hist = [0.5 * float(np.mean(u**2 + v**2))]
divergence_hist = [float(np.sqrt(np.mean(g.Div_central([u, v]) ** 2)))]
udiff_hist = [1.0]

step = 0
while step < MAX_STEPS:
    step += 1
    un = u.copy()

    b = build_pressure_rhs(u, v)
    p = solve_pressure_poisson_explicit(p, b, NIT)

    state = np.stack([u, v], axis=0)
    state = rk1.step(momentum_rhs, state, DT, p)
    u, v = state[0], state[1]

    apply_velocity_bc(u, v)

    sum_u = float(np.sum(np.abs(u)))
    if sum_u > 0:
        udiff = float(np.sum(np.abs(u - un)) / sum_u)
    else:
        udiff = float(np.sum(np.abs(u - un)))

    kinetic_energy_hist.append(0.5 * float(np.mean(u**2 + v**2)))
    divergence_hist.append(float(np.sqrt(np.mean(g.Div_central([u, v]) ** 2))))
    udiff_hist.append(udiff)

    if step % max(1, MAX_STEPS // 10) == 0:
        print(
            f"step={step:5d}/{MAX_STEPS}, udiff={udiff:.3e}, "
            f"KE={kinetic_energy_hist[-1]:.3e}, divL2={divergence_hist[-1]:.3e}"
        )

    if udiff < STEADY_TOL:
        print(f"Steady criterion met at step={step}, udiff={udiff:.3e}")
        break

if step == MAX_STEPS:
    print(f"Reached MAX_STEPS={MAX_STEPS} with udiff={udiff_hist[-1]:.3e}")


# ------------------------ Post-processing ------------------------- #

# Show vector field
viz.show_vector_field(
    field=np.stack([u, v], axis=-1),
    width=500,
    height=500,
    tracers_count=2000,
    speed=0.5,
    point_size=3,
    decay=0.001,
)
