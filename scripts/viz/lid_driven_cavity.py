"""
Solve 2D lid-driven cavity flow (Lorena Barba Step 12 style).

Problem setup:
  - No-slip walls on all sides
  - Moving lid at top (top wall)
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

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "lid_driven_cavity2d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------- Parameters --------------------------- #

# Physical parameters
RHO = 1.0
NU = 0.01
U_LID = 1.0

# Numerical parameters
LX, LY = 2.0, 2.0
NX, NY = 200, 200
NT = 5_000
NIT = 50  # pressure Poisson iterations per step
DT = 0.002

# Spatial schemes for momentum equation
CONVECTION_SCHEME = "CENTRAL"  # "CENTRAL" or "UPWIND"
TIME_SCHEME = "RK1"  # requested RK1 / Euler


# ----------------------------- Setup ------------------------------ #

# fmt: off
g = Ω.Grid2d(
    xa=0.0, xb=LX, nx=NX, bc_x=dΩ.DIRICHLET, r_width_x=1,
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
    """Apply no-slip walls and moving lid (top wall)."""
    # Side walls
    u_field[:, 0] = 0.0
    u_field[:, -1] = 0.0
    v_field[:, 0] = 0.0
    v_field[:, -1] = 0.0

    # Bottom wall
    u_field[0, :] = 0.0
    v_field[0, :] = 0.0

    # Top moving lid
    u_field[-1, :] = U_LID
    v_field[-1, :] = 0.0


def apply_pressure_bc(p_field: np.ndarray) -> None:
    """Barba-style pressure BCs for cavity flow."""
    # dp/dx = 0 on left/right walls
    p_field[:, 0] = p_field[:, 1]
    p_field[:, -1] = p_field[:, -2]

    # dp/dy = 0 at bottom wall
    p_field[0, :] = p_field[1, :]

    # p = 0 at top wall (pressure reference)
    p_field[-1, :] = 0.0


def build_pressure_rhs(u_field: np.ndarray, v_field: np.ndarray) -> np.ndarray:
    """RHS of pressure Poisson equation."""
    ux = g.Dx_central(u_field)
    uy = g.Dy_central(u_field)
    vx = g.Dx_central(v_field)
    vy = g.Dy_central(v_field)
    return RHO * ((ux + vy) / DT - ux**2 - 2.0 * uy * vx - vy**2)


def solve_pressure_poisson_explicit(
    p_field: np.ndarray, b_field: np.ndarray, nit: int
) -> np.ndarray:
    """Explicit Jacobi-style pressure-Poisson iterations."""
    p_new = p_field.copy()
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 * (dx2 + dy2)

    for _ in range(nit):
        pn = p_new.copy()
        p_new[1:-1, 1:-1] = (
            (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2
            + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2
            - b_field[1:-1, 1:-1] * dx2 * dy2
        ) / denom
        apply_pressure_bc(p_new)

    return p_new


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

    rhs_u = -conv_u - (1.0 / RHO) * g.Dx_central(p) + NU * g.Laplacian_central(u)
    rhs_v = -conv_v - (1.0 / RHO) * g.Dy_central(p) + NU * g.Laplacian_central(v)
    return np.stack([rhs_u, rhs_v], axis=0)


# ----------------------------- Solve ------------------------------ #

kinetic_energy = np.zeros(NT + 1)
divergence_l2 = np.zeros(NT + 1)

apply_velocity_bc(u, v)
kinetic_energy[0] = 0.5 * float(np.mean(u**2 + v**2))
divergence_l2[0] = float(np.sqrt(np.mean(g.Div_central([u, v]) ** 2)))

for n in range(1, NT + 1):
    b = build_pressure_rhs(u, v)
    p = solve_pressure_poisson_explicit(p, b, NIT)

    state = np.stack([u, v], axis=0)
    state = rk1.step(momentum_rhs, state, DT, p)
    u, v = state[0], state[1]

    apply_velocity_bc(u, v)

    kinetic_energy[n] = 0.5 * float(np.mean(u**2 + v**2))
    divergence_l2[n] = float(np.sqrt(np.mean(g.Div_central([u, v]) ** 2)))

    if n % max(1, NT // 10) == 0:
        print(
            f"step={n:4d}/{NT}, KE={kinetic_energy[n]:.4e}, "
            f"divL2={divergence_l2[n]:.4e}"
        )


# ------------------------ Post-processing ------------------------- #

# Show vector field
viz.show_vector_field(
    field=np.stack([u, v], axis=-1),
    width=500,
    height=500,
    tracers_count=3000,
    speed=0.6,
    point_size=3,
    decay=0.001,
)
