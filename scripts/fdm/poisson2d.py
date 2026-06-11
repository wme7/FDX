# %% 2D Gauss-Jacobi solver for the Poisson equation
#
# Solve the pressure Poisson equation on a rectangular domain:
#
#     ∂²p/∂x² + ∂²p/∂y² = b
#
# with homogeneous Dirichlet boundary conditions (state = 0 on all walls) and
# two opposing source spikes that relax in pseudo-time using Jacobi relaxation.
#
# Spatial discretisation uses the FDX Grid2d 2nd-order central Laplacian
# (grid.Laplacian_central) for residual monitoring.

from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

from fdx import finite_differences_grid as Ω
from fdx.finite_differences_grid import BoundaryCondition as dΩ

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "gauss_jacobi"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  Source terms
# ------------------------------------------------------------------ #


def exact_state(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def source_term1(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Manufactured solution  u_exact = sin(πx)sin(πy)  →  b = -2π²·u_exact."""
    return -2.0 * np.pi**2 * exact_state(X, Y)


def source_term2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Two opposing point spikes (Barba's "12 steps to NS", Lesson 10)."""
    b = np.zeros_like(X)
    b[int(0.25 * Y.shape[0]), int(0.25 * X.shape[1])] = 100.0
    b[int(0.75 * Y.shape[0]), int(0.75 * X.shape[1])] = -100.0
    return b


# ------------------------------------------------------------------ #
#  Matrix-free Laplacian operator
# ------------------------------------------------------------------ #


def apply_dirichlet(state: np.ndarray) -> None:
    """Homogeneous Dirichlet: state = 0 on all four boundaries."""
    state[0, :] = state[-1, :] = 0.0
    state[:, 0] = state[:, -1] = 0.0


def jacobi_step(state: np.ndarray, b: np.ndarray, dx2: float, dy2: float) -> np.ndarray:
    """One Jacobi relaxation sweep for ∇²p = b (5-point stencil, interior only)."""
    pd = state.copy()
    state[1:-1, 1:-1] = (
        (pd[1:-1, 2:] + pd[1:-1, :-2]) * dy2
        + (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx2
        - b[1:-1, 1:-1] * dx2 * dy2
    ) / (2.0 * (dx2 + dy2))
    apply_dirichlet(state)
    return state


# ------------------------------------------------------------------ #
#  Solver
# ------------------------------------------------------------------ #


def solve_poisson_jacobi(
    b: np.ndarray,
    *,
    dx: float,
    dy: float,
    n_iter: int,
    p0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Pseudo-time Jacobi iteration until n_iter sweeps are complete."""

    # Set initial condition
    state = np.zeros_like(b) if p0 is None else p0.copy()

    # Apply boundary conditions
    apply_dirichlet(state)

    # Auxiliary variables
    dx2, dy2 = dx * dx, dy * dy

    # Pseudo-time Jacobi iteration
    residuals: list[float] = []
    for _ in range(n_iter):
        state = jacobi_step(state, b, dx2, dy2)
        residuals.append(residual(state, b))
    return state, residuals


def residual(state: np.ndarray, b: np.ndarray) -> float:
    """RMS residual of ∇²p = b on interior nodes (FDX central Laplacian)."""
    lap_p = grid.Laplacian_central(state)
    diff = lap_p[1:-1, 1:-1] - b[1:-1, 1:-1]
    return float(np.linalg.norm(diff) / np.sqrt(diff.size))


# --------------------------- Parameters -------------------------- #

NT = 200  # number of Jacobi relaxation sweeps
MANUFACTURED = True  # use manufactured solution

# ----------------------------- setup ----------------------------- #

# fmt: off
grid = Ω.Grid2d(
    xa=0.0, xb=2.0, nx=50, bc_x=dΩ.DIRICHLET,
    ya=0.0, yb=1.0, ny=50, bc_y=dΩ.DIRICHLET,
    verbose=False,
)
# fmt: on

# Get grid parameters
x, y, Δx, Δy = grid.X, grid.Y, grid.x.h, grid.y.h

# Set source term
if MANUFACTURED:
    b = source_term1(x, y)  # manufactured solution
else:
    b = source_term2(x, y)  # point spikes

# ----------------------------- solve ----------------------------- #

state, residuals = solve_poisson_jacobi(b, dx=Δx, dy=Δy, n_iter=NT)

err = residual(state, b)
print(f"Interior residual  ||∇²p − b||₂/√N  ({NT} sweeps): {err:.3e}")

# Exact error for the manufactured case
if MANUFACTURED:
    err_inf = float(np.max(np.abs(state - exact_state(x, y))))
    print(f"Max pointwise error vs exact solution: {err_inf:.3e}")

# ------------------------ post-processing ------------------------ #

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    x,
    y,
    state,
    rstride=1,
    cstride=1,
    cmap=cm.viridis,
    linewidth=0,
    antialiased=False,
)
ax.view_init(30, 225)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$state$")
ax.set_title(f"Poisson pressure field  (Jacobi, {NT} sweeps)")
fig.colorbar(surf, shrink=0.55, aspect=12, label="$state$")
out_3d = FIGURES_DIR / "poisson_step10_surface.png"
fig.savefig(out_3d, dpi=140, bbox_inches="tight")
print(f"Saved: {out_3d}")

# -- 2D panels ---------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

# Auxiliary variables for plotting
vmax_b = float(np.max(np.abs(b)))
vmax_p = float(np.max(np.abs(state)))
domain = [grid.x.min, grid.x.max, grid.y.min, grid.y.max]
interp = "nearest"

# Panel 0 — source term
im0 = axes[0].imshow(
    b,
    origin="lower",
    extent=domain,
    cmap="RdBu_r",
    vmin=-vmax_b,
    vmax=vmax_b,
    interpolation=interp,
)
axes[0].set_title("Source  $b$")
axes[0].set_xlabel("$x$")
axes[0].set_ylabel("$y$")
fig.colorbar(im0, ax=axes[0], shrink=0.85)

# Panel 1 — pressure solution
im1 = axes[1].imshow(
    state,
    origin="lower",
    extent=domain,
    cmap="viridis",
    vmin=-vmax_p,
    vmax=vmax_p,
    interpolation=interp,
)
axes[1].set_title(f"Pressure  $state$  ({NT} Jacobi sweeps)")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")
fig.colorbar(im1, ax=axes[1], shrink=0.85)

# Panel 2 — residual history
axes[2].semilogy(range(1, len(residuals) + 1), residuals, "o-", ms=3)
axes[2].set_xlabel("Jacobi sweep")
axes[2].set_ylabel(r"interior $\|\nabla^2 state - b\|_2 / \sqrt{N}$")
axes[2].set_title("Pseudo-time residual")
axes[2].grid(True, which="both", alpha=0.4)

# Add title and save the plot
fig.suptitle("2D Poisson equation — Jacobi relaxation", fontsize=12)
out_panel = FIGURES_DIR / "poisson_jacobi_panel.png"
fig.savefig(out_panel, dpi=140, bbox_inches="tight")
print(f"Saved: {out_panel}")

# Show the plot if not in batch mode
if plt.get_backend().lower() != "agg":
    plt.show()
