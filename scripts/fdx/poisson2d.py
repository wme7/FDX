# %% 2D matrix-free MINRES solver for the Poisson equation
#
# Solve the pressure Poisson equation on a rectangular domain:
#
#     ∂²p/∂x² + ∂²p/∂y² = b
#
# with homogeneous Dirichlet BCs (state = 0 on all walls).
#
# FDX's grid.Laplacian_central is wrapped in a SciPy LinearOperator
# boundary rows/columns are always kept at zero, which implicitly enforces
# the homogeneous Dirichlet boundary condition.
#
# MINRES is chosen because ∇²_h (with Dirichlet BCs) is symmetric negative
# definite, which is exactly the class of systems MINRES handles optimally.

from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.sparse.linalg import LinearOperator, minres

from fdx import finite_differences_grid as Ω
from fdx.finite_differences_grid import BoundaryCondition as dΩ

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "minres_poisson"
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
#  Laplacian operator
# ------------------------------------------------------------------ #


def build_laplacian_operator(grid: Ω.Grid2d) -> LinearOperator:
    """Return a LinearOperator that applies the interior 5-point Laplacian.

    DoF ordering: row-major interior nodes, size (ny-2)*(nx-2).
    The operator embeds the interior vector into the full (ny×nx) grid with
    zero padding on the boundary (homogeneous Dirichlet), applies
    grid.Laplacian_central, and returns the interior slice — no matrix stored.
    """
    ny, nx = grid.x.n, grid.y.n
    n_int = (ny - 2) * (nx - 2)

    def matvec(v: np.ndarray) -> np.ndarray:
        p_full = np.zeros((ny, nx))
        p_full[1:-1, 1:-1] = v.reshape(ny - 2, nx - 2)
        lap = grid.Laplacian_central(p_full)
        return lap[1:-1, 1:-1].ravel()

    return LinearOperator((n_int, n_int), matvec=matvec, dtype=float)


# ------------------------------------------------------------------ #
#  Solver
# ------------------------------------------------------------------ #


def solve_poisson_minres(
    b: np.ndarray,
    lo: LinearOperator,
    *,
    tol: float = 1e-10,
    maxiter: int = 10_000,
) -> tuple[np.ndarray, list[float]]:
    """Solve ∇²p = b with MINRES via a matrix-free LinearOperator.

    Parameters
    ----------
    b:
        2-D RHS array (full grid, same shape as grid.X).
    lo:
        Laplacian operator.
    tol:
        Relative tolerance passed to minres.
    maxiter:
        Maximum MINRES iterations.

    Returns
    -------
    state:
        2-D pressure field (zeros on boundary, solution on interior).
    residuals:
        RMS residual  ‖L·p − b‖₂/√N  at every MINRES iteration.
    """
    ny, nx = b.shape
    b_int = b[1:-1, 1:-1].ravel()

    residuals: list[float] = []

    def callback(xk: np.ndarray) -> None:
        r = b_int - lo @ xk
        residuals.append(float(np.linalg.norm(r) / np.sqrt(r.size)))

    x_int, info = minres(lo, b_int, rtol=tol, maxiter=maxiter, callback=callback)

    if info != 0:
        print(f"  minres did not fully converge (info={info})")

    final_rms = residuals[-1] if residuals else float("nan")
    n_iter = len(residuals)
    print(
        f"MINRES converged in {n_iter} iterations, final RMS residual = {final_rms:.3e}"
    )

    state = np.zeros((ny, nx))
    state[1:-1, 1:-1] = x_int.reshape(ny - 2, nx - 2)
    return state, residuals


# --------------------------- Parameters -------------------------- #

MANUFACTURED = True  # True → manufactured solution, False → point spikes

# ----------------------------- setup ----------------------------- #

# fmt: off
grid = Ω.Grid2d(
    xa=0.0, xb=2.0, nx=50, bc_x=dΩ.DIRICHLET,
    ya=0.0, yb=1.0, ny=50, bc_y=dΩ.DIRICHLET,
    verbose=False,
)
# fmt: on

# Get grid parameters
x, y = grid.X, grid.Y

# Set source term
if MANUFACTURED:
    b = source_term1(x, y)  # manufactured solution
else:
    b = source_term2(x, y)  # point spikes

# Build the Laplacian operator
L = build_laplacian_operator(grid)

# ----------------------------- solve ----------------------------- #

# Solve the Poisson equation
state, residuals = solve_poisson_minres(b, L)

# number of iterations
n_iter = len(residuals)

# Exact error for the manufactured case
if MANUFACTURED:
    err_inf = float(np.max(np.abs(state - exact_state(x, y))))
    print(f"Max pointwise error vs exact solution: {err_inf:.3e}")

# ------------------------ post-processing ------------------------ #

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(
    x, y, state, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False
)
ax.view_init(30, 225)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$state$")
ax.set_title(f"Poisson pressure field  (MINRES, {n_iter} iterations)")
fig.colorbar(surf, shrink=0.55, aspect=12, label="$state$")
out_3d = FIGURES_DIR / "poisson_minres_surface.png"
fig.savefig(out_3d, dpi=140, bbox_inches="tight")
print(f"Saved: {out_3d}")

# -- 2D panels -------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

# Auxiliary variables for plotting
vmax_b = float(np.max(np.abs(b))) or 1.0
vmax_p = float(np.max(np.abs(state))) or 1.0
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
axes[1].set_title(f"Pressure  $state$  ({n_iter} MINRES iters)")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$y$")
fig.colorbar(im1, ax=axes[1], shrink=0.85)

# Panel 2 — residual history
axes[2].semilogy(range(1, len(residuals) + 1), residuals, lw=1.2)
axes[2].set_xlabel("MINRES iteration")
axes[2].set_ylabel(r"$\|L\,state - b\|_2\;/\;\sqrt{N}$")
axes[2].set_title("Residual history")
axes[2].grid(True, which="both", alpha=0.4)

# Add title and save the plot
fig.suptitle("2D Poisson equation — matrix-free MINRES", fontsize=12)
out_panel = FIGURES_DIR / "poisson_minres_panel.png"
fig.savefig(out_panel, dpi=140, bbox_inches="tight")
print(f"Saved: {out_panel}")

if plt.get_backend().lower() != "agg":
    plt.show()
