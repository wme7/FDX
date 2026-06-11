# Visualization of 1D Burgers' equation using Matplotlib and FDX library.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if True:
    from fdx import finite_differences_grid as Ω
    from fdx.finite_differences_grid import BoundaryCondition as dΩ
else:
    from fdx import essentially_nonoscillatory_grid as Ω
    from fdx.essentially_nonoscillatory_grid import BoundaryCondition as dΩ

from fdx.drivers import Driver4Visualization
from fdx.time_integrators import RungeKutta as rk

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "invicid_burgers1d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def sine_wave(x):
    return np.sin(2 * np.pi * x)


def gaussian_pulse(x, center=0.5, width=0.1):
    return np.exp(-((x - center) ** 2) / (2 * width**2))


def square_wave(x, center=0.4, width=0.3):
    return np.where((x >= center - width / 2) & (x <= center + width / 2), 1.0, 0.1)


# --------------------------- Constants --------------------------- #

π = np.pi

# --------------------------- Parameters -------------------------- #

L = 1.0  # Length of the domain
NX = 200  # Number of grid points
NS = 10  # Number of snapshots
TS = "RK3"  # Time-stepping method: "Euler/RK1", "RK2", "RK3", or "RK4"
SCH = "GODUNOV"  # spatial scheme: "UPWIND", "DOWNWIND", "CENTRAL", etc.
CFL = 0.5  # CFL number for stability
tEnd = 1.0  # End time of the simulation

# ----------------------------- setup ----------------------------- #

# Use a periodic grid for x in [0, L] with NX points
grid = Ω.Grid1d(0, L, NX, bc=dΩ.PERIODIC, r_width=1, verbose=False)

# Get grid parameters
xmin, xmax, x, Δx = grid.min, grid.max, grid.nodes, grid.h

# Set initial condition, time and iteration count
u0, t0 = sine_wave(x), 0.0

# Compute time step based on CFL condition
Δt0 = CFL * Δx / np.abs(u0).max()

# Prepare interactive plot
fig, ax = plt.subplots()
(line1,) = ax.plot(x, u0, "-", label="Initial Condition")
(line2,) = ax.plot(x, u0, "-", label=f"{SCH.upper()} scheme")
ax.set_xlabel("$X$")
ax.set_ylabel("$u(t)$")
ax.grid()
ax.legend()


# Define BCs (if needed)
def apply_bc(u):
    match grid.bc:
        case dΩ.DIRICHLET:
            u[0], u[-1] = u0[0], u0[-1]
        case dΩ.GHOST_POINTS:
            u[: grid.n_gps], u[-grid.n_gps :] = u0[: grid.n_gps], u0[-grid.n_gps :]
    return u


# Define RHS function for the Burgers' equation
def burgers_rhs(u):
    match SCH:
        case "UPWIND":
            return -u * grid.Dx_upwind(u)
        case "DOWNWIND":
            return -u * grid.Dx_downwind(u)
        case "CENTRAL":
            return -u * grid.Dx_central(u)
        case "PADE":
            return -u * grid.Dx_pade(u)
        case "COMPACT":
            return -u * grid.Dx_compact(u)
        case "GODUNOV":
            return np.where(u > 0, -u * grid.Dx_upwind(u), -u * grid.Dx_downwind(u))
        case "FLUX_SPLITTING":
            f = 0.5 * u * u
            fP = 0.5 * (f + np.abs(u) * u)
            fN = 0.5 * (f - np.abs(u) * u)
            return -grid.Dx_upwind(fP) - grid.Dx_downwind(fN)
        case "RUSANOV":
            return -grid.Dx_rusanov(u, lambda u: 0.5 * u * u)
        case _:
            raise ValueError(f"Invalid scheme: {SCH}")


# ----------------------------- driver ----------------------------- #

snapshots = Driver4Visualization(
    rk_scheme=rk.from_name(TS),
    initial_state=u0,
    initial_time=t0,
    final_time=tEnd,
    initial_time_step=Δt0,
    number_of_snapshots=NS,
    rhs_fn=burgers_rhs,
    plot_object=line2,  # <-- visualize line2 as it evolves in time
)

# ------------------------- post-processing ------------------------- #

# Plot the solution
fig, ax = plt.subplots()
for time, state in snapshots:
    ax.plot(x, state, linewidth=1, label=f"t = {time:.3f}")
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$U$")
ax.set_xlim(xmin, xmax)
ax.grid(False)
ax.legend(loc="upper right")
fig.tight_layout()
out_snap = FIGURES_DIR / "invicid_burgers1d_snapshots.png"
out_snap.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_snap, dpi=140, bbox_inches="tight")
print(f"Saved snapshots: {out_snap}")
plt.show()
