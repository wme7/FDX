# Visualization of 2D Acoustic Wave using Matplotlib and FDX project

# Solve the linearised Euler / acoustics system on a periodic [0, 1]²:
#     ∂_t p + c₀² (∂_x u + ∂_y v) = 0
#     ∂_t u + ∂_x p = 0
#     ∂_t v + ∂_y p = 0
#
# with a Gaussian pressure pulse as the initial condition::
#     p₀(x, y) = exp(−r² / σ²),    r² = (x − 1/2)² + (y − 1/2)²,
#     u₀ = v₀ = 0.
#
# Continuous total energy
#     E = (1/2) ∫∫ (p² / c₀² + u² + v²) dx dy
#
# is exactly conserved.

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from fdx import finite_differences_grid as Ω
from fdx.drivers import Driver4Visualization
from fdx.finite_differences_grid import BoundaryCondition as dΩ
from fdx.time_integrators import RungeKutta as rk

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "acoustic_wave2d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def Gaussian_pulse(X, Y, x0=0.5, y0=0.5, sigma=0.06):
    p0 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / sigma**2)
    u0 = np.zeros_like(p0)
    v0 = np.zeros_like(p0)
    return np.stack([p0, u0, v0], axis=0)


def total_energy(state, c0, hx, hy):
    p, u, v = state[0], state[1], state[2]
    return 0.5 * float(np.sum(p**2 / c0**2 + u**2 + v**2)) * hx * hy


# --------------------------- Constants --------------------------- #

c0 = 1.0  # Speed of sound
σ = 0.06  # Standard deviation of the Gaussian pulse

# --------------------------- Parameters -------------------------- #

L = 1.0  # Length of the domain
NX = 256  # Number of grid points
NY = 256  # Number of grid points
NS = 5  # Number of snapshots
TS = "RK4"  # Time-stepping method: "Euler/RK1", "RK2", "RK3", or "RK4"
SCH = "COMPACT"  # spatial scheme: "UPWIND", "DOWNWIND", "CENTRAL", etc.
CFL = 0.8  # CFL number for stability
tEnd = 0.4  # End time of the simulation

# ----------------------------- setup ----------------------------- #

# Use a periodic grid fo x in [0, 1] with 100 points
# fmt: off
grid = Ω.Grid2d(
    xa=0, xb=L, nx=NX, bc_x=dΩ.PERIODIC, r_width_x=1,
    ya=0, yb=L, ny=NY, bc_y=dΩ.PERIODIC, r_width_y=1,
    verbose=False,
)
# fmt: on

# Get grid parameters
x, y, Δx, Δy = grid.X, grid.Y, grid.x.h, grid.y.h

# Set initial condition
q0, t0 = Gaussian_pulse(x, y, 0.5, 0.5, σ), 0.0

# Compute time step based on CFL condition
Δt0 = CFL * min(Δx, Δy) / c0

# Create interactive 2D image
fig, ax = plt.subplots()
image = ax.imshow(
    q0[0],
    origin="lower",
    extent=[0, L, 0, L],
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0,
    interpolation="nearest",
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
fig.colorbar(image, ax=ax, shrink=0.8, label="p")


# Define sample function
def sample_fn(state):
    return total_energy(state, c0, Δx, Δy)


# Define RHS for acoustics equations
def acoustics_rhs(state) -> np.ndarray:
    p, u, v = state[0], state[1], state[2]
    rhs_0 = -(c0 * c0) * grid.Div_central([u, v])
    rhs_1 = -grid.Dx_central(p)
    rhs_2 = -grid.Dy_central(p)
    return np.stack([rhs_0, rhs_1, rhs_2], axis=0)


# ----------------------------- driver ----------------------------- #

[t_sample, q_sample], snapshots = Driver4Visualization(
    rk_scheme=rk.from_name(TS),
    initial_state=q0,
    initial_time=t0,
    final_time=tEnd,
    initial_time_step=Δt0,
    sample_frequency=2,
    sample_fn=sample_fn,
    number_of_snapshots=NS,
    rhs_fn=acoustics_rhs,
    plot_object=image,  # <-- visualize image as it evolves in time
)

# ------------------------- post-processing ------------------------- #

# Plot snapshots panel
n = len(snapshots)
fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2), sharex=True, sharey=True)
for col, (time, q) in enumerate(snapshots):
    im = axes[col].imshow(
        q[0],
        origin="lower",
        extent=[0, L, 0, L],
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[col].set_title(f"t={time:.2f}")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="p")
fig.suptitle(f"Acoustic Wave 2D  n={NX}x{NY}, c0={c0}, σ={σ}", fontsize=11)
out_snap = FIGURES_DIR / "acoustic_wave2d_snapshots.png"
fig.savefig(out_snap, dpi=140, bbox_inches="tight")
print(f"Saved: {out_snap}")

# Plot energy time series
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_sample, q_sample, label="Energy")
ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title("Energy Time Series")
ax.legend()
ax.grid(True)
out_energy = FIGURES_DIR / "acoustic_wave2d_energy.png"
fig.savefig(out_energy, dpi=140, bbox_inches="tight")
print(f"Saved: {out_energy}")
