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
# is exactly conserved.

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from fdx import finite_differences_grid as Ω
from fdx import time_integrators as τ

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "acoustic_wave2d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Constants
c0 = 1.0  # Speed of sound
σ = 0.06  # Standard deviation of the Gaussian pulse


def Gaussian_pulse(X, Y, x0=0.5, y0=0.5, sigma=0.06):
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / sigma**2)


def total_energy(state, c0, hx, hy):
    p, u, v = state[0], state[1], state[2]
    return 0.5 * float(np.sum(p**2 / c0**2 + u**2 + v**2)) * hx * hy


# Simulation parameters
L = 1.0  # Length of the domain
NX = 256  # Number of grid points
NY = 256  # Number of grid points
NS = 5  # Number of snapshots
RK = "RK3"  # Time-stepping method: "Euler/RK1", "RK2", "RK3", or "RK4"
CFL = 0.9  # CFL number for stability
Tend = 0.4  # End time of the simulation

# ----------------------------- setup ----------------------------- #

# Use a periodic grid fo x in [0, 1] with 100 points
grid = Ω.Grid2d(
    xa=0,
    xb=L,
    nx=NX,
    rx=2,
    ya=0,
    yb=L,
    ny=NY,
    ry=2,
    bcx=Ω.BoundaryCondition.PERIODIC,
    bcy=Ω.BoundaryCondition.PERIODIC,
    scheme=Ω.FiniteDifferenceScheme.CENTRAL,
    verbose=False,
)
x, y = np.meshgrid(grid.x, grid.y, indexing="xy")
Δx, Δy = grid.hx, grid.hy


# RHS for acoustics equations
def acoustics_rhs(state) -> np.ndarray:
    """state = (3, n, n) array: (p, u, v).  Returns dstate/dt."""
    p, u, v = state[0], state[1], state[2]
    div_u = grid.Derivative(u, "x") + grid.Derivative(v, "y")
    dp_dx = grid.Derivative(p, "x")
    dp_dy = grid.Derivative(p, "y")
    return np.stack([-(c0**2) * div_u, -dp_dx, -dp_dy], axis=0)


# Set initial condition
t, it = 0.0, 0
p0 = Gaussian_pulse(x, y, 0.5, 0.5, σ)
u0 = np.zeros_like(p0)
v0 = np.zeros_like(p0)
state = np.stack([p0, u0, v0], axis=0)

# Compute time step based on CFL condition
Δt = CFL * min(Δx, Δy) / c0

# Initialize space-time arrays
nt = 2 + int(Tend / Δt)  # 2: initial condition and final condition
snap_times = np.linspace(0, Tend, NS)
snaps = [(0.0, state[0].copy())]
next_snap = 1
state_energies = np.zeros(nt)
state_energies[it] = total_energy(state, c0, Δx, Δy)

# Enable interactive mode
plt.ion()

# Create interactive 2D image plot (fixed color limits for stable animation)
p_lim = 0.7 * float(np.max(np.abs(p0)))
fig, ax = plt.subplots()
im = ax.imshow(
    p0,
    origin="lower",
    extent=[0, L, 0, L],
    cmap="RdBu_r",
    vmin=-p_lim,
    vmax=p_lim,
    interpolation="nearest",
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Acoustic Wave 2D")
fig.colorbar(im, ax=ax, shrink=0.8, label="p")

# ----------------------------- driver ----------------------------- #

rk_scheme = τ.RungeKutta.from_name(RK)
tStart = time.process_time()
while t < Tend:
    # Update time & iteration count
    t += Δt
    it += 1

    # Update the solution
    state = rk_scheme.step(acoustics_rhs, state, Δt)

    # Capture snapshot at evenly spaced times
    if next_snap < len(snap_times) and t >= snap_times[next_snap] - 0.5 * Δt:
        snaps.append((t, state[0].copy()))
        next_snap += 1

    # Capture total energy
    state_energies[it] = total_energy(state, c0, Δx, Δy)

    # Update the plot
    if it % 10 == 0:
        im.set_data(state[0])
        fig.canvas.draw_idle()
        plt.pause(0.01)

    # Adjust time step to ensure we don't exceed the end time
    if t + Δt > Tend:
        Δt = Tend - t

# Compute the elapsed time
elapsed = time.process_time() - tStart
print(f"CPU time: {elapsed:.4f} s")

# ------------------------- post-processing ------------------------- #

# Keep the plot open
plt.ioff()  # Disable interactive mode

# Plot snapshots panel
n_snap = len(snaps)
fig, axes = plt.subplots(
    1, n_snap, figsize=(2.6 * n_snap, 3.2), sharex=True, sharey=True
)
if n_snap == 1:
    axes = [axes]
vmax = max(float(np.max(np.abs(p))) for _, p in snaps)
for col, (t_snap, p) in enumerate(snaps):
    im = axes[col].imshow(
        p,
        origin="lower",
        extent=[0, L, 0, L],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[col].set_title(f"t={t_snap:.2f}")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="p")
fig.suptitle(f"Acoustic Wave 2D  n={NX}x{NY}, c0={c0}, σ={σ}", fontsize=11)
out_snap = FIGURES_DIR / "acoustic_wave2d_snapshots.png"
fig.savefig(out_snap, dpi=140, bbox_inches="tight")
print(f"Saved: {out_snap}")

# Plot energy time series
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.linspace(0, Tend, nt), state_energies, label="Energy")
ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title("Energy Time Series")
ax.legend()
ax.grid(True)
out_energy = FIGURES_DIR / "acoustic_wave2d_energy.png"
fig.savefig(out_energy, dpi=140, bbox_inches="tight")
print(f"Saved: {out_energy}")
