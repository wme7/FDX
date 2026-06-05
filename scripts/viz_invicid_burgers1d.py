# Visualization of 1D Burgers' equation using Matplotlib and FDX library.

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fdx import essentially_nonoscillatory_grid as Ω
from fdx import time_integrators as τ

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "invicid_burgers1d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Constants
π = np.pi


# Define initial condition functions
def sine_wave(x):
    return np.sin(2 * np.pi * x) + 0.0


def square_jump(x):
    return np.where((x > 0.3) & (x < 0.5), 1.0, 0.0)


# Simulation parameters
L = 1.0  # Length of the domain
NX = 200  # Number of grid points
NS = 10  # Number of snapshots
RK = "RK3"  # Time-stepping method: "Euler/RK1", "RK2", "RK3", or "RK4"
CFL = 0.5  # CFL number for stability
Tend = 0.25  # End time of the simulation

# ----------------------------- setup ----------------------------- #

grid = Ω.Grid1d(
    a=0,
    b=L,
    n=NX,
    bc=Ω.BoundaryCondition.DIRICHLET,
    scheme=Ω.NonOscillatoryScheme.CRWENO5,
    verbose=False,
)
x, Δx = grid.x, grid.h  # Grid points and spacing


# RHS function for the Burgers' equation
def burgers_rhs(u):
    return np.where(u > 0, -u * (grid.Dx_upwind(u)), -u * (grid.Dx_downwind(u)))


# Set initial condition, time and iteration count
u0, t, it = sine_wave(x), 0.0, 0
u = u0.copy()  # Initialize solution array

# Compute time step based on CFL condition
Δt = CFL * Δx / np.abs(u).max()

# Initialize space time array
nt = 2 + int(Tend / Δt)  # 2: initial condition and final condition
u_st = np.zeros((len(x), NS + 1))
u_st[:, 0] = u0.copy()
freq = int(nt / NS)
k = 1

# Enable interactive mode
plt.ion()

# Create interactive plot
fig, ax = plt.subplots()
(line1,) = ax.plot(grid.x, u0, "-", label="Initial Condition")
(line2,) = ax.plot(grid.x, u, "-", label="Numerical Solution")
ax.set_xlabel("$X$")
ax.set_ylabel("$u(t)$")
ax.grid()
ax.legend()

# ----------------------------- driver ----------------------------- #

rk_scheme = τ.RungeKutta.from_name(RK)
tStart = time.process_time()
while t < Tend:
    # Update time & iteration count
    t += Δt
    it += 1

    # Update the solution
    u = rk_scheme.step(burgers_rhs, u, Δt)

    # Capture snapshot
    if it % freq == 0:
        u_st[:, k] = u.copy()
        k += 1

    # Update the plot
    if it % 10 == 0:  # Update the plot every 10 iterations
        line2.set_ydata(u)  # Update the plot with the new solution
        plt.draw()
        plt.pause(0.1)  # Small delay for animation

    # Adjust time step to ensure we don't exceed the end time
    if t + Δt > Tend:
        Δt = Tend - t

# Compute the elapsed time
elapsed = time.process_time() - tStart
print(f"CPU time: {elapsed:.4f} s")

# ------------------------- post-processing ------------------------- #

# Keep the plot open
plt.ioff()  # Disable interactive mode

# Plot the solution
fig, ax = plt.subplots()
time_legend = [f"t = {t:.2f}" for t in np.linspace(0, Tend, NS + 1)]
for j in range(0, NS):
    ax.plot(x, u_st[:, j], linewidth=1, label=time_legend[j])
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$U$")
ax.set_xlim(x.min(), x.max())
ax.grid(False)
ax.legend()
fig.tight_layout()
out_snap = FIGURES_DIR / "invicid_burgers1d_snapshots.png"
out_snap.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_snap, dpi=140, bbox_inches="tight")
print(f"Saved snapshots: {out_snap}")
plt.show()
