# Visualization of 1D Burgers' equation using Matplotlib and FDX library.

import time

import matplotlib.pyplot as plt
import numpy as np

from fdx import essentially_nonoscillatory_grid as Ω

# Constants
π = np.pi
f1o3, f2o3 = 1 / 3, 2 / 3
f1o4, f3o4 = 1 / 4, 3 / 4
half, f1o6 = 1 / 2, 1 / 6


# Define initial condition functions
def sine_wave(x):
    return np.sin(2 * np.pi * x) + 0.0


def square_jump(x):
    return np.where((x > 0.3) & (x < 0.5), 1.0, 0.0)


# Simulation parameters
L = 1.0  # Length of the domain
NX = 200  # Number of grid points
NS = 10  # Number of snapshots
RK = "RK3"  # Time-stepping method: "Euler", "RK2", "RK3", or "RK4"
CFL = 0.5  # CFL number for stability
Tend = 0.25  # End time of the simulation

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
def rhs(u):
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

# Time-stepping loop
tStart = time.process_time()
while t < Tend:
    # Update time & iteration count
    t += Δt
    it += 1

    # Update the solution
    match RK:
        case "Euler":
            u += Δt * rhs(u)
        case "RK2":
            k1 = rhs(u)
            k2 = rhs(u + Δt * k1)
            u += half * Δt * (k1 + k2)
        case "RK3":
            uo = u + Δt * rhs(u)
            us = f3o4 * u + f1o4 * (uo + Δt * rhs(uo))
            u = f1o3 * u + f2o3 * (us + Δt * rhs(us))
        case "RK4":
            k1 = rhs(u)
            k2 = rhs(u + half * Δt * k1)
            k3 = rhs(u + half * Δt * k2)
            k4 = rhs(u + Δt * k3)
            u += f1o6 * Δt * (k1 + 2 * k2 + 2 * k3 + k4)

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


# Keep the plot open
plt.ioff()  # Disable interactive mode


# Plot the solution
fig, ax = plt.subplots()
for j in range(1, NS):
    ax.plot(x, u_st[:, j], linewidth=1)
ax.set_xlabel(r"$X$")
ax.set_ylabel(r"$U$")
ax.set_xlim(x.min(), x.max())
ax.grid(False)
fig.tight_layout()
fig.savefig("invicid_burgers1d.pdf")
plt.show()
