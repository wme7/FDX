# Visualization of 1D Advection using Matplotlib and FDX project

import matplotlib.pyplot as plt
import numpy as np

from fdx import finite_differences_grid as Ω

# Constants
π = np.pi
f1o3, f2o3 = 1 / 3, 2 / 3
f1o4, f3o4 = 1 / 4, 3 / 4


# Define initial condition functions
def sine_wave(x):
    return np.sin(2 * np.pi * x)


def gaussian_pulse(x, center=0.5, width=0.1):
    return np.exp(-((x - center) ** 2) / (2 * width**2))


def square_wave(x, center=0.5, width=0.3):
    return np.where((x >= center - width / 2) & (x <= center + width / 2), 1.0, 0.0)


# Simulation parameters
C = 1.0  # Advection speed
L = 1.0  # Length of the domain
NX = 64  # Number of grid points
RK = "RK4"  # Time-stepping method: "Euler", "RK2", "RK3", or "RK4"
CFL = 0.8  # CFL number for stability
Tend = 100.0  # End time of the simulation

# Use a periodic grid fo x in [0, 1] with 100 points
grid = Ω.Grid1d(
    a=0,
    b=L,
    n=NX,
    r=2,
    bc=Ω.BoundaryCondition.PERIODIC,
    scheme=Ω.FiniteDifferenceScheme.PADE,
    verbose=False,
)
x, Δx = grid.x, grid.h  # Grid points and spacing

# Compute time step based on CFL condition
Δt = CFL * Δx / np.abs(C)
halfdt, f1o6dt = Δt / 2, Δt / 6


# RHS function for the advection equation
def rhs(u):
    return -C * (grid.Dx @ u)


# Set initial condition, time and iteration count
u0, t, it = gaussian_pulse(x), 0.0, 0
u = u0.copy()  # Initialize solution array

# Enable interactive mode
plt.ion()

# Create interactive plot
fig, ax = plt.subplots()
(line1,) = ax.plot(grid.x, u0, "-", label="Initial Condition")
(line2,) = ax.plot(grid.x, u, ".", label="Advection Solution")
ax.set_xlabel("$X$")
ax.set_ylabel("$u(t)$")
ax.legend()

# Time-stepping loop
while t < Tend:
    # Update time & iteration count
    t += Δt
    it += 1

    # Update the solution
    match RK:
        case "Euler":
            u = u + Δt * rhs(u)
        case "RK2":
            uo = u.copy()
            k1 = rhs(u)
            k2 = rhs(u + Δt * k1)
            u = uo + halfdt * (k1 + k2)
        case "RK3":
            uo = u + Δt * rhs(u)
            us = f3o4 * u + f1o4 * (uo + Δt * rhs(uo))
            u = f1o3 * u + f2o3 * (us + Δt * rhs(us))
        case "RK4":
            uo = u.copy()
            k1 = rhs(u)
            k2 = rhs(u + halfdt * k1)
            k3 = rhs(u + halfdt * k2)
            k4 = rhs(u + Δt * k3)
            u = uo + f1o6dt * (k1 + 2 * k2 + 2 * k3 + k4)

    # Update the plot
    if it % 10 == 0:  # Update the plot every 10 iterations
        line2.set_ydata(u)  # Update the plot with the new solution
        plt.draw()
        plt.pause(0.0001)  # Small delay for animation

    # Adjust time step to ensure we don't exceed the end time
    if t + Δt > Tend:
        Δt = Tend - t

# Plot the final solution
line2.set_ydata(u)
plt.draw()
plt.pause(0.0001)

# Keep the plot open
plt.ioff()  # Disable interactive mode
plt.show()
