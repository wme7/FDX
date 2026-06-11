from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from fdx import finite_differences_grid as Ω
from fdx.finite_differences_grid import BoundaryCondition as dΩ

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "field2d"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def Gaussian_pulse(x, y, x0=0.5, y0=0.5, sigma=0.06):
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma**2)


# Simulation parameters
x0, y0, σ = 0.5, 1.5, 0.10
nx, ny = 32, 32
bc = dΩ.DIRICHLET
n_gp = 2

# Initialize 4 2D-grids arranged in a rotated L-shaped domain
grids = []
grids.append(Ω.Grid2d(0, 1, nx, 0, 1, ny, bc_x=bc, bc_y=bc))
grids.append(Ω.Grid2d(0, 1, nx, 1, 2, ny, bc_x=bc, bc_y=bc))
grids.append(Ω.Grid2d(1, 2, nx, 1, 2, ny, bc_x=bc, bc_y=bc))
grids.append(Ω.Grid2d(2, 3, nx, 1, 2, ny, bc_x=bc, bc_y=bc))

# Initialize the 4 grids
X, Y, states = [], [], []
for g in grids:
    x, y = np.meshgrid(g.x.nodes, g.y.nodes, indexing="xy")
    u = Gaussian_pulse(x, y, x0, y0, σ)
    v = Gaussian_pulse(x, y, x0, y0, σ)
    X.append(x)
    Y.append(y)
    states.append(np.stack([u, v], axis=0))

# Plot the 4 grids using matplotlib
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_aspect("equal")
for g, x, y in zip(grids, X, Y):
    # Nodes
    ax.plot(x.ravel(), y.ravel(), "k.", markersize=2)
    # Domain outline
    ax.plot(
        [g.x.a, g.x.b, g.x.b, g.x.a, g.x.a],
        [g.y.a, g.y.a, g.y.b, g.y.b, g.y.a],
        "b-",
        lw=1.5,
    )
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()
plt.show()

# Plot the initial condition
fix, ax = plt.subplots(figsize=(6, 8))
ax.set_aspect("equal")
for x, y, q in zip(X, Y, states):
    ax.pcolormesh(x, y, q[0], cmap="jet", vmin=0, vmax=1)
ax.grid()
plt.show()
