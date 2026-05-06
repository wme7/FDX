# %% Compute derivatives of a test function using FDX Grid2d class

import os

import numpy as np
import sympy as sb
from matplotlib import pyplot as plt

from fdx import finite_differences_grid as Ω

# Symbols & Constants
π = np.pi
x, y = sb.symbols("x, y")

# Control parameters
Nx, Ny = 100, 100  # number of grid points in x and y directions
Lx, Ly = 1.0, 1.0  # domain length in x and y directions
rx, ry = 2, 2  # stencil width (total width is 2*r + 1)
BCx, BCy = Ω.BoundaryCondition.PERIODIC, Ω.BoundaryCondition.DIRICHLET
FD = Ω.FiniteDifferenceScheme.EXPLICIT

# Grid setup
grid = Ω.Grid2d(
    xa=0.0,
    xb=Lx,
    nx=Nx,
    rx=rx,
    ya=0.0,
    yb=Ly,
    ny=Ny,
    ry=ry,
    bcx=BCx,
    bcy=BCy,
    scheme=FD,
    verbose=False,
)

x_num, y_num = np.meshgrid(grid.x, grid.y)
print(f"grid spacing: hx = {grid.hx}, hy = {grid.hy}")

# Set test function
u = sb.sin(2 * π * x) * sb.sin(2 * π * y)

# Compute derivatives (Exact solutions)
du_dx = sb.diff(u, x, 1)
du_dy = sb.diff(u, y, 1)
du_dx_dy = sb.diff(du_dx, y, 1)
grad_u = sb.Matrix([du_dx, du_dy])
laplacian_u = sb.diff(u, x, 2) + sb.diff(u, y, 2)

# Convert to NumPy functions
u_func = sb.lambdify([x, y], u, "numpy")
du_dx_func = sb.lambdify([x, y], du_dx, "numpy")
du_dy_func = sb.lambdify([x, y], du_dy, "numpy")
du_dx_dy_func = sb.lambdify([x, y], du_dx_dy, "numpy")
grad_u_func = sb.lambdify([x, y], grad_u, "numpy")
laplacian_u_func = sb.lambdify([x, y], laplacian_u, "numpy")

# Compute numerical derivatives
u_num = u_func(x_num, y_num)
Dux = grid.Derivative(u_num, "x")
Duy = grid.Derivative(u_num, "y")
Duxy = grid.Derivative(u_num, "xy")
Gradu = grid.Grad(u_num)
Lu = grid.Laplacian(u_num)

# Plot Derivatives and Laplacian
fig = plt.figure(figsize=(10, 16))
plt.subplot(4, 2, 1)
plt.imshow(du_dx_func(x_num, y_num), cmap="jet")
plt.colorbar()
plt.title("$\\partial_x u$")
plt.subplot(4, 2, 2)
plt.imshow(Dux, cmap="jet")
plt.colorbar()
plt.title("$D_x u$")
plt.subplot(4, 2, 3)
plt.imshow(du_dy_func(x_num, y_num), cmap="jet")
plt.colorbar()
plt.title("$\\partial_y u$")
plt.subplot(4, 2, 4)
plt.imshow(Duy, cmap="jet")
plt.colorbar()
plt.title("$D_y u$")
plt.subplot(4, 2, 5)
plt.imshow(du_dx_dy_func(x_num, y_num), cmap="jet")
plt.colorbar()
plt.title("$\\partial_{xy} u$")
plt.subplot(4, 2, 6)
plt.imshow(Duxy, cmap="jet")
plt.colorbar()
plt.title("$D_x D_y u$")
plt.subplot(4, 2, 7)
plt.imshow(laplacian_u_func(x_num, y_num), cmap="jet")
plt.colorbar()
plt.title("$\\nabla^2 u$")
plt.subplot(4, 2, 8)
plt.imshow(Lu, cmap="jet")
plt.colorbar()
plt.title("$(D_x^2 + D_y^2) u$")
plt.show()

# %%
PATH = "../figures/derivatives"
os.makedirs(PATH, exist_ok=True)
fig.savefig(os.path.join(PATH, "operators_2d.png"), bbox_inches="tight")
