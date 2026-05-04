# %% Compute derivatives of a test function using FDX Grid1d class

import numpy as np
import sympy as sb
from matplotlib import pyplot as plt
from prettytable import PrettyTable as PT

from fdx import finite_differences_grid as Ω
from fdx.utils import compute_order_of_accuracy

# Symbols & Constants
π = np.pi
x = sb.symbols("x")

# Control parameters
N = 24  # number of grid points
L = 3.0  # domain length
r = 2  # stencil width (total width is 2*r + 1)
BC = Ω.BoundaryCondition.DIRICHLET
FD = Ω.FiniteDifferenceScheme.EXPLICIT

# Create grid
grid = Ω.Grid1d(a=0, b=L, n=N, r=r, bc=BC, scheme=FD, verbose=True)
print(f"grid spacing: h = {grid.h}")

# Set Test function
match grid.bc:
    case Ω.BoundaryCondition.PERIODIC:
        # use a periodic sine function as test function
        u = sb.sin(2 * π * x)
    case _:
        # use a Gaussian function as test function
        u = sb.exp(-((x - L / 2) ** 2) / (2 * 0.25**2))

du = sb.diff(u, x, 1)
u_func = sb.lambdify(x, u, "numpy")
du_func = sb.lambdify(x, du, "numpy")

# Compute numerical derivative
x_num = grid.x
u_num = u_func(x_num)
du_num = grid.Dx @ u_num

# Plot
x_func = np.linspace(0, L, 1000)
plt.plot(x_func, u_func(x_func), "-b", label="Analytical $u(x)$")
plt.plot(x_num, u_num, ".k", label="Numerical $u(x)$")
plt.plot(x_func, du_func(x_func), "-r", label="Analytical $u'(x)$")
plt.plot(x_num, du_num, ".k", label="Numerical $u'(x)$")
plt.legend(loc="best")
plt.title("Derivative of sin(2πx)")
plt.xlabel("x")
plt.ylabel("du/dx")
plt.xlim(0, L)
plt.show()

# %% ---------------------------------------------------------------------- #
# Does the derivative converge to the analytical solution?
# ------------------------------------------------------------------------- #
N_list = [75, 125, 250, 500, 1000]  # grid size
h_list = np.zeros(len(N_list))
l1_list = np.zeros(len(N_list))

for i, n in enumerate(N_list):
    # Create numerical grid
    grid = Ω.Grid1d(a=0, b=L, n=n, r=r, bc=BC, scheme=FD)

    # Compute grid spacing
    h_list[i] = grid.h

    # Compute numerical derivative
    du_num = grid.Dx @ u_func(grid.x)

    # Compute L1 norm of the error
    l1_list[i] = grid.h * np.linalg.norm(du_num - du_func(grid.x), ord=1)

# Plot the error vs. grid spacing
plt.loglog(h_list, l1_list, "-s", label="$D_x$")
plt.xlabel("h")
plt.ylabel("error")
plt.grid()
plt.legend()
plt.show()

# Print Table of Results
table = PT()
table.field_names = ["N", "h", "L1 Error Dx", "Order Dx"]
order_of_accuracy = compute_order_of_accuracy(h_list, l1_list)

for i in range(len(N_list)):
    row = [
        N_list[i],
        f"{h_list[i]:.3e}",
        f"{l1_list[i]:.3e}",
        f"{order_of_accuracy[i]:.2f}" if i > 0 else "-",
    ]
    table.add_row(row)

print(table)
