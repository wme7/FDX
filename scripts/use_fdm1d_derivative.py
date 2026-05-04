# %% Compute derivatives of a test function using the weights explicitly computed

import numpy as np
import sympy as sb
from matplotlib import pyplot as plt
from prettytable import PrettyTable as PT

from fdx.fornberg_weights import fd_explicit_weights
from fdx.utils import compute_order_of_accuracy

# Define explicit stencils
αL = np.array([0, 1, 2, 3, 4])  # left boundary
αl = np.array([-1, 0, 1, 2, 3])  # left interior
α0 = np.array([-2, -1, 0, 1, 2])  # interior scheme
αr = np.array([-3, -2, -1, 0, 1])  # right interior
αR = np.array([-4, -3, -2, -1, 0])  # right boundary

# Weights for each stencil
m_derivative = 1
w_αL = fd_explicit_weights(m=m_derivative, x=0, alpha=αL.tolist())
w_αl = fd_explicit_weights(m=m_derivative, x=0, alpha=αl.tolist())
w_α0 = fd_explicit_weights(m=m_derivative, x=0, alpha=α0.tolist())
w_αr = fd_explicit_weights(m=m_derivative, x=0, alpha=αr.tolist())
w_αR = fd_explicit_weights(m=m_derivative, x=0, alpha=αR.tolist())

# Print weights for each stencil
sb.pprint(
    f"α_coefs: {[sb.Rational(sb.nsimplify(v)) for v in w_αL]} - left boundary scheme"
)
sb.pprint(
    f"α_coefs: {[sb.Rational(sb.nsimplify(v)) for v in w_αl]} - left interior scheme"
)
sb.pprint(f"α_coefs: {[sb.Rational(sb.nsimplify(v)) for v in w_α0]} - interior scheme")
sb.pprint(
    f"α_coefs: {[sb.Rational(sb.nsimplify(v)) for v in w_αr]} - right interior scheme"
)
sb.pprint(
    f"α_coefs: {[sb.Rational(sb.nsimplify(v)) for v in w_αR]} - right boundary scheme"
)

# Numerical Grid and spacing
N = 24
L = 3.0
Δx = L / (N - 1)
x_num = np.linspace(0, L, N)
print(f"grid spacing: h = {Δx}")

# Test function and its derivative
x = sb.symbols("x")
# u = sb.sin(2 * np.pi * x)
u = sb.exp(-((x - L / 2) ** 2) / (2 * 0.25**2))
du = sb.diff(u, x, 1)
u_func = sb.lambdify(x, u, "numpy")
du_func = sb.lambdify(x, du, "numpy")

# Discrete u(x) values
u_num = u_func(x_num)

# Compute numerical derivative
du_num = np.zeros(N)

# Left boundaries schemes (Dirichlet BC):
du_num[0] = np.sum(w_αL * u_num[0 + αL]) / Δx
du_num[1] = np.sum(w_αl * u_num[1 + αl]) / Δx

# Interior point scheme:
for i in range(2, N - 2):
    du_num[i] = np.sum(w_α0 * u_num[i + α0]) / Δx

# Right boundaries schemes (Dirichlet BC):
du_num[N - 2] = np.sum(w_αr * u_num[N - 2 + αr]) / Δx
du_num[N - 1] = np.sum(w_αR * u_num[N - 1 + αR]) / Δx

# plot
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
    x_num = np.linspace(0, L, n)

    # Compute grid spacing
    Δx = L / (n - 1)
    h_list[i] = Δx

    # Compute numerical derivative
    u_num = u_func(x_num)
    du_num = np.zeros(n)

    # Left boundaries schemes (Dirichlet BC):
    du_num[0] = np.sum(w_αL * u_num[0 + αL]) / Δx
    du_num[1] = np.sum(w_αl * u_num[1 + αl]) / Δx

    # Interior point scheme:
    for j in range(2, n - 2):
        du_num[j] = np.sum(w_α0 * u_num[j + α0]) / Δx

    # Right boundaries schemes (Dirichlet BC):
    du_num[n - 2] = np.sum(w_αr * u_num[n - 2 + αr]) / Δx
    du_num[n - 1] = np.sum(w_αR * u_num[n - 1 + αR]) / Δx

    # Compute L1 norm of the error
    l1_list[i] = Δx * np.linalg.norm(du_num - du_func(x_num), ord=1)

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
