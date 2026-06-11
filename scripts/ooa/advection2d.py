# Test the order of accuracy of the 2D Advection solver.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if True:
    from fdx import finite_differences_grid as Ω
    from fdx.finite_differences_grid import BoundaryCondition as dΩ

    GRID_TYPE = "FD"
else:
    from fdx import essentially_nonoscillatory_grid as Ω
    from fdx.essentially_nonoscillatory_grid import BoundaryCondition as dΩ

    GRID_TYPE = "ENO"

from prettytable import PrettyTable as PT

from fdx.drivers import Driver4TestOrderOfAccuracy
from fdx.time_integrators import RungeKutta as rk
from fdx.utils import compute_order_of_accuracy as OOA

RESULTS_DIR = Path(__file__).resolve().parent.parent / "measurements" / "advection2d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sine_wave(x, y, t, L):
    ξx = (x - Cx * t) % L
    ξy = (y - Cy * t) % L
    return np.sin(2 * π * ξx) * np.sin(2 * π * ξy)


def gaussian_wave(x, y, t, L):
    ξx = (x + L / 2 - Cx * t) % L
    ξy = (y + L / 2 - Cy * t) % L
    return np.exp(-((ξx - L / 2) ** 2 + (ξy - L / 2) ** 2) / 0.1**2)


# --------------------------- Constants --------------------------- #

π = np.pi
Cx = 1.0  # Advection speed in x
Cy = 0.5  # Advection speed in y
L = 1.0  # Length of the domain (square, periodic in x and y)

# --------------------------- Parameters -------------------------- #

IC = "GAUSSIAN_WAVE"  # Initial condition: "SINE_WAVE", "GAUSSIAN_WAVE"
TS = "RK4"  # time-stepping method: "Euler/RK1", "RK2", "RK3", or "RK4"
SCH = "CENTRAL"  # spatial scheme: "UPWIND", "DOWNWIND", "CENTRAL", etc.
R = 2  # stencil width
CFL = 0.8  # CFL number for stability
tEnd = 10.0  # End time of the simulation
REPEAT = 5  # Number of times to repeat the simulation
DEBUG = False

# ----------------------------- setup ----------------------------- #

# Initialize lists
# N_list = [20, 40, 80, 160, 320, 640]
N_list = [16, 32, 64, 128, 256, 512]
h_list = np.zeros(len(N_list))
l1_list = np.zeros(len(N_list))
linf_list = np.zeros(len(N_list))
cpu_time_list = np.zeros(len(N_list))


def exact_solution(x, y, t):
    match IC:
        case "SINE_WAVE":
            return sine_wave(x, y, t, L)
        case "GAUSSIAN_WAVE":
            return gaussian_wave(x, y, t, L)
        case _:
            raise ValueError(f"Invalid initial condition: {IC}")


def advection_rhs_factory(grid):
    """Build the RHS for ∂u/∂t + Cx ∂u/∂x + Cy ∂u/∂y = 0."""

    def advection_rhs(u):
        match SCH:
            case "UPWIND":
                return -Cx * grid.Dx_upwind(u) - Cy * grid.Dy_upwind(u)
            case "DOWNWIND":
                return -Cx * grid.Dx_downwind(u) - Cy * grid.Dy_downwind(u)
            case "CENTRAL":
                # u_flat = u.ravel()
                # du_dx = grid.Dx_operator @ u_flat
                # du_dy = grid.Dy_operator @ u_flat
                # return -(Cx * du_dx + Cy * du_dy).reshape(u.shape)
                return -Cx * grid.Dx_central(u) - Cy * grid.Dy_central(u)
            case "PADE":
                return -Cx * grid.Dx_pade(u) - Cy * grid.Dy_pade(u)
            case "COMPACT":
                return -Cx * grid.Dx_compact(u) - Cy * grid.Dy_compact(u)
            case "GODUNOV":
                du_dx = grid.Dx_upwind(u) if Cx > 0 else grid.Dx_downwind(u)
                du_dy = grid.Dy_upwind(u) if Cy > 0 else grid.Dy_downwind(u)
                return -Cx * du_dx - Cy * du_dy
            case "FLUX_SPLITTING":
                fx = Cx * u
                fxP = 0.5 * (fx + np.abs(Cx) * u)
                fxN = 0.5 * (fx - np.abs(Cx) * u)
                fy = Cy * u
                fyP = 0.5 * (fy + np.abs(Cy) * u)
                fyN = 0.5 * (fy - np.abs(Cy) * u)
                return (
                    -grid.Dx_upwind(fxP)
                    - grid.Dx_downwind(fxN)
                    - grid.Dy_upwind(fyP)
                    - grid.Dy_downwind(fyN)
                )
            case "RUSANOV":
                return -grid.Dx_rusanov(u, lambda u: Cx * u) - grid.Dy_rusanov(
                    u, lambda u: Cy * u
                )
            case _:
                raise ValueError(f"Invalid scheme: {SCH}")

    return advection_rhs


# ----------------------------- TEST ----------------------------- #

for i, n in enumerate(N_list):
    # Periodic square grid on [-L/2, L/2] × [-L/2, L/2] with n × n points
    # fmt: off
    grid = Ω.Grid2d(
        xa=-L / 2, xb=L / 2, nx=n, bc_x=dΩ.PERIODIC, r_width_x=R,
        ya=-L / 2, yb=L / 2, ny=n, bc_y=dΩ.PERIODIC, r_width_y=R,
    )
    # fmt: on

    # Compute grid spacing (square cells)
    h_list[i] = grid.x.h

    # Define RHS function for the advection equation
    advection_rhs = advection_rhs_factory(grid)

    # CFL: Δt ≤ CFL · min(Δx, Δy) / (|Cx| + |Cy|)
    Δt0 = CFL * min(grid.x.h, grid.y.h) / (np.abs(Cx) + np.abs(Cy))

    # Repeat the simulation REPEAT times
    for rep in range(REPEAT):
        # Compute initial state and time
        t0 = 0.0
        initial_state = exact_solution(grid.X, grid.Y, t0)

        # Numerical Driver
        time, state, cpu_time = Driver4TestOrderOfAccuracy(
            rk_scheme=rk.from_name(TS),
            initial_state=initial_state,
            initial_time=t0,
            final_time=tEnd,
            initial_time_step=Δt0,
            rhs_fn=advection_rhs,
        )

        # Accumulate the CPU time (skip first repetition)
        if rep > 0:
            cpu_time_list[i] += cpu_time

    # Compute the exact solution at the final time
    expectation = exact_solution(grid.X, grid.Y, time)

    # Plot the exact solution and the numerical solution
    if DEBUG:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        vmin = min(expectation.min(), state.min())
        vmax = max(expectation.max(), state.max())
        axes[0].imshow(expectation, origin="lower", vmin=vmin, vmax=vmax)
        axes[0].set_title("Exact Solution")
        axes[1].imshow(state, origin="lower", vmin=vmin, vmax=vmax)
        axes[1].set_title("Numerical Solution")
        axes[2].imshow(np.abs(state - expectation), origin="lower")
        axes[2].set_title("|Error|")
        plt.tight_layout()
        plt.show()

    # Compute the error norms (discrete L1 / L∞ with cell area weighting)
    area = grid.x.h * grid.y.h
    error = (state - expectation).ravel()
    l1_list[i] = area * np.linalg.norm(error, ord=1)
    linf_list[i] = area * np.linalg.norm(error, ord=np.inf)

# ----------------------- Post-processing ------------------------- #

# Compute the order of accuracy
order_of_accuracy = OOA(h_list, l1_list)

# Return table of results
table = PT()
table.field_names = [
    "Method",
    "CFL",
    "N",
    "h",
    "L1 Error",
    "Linf Error",
    "Order",
    "Wall-Time (s)",
]
for i in range(len(N_list)):
    row = [
        f"{TS}_{SCH}_r{grid.x.r}",
        f"{CFL:.2f}",
        N_list[i],
        f"{h_list[i]:.3e}",
        f"{l1_list[i]:.3e}",
        f"{linf_list[i]:.3e}",
        f"{order_of_accuracy[i]:.2f}" if i > 0 else "-",
        f"{cpu_time_list[i] / (REPEAT - 1):.3f}",
    ]
    table.add_row(row)

# Print table of results to terminal
print(table)

# Save table of results to file
tag = f"{GRID_TYPE}_advection2d"
with open(RESULTS_DIR / f"{tag}.txt", "a") as f:
    f.write(table.get_string())
    f.write("\n")

# Save table of results to LaTeX file
with open(RESULTS_DIR / f"{tag}.tex", "a") as f:
    f.write(table.get_latex_string())
    f.write("\n")

# Save table of results to CSV file
with open(RESULTS_DIR / f"{tag}.csv", "a") as f:
    f.write(table.get_csv_string())
    f.write("\n")
