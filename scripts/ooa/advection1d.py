# Test the order of accuracy of the 1D Advection solver.

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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "measurements" / "advection1d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sine_wave(x, t, L):
    ξ = (x - C * t) % L
    return np.sin(2 * π * ξ)


def gaussian_wave(x, t, L):
    ξ = (x + L / 2 - C * t) % L
    return np.exp(-((ξ - L / 2) ** 2) / 0.1**2)


# --------------------------- Constants --------------------------- #

π = np.pi
C = 1.0  # Advection speed
L = 1.0  # Length of the domain

# --------------------------- Parameters -------------------------- #

IC = "GAUSSIAN_WAVE"  # Initial condition: "SINE_WAVE", "GAUSSIAN_WAVE", "SQUARE_PULSE"
TS = "RK4"  # time-stepping method: "Euler/RK1", "RK2", "RK3", or "RK4"
SCH = "PADE"  # spatial scheme: "UPWIND", "DOWNWIND", "CENTRAL", etc.
R = 2  # stencil width
CFL = 0.6  # CFL number for stability
tEnd = 10.0  # End time of the simulation
REPEAT = 5  # Number of times to repeat the simulation
DEBUG = False

# ----------------------------- setup ----------------------------- #

# Initialize lists
# N_list = [20, 40, 80, 160, 320, 640, 1280]
N_list = [32, 64, 128, 256, 512, 1024]
h_list = np.zeros(len(N_list))
l1_list = np.zeros(len(N_list))
linf_list = np.zeros(len(N_list))
cpu_time_list = np.zeros(len(N_list))


def exact_solution(x, t):
    match IC:
        case "SINE_WAVE":
            return sine_wave(x, t, L)
        case "GAUSSIAN_WAVE":
            return gaussian_wave(x, t, L)
        case _:
            raise ValueError(f"Invalid initial condition: {IC}")


def advection_rhs_factory(grid):
    """Build the RHS for ∂u/∂t + C ∂u/∂x = 0."""

    def advection_rhs(u):
        match SCH:
            case "UPWIND":
                return -C * grid.Dx_upwind(u)
            case "DOWNWIND":
                return -C * grid.Dx_downwind(u)
            case "CENTRAL":
                return -C * grid.Dx_central(u)
            case "PADE":
                return -C * grid.Dx_pade(u)
            case "COMPACT":
                return -C * grid.Dx_compact(u)
            case "GODUNOV":
                return np.where(C > 0, -C * grid.Dx_upwind(u), -C * grid.Dx_downwind(u))
            case "FLUX_SPLITTING":
                f = C * u
                fP = 0.5 * (f + np.abs(C) * u)
                fN = 0.5 * (f - np.abs(C) * u)
                return -grid.Dx_upwind(fP) - grid.Dx_downwind(fN)
            case "RUSANOV":
                return -grid.Dx_rusanov(u, lambda u: C * u)
            case _:
                raise ValueError(f"Invalid scheme: {SCH}")

    return advection_rhs


# ----------------------------- TEST ----------------------------- #

for i, n in enumerate(N_list):
    # Use a periodic grid for x in [0, L] with N grid points
    grid = Ω.Grid1d(a=-L / 2, b=L / 2, n=n, bc=dΩ.PERIODIC, r_width=R)

    # Compute grid spacing
    h_list[i] = grid.h

    # Define RHS function for the advection equation
    advection_rhs = advection_rhs_factory(grid)

    # CFL: Δt ≤ CFL · Δx / |C|
    Δt0 = CFL * grid.h / np.abs(C)

    # Repeat the simulation REPEAT times
    for rep in range(REPEAT):
        # Compute initial state and time
        t0 = 0.0
        initial_state = exact_solution(grid.nodes, t0)

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
    expectation = exact_solution(grid.nodes, time)

    # Plot the exact solution and the numerical solution
    if DEBUG:
        plt.plot(grid.nodes, expectation, label="Exact Solution")
        plt.plot(grid.nodes, state, label="Numerical Solution")
        plt.legend()
        plt.show()

    # Compute the error norms
    l1_list[i] = grid.h * np.linalg.norm(state - expectation, ord=1)
    linf_list[i] = grid.h * np.linalg.norm(state - expectation, ord=np.inf)

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
        f"{TS}_{SCH}_r{grid.r}",
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
tag = f"{GRID_TYPE}_advection1d"
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
