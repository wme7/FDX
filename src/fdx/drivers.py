# Drivers for the FDX library

import time as cpu
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from .time_integrators import RungeKutta as rk

# ---------------------------------------------------------------------- #
# Visualization Driver
# ---------------------------------------------------------------------- #


def Driver4Visualization(
    rk_scheme: rk,
    initial_state: np.array,
    initial_time: float,
    final_time: float,
    initial_time_step: float,
    rhs_fn: Callable,
    *,
    sample_frequency: int | None = None,
    sample_fn: Callable | None = None,
    number_of_snapshots: int | None = None,
    plot_object: Line2D | AxesImage | None = None,
    pause: float = 0.1,
):
    """
    Driver for visualization of 1D/2D simulations.
    """

    # Initial condition and time
    state = initial_state.copy()
    time_step = initial_time_step
    time = initial_time
    iterations = 0

    # Initialize snapshot arrays
    if number_of_snapshots is not None:
        nt = 1 + int(final_time / initial_time_step)
        snapshot_frequency = int(nt / number_of_snapshots)
        snapshots = [(time, state.copy())]
    else:
        snapshots = []

    # Initialize sample function
    if sample_fn is not None and sample_frequency is not None:
        n_samples = 1 + int(final_time / initial_time_step) // sample_frequency
        t_sample, q_sample = np.zeros(n_samples), np.zeros(n_samples)
        t_sample[0], q_sample[0] = time, sample_fn(state)
        k = 1  # sample counter

    # Enable interactive mode
    plt.ion()

    # Start the simulation
    start_time = cpu.process_time()

    # Main simulation loop
    while time < final_time:
        # Update time & iteration count
        time += time_step
        iterations += 1

        # Update the solution
        state = rk_scheme.step(rhs_fn, state, time_step)

        # Capture snapshot
        if number_of_snapshots is not None:
            if iterations % snapshot_frequency == 0:
                snapshots.append((time, state.copy()))

        # Capture sample
        if sample_fn is not None:
            if iterations % sample_frequency == 0:
                t_sample[k], q_sample[k] = time, sample_fn(state)
                k += 1

        # Update the plot data
        if iterations % snapshot_frequency == 0 or time == final_time:
            match plot_object:
                case Line2D():
                    plot_object.set_ydata(state)
                case AxesImage():
                    plot_object.set_data(state[0])  # Only 1st component
            plt.draw()
            plt.pause(pause)

        # Adjust time step to ensure we don't exceed the end time
        if time + time_step > final_time:
            time_step = final_time - time

    # Compute the elapsed time
    elapsed = cpu.process_time() - start_time

    # Short report on the simulation
    fields = {
        "dt0": f"{initial_time_step:.6f}",
        "iterations": iterations,
        "OUT time": f"{time:.2f}",
        "CPU time": f"{elapsed:.2f} s",
    }
    body = ", ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{rk_scheme.name}({body})")

    # Keep the plot open
    plt.ioff()  # Disable interactive mode

    # Return the results
    if sample_frequency is not None and number_of_snapshots is not None:
        return [t_sample, q_sample], snapshots
    elif sample_frequency is not None:
        return [t_sample, q_sample]
    elif number_of_snapshots is not None:
        return snapshots
    else:
        return [time, state]


# ---------------------------------------------------------------------- #
# Animation Driver
# ---------------------------------------------------------------------- #


def Driver4Animation(
    rk_scheme: rk,
    initial_state: np.array,
    initial_time: float,
    final_time: float,
    initial_time_step: float,
    rhs_fn: Callable,
    number_of_frames: int,
):
    """
    Driver for animation of 1D/2D simulations.
    """

    # Initial condition and time
    state = initial_state.copy()
    time_step = initial_time_step
    time = initial_time
    iterations = 0

    # Initialize frame array
    frames = [(time, state.copy())]
    nt_estimate = int((final_time - initial_time) / time_step)
    frame_frequency = max(1, nt_estimate // number_of_frames)

    # Start the simulation
    start_time = cpu.process_time()

    # Solution loop
    while time < final_time:
        # Update time & iteration count
        time += time_step
        iterations += 1

        # Update the solution
        state = rk_scheme.step(rhs_fn, state, time_step)

        # NaN detection
        if np.isnan(state).any():
            raise ValueError(f"NaN detected at time {time}")

        # Capture frame
        if iterations % frame_frequency == 0:
            frames.append((time, state.copy()))

    # Compute the elapsed time
    elapsed = cpu.process_time() - start_time

    # Short report on the simulation
    fields = {
        "dt0": f"{initial_time_step:.6f}",
        "iterations": iterations,
        "OUT time": f"{time:.2f}",
        "CPU time": f"{elapsed:.2f} s",
    }
    body = ", ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{rk_scheme.name}({body})")

    # Return the results
    return frames


# ---------------------------------------------------------------------- #
# Test Convergence Driver
# ---------------------------------------------------------------------- #


def Driver4TestOrderOfAccuracy(
    rk_scheme: rk,
    initial_state: np.array,
    initial_time: float,
    final_time: float,
    initial_time_step: float,
    rhs_fn: Callable,
):
    """
    Minimal driver for testing convergence and cpu times of a numerical method.
    """

    # Initial condition and time
    state = initial_state.copy()
    time_step = initial_time_step
    time = initial_time
    iterations = 0

    # Start the simulation
    start_time = cpu.process_time()

    # Solution loop
    while time < final_time:
        # Update time & iteration count
        time += time_step
        iterations += 1

        # Update the solution
        state = rk_scheme.step(rhs_fn, state, time_step)

        # NaN detection
        if np.isnan(state).any():
            raise ValueError(f"NaN detected at time {time}")

    # Compute the elapsed time
    elapsed = cpu.process_time() - start_time

    # Short report on the simulation
    fields = {
        "dt0": f"{initial_time_step:.6f}",
        "iterations": iterations,
        "OUT time": f"{time:.2f}",
        "CPU time": f"{elapsed:.2f} s",
    }
    body = ", ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{rk_scheme.name}({body})")

    # Return the results
    return time, state, elapsed
