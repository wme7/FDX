# %% 2D animations in 300+ dpi, saved as gif.
#
# Two animated scenarios:
#   1. "acoustics": Gaussian pressure pulse propagating in 2D linear
#      acoustics, integrated with skew-symmetric CUD-5 (zero
#      dissipation; only RK4 time error perturbs the energy).
#   2. "swirling":  Off-centre Gaussian advected by a steady rigid-
#      body rotation, integrated with CUD-5 multioperator (mirrored
#      s-list).  Exact solution at t = 1 coincides with the initial
#      condition (one full revolution).
#
# Usage:
#     python scripts/use_mo_2d_animations.py                 # both
#     python scripts/use_mo_2d_animations.py --only acoustics
#     python scripts/use_mo_2d_animations.py --only swirling --frames 60

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fdx.compact_upwind_differences_grid import (
    BoundaryCondition,
    CUDGrid2d,
    build_skew_symmetric_cud5,
)
from fdx.compact_upwind_differences_grid import (
    CompactUpwindDifferenceScheme as CUD,
)
from fdx.drivers import Driver4Animation
from fdx.time_integrators import RK4

ANIMATIONS_DIR = Path(__file__).resolve().parent.parent / "figures" / "tolstykh"
ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)

_PERIODIC = BoundaryCondition.PERIODIC


def gaussian_pulse(X, Y, x0=0.5, y0=0.5, sigma=0.1):
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / sigma**2)


def acoustic_pulse(X, Y, x0=0.5, y0=0.5, sigma=0.1):
    p0 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / sigma**2)
    u0 = np.zeros_like(p0)
    v0 = np.zeros_like(p0)
    return np.stack([p0, u0, v0], axis=0)


def rotational_field(X, Y, omega=1.0):
    U = -omega * Y
    V = +omega * X
    return U.ravel(), V.ravel()


# -------------------- 2D acoustics simulation -------------------- #


def simulate_acoustics(simulation_time, cfl, frames_per_unit_time, resolution=96):
    """Run the acoustics simulation and return list[(t, p_frame)]."""
    c0 = 1.0
    h = 1.0 / resolution
    x = np.linspace(0, 1, resolution, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")

    # Skew-symmetric CUD-5, mirrored s-list (zero dissipation by construction).
    D = build_skew_symmetric_cud5([0.5, 1.0], resolution, h, _PERIODIC)

    # Initial condition
    state = acoustic_pulse(X, Y, 0.5, 0.5, 0.06)

    # Compute time step based on CFL condition
    dt = cfl * h / c0

    # Acoustics right-hand side
    def acoustics_rhs(state):
        p, u, v = state[0], state[1], state[2]
        div_u = u @ D.T + D @ v
        dp_dx = p @ D.T
        dp_dy = D @ p
        return np.stack([-(c0**2) * div_u, -dp_dx, -dp_dy], axis=0)

    frames_vec_state = Driver4Animation(
        rk_scheme=RK4(),
        initial_state=state,
        initial_time=0.0,
        final_time=simulation_time,
        initial_time_step=dt,
        rhs_fn=acoustics_rhs,
        number_of_frames=frames_per_unit_time * simulation_time,
    )
    # Return only the first field in state vector array
    frames = [(t, vec_state[0]) for t, vec_state in frames_vec_state]
    return X, Y, frames


# ------------- swirling-vortex advection simulation -------------- #


def simulate_swirling(
    simulation_time, cfl, omega, frames_per_unit_time, resolution=128
):
    """Run swirling-vortex advection: u = -omega y, v = +omega x; omega = 2 pi."""
    h = 2.0 / resolution
    x = np.linspace(-1, 1, resolution, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")

    # Sparse Kronecker linop via the matrix-free 2D path (avoids dense O(n²) D1).
    # fmt: off
    g = CUDGrid2d(
        xa=-1, xb=1, nx=resolution, bcx=_PERIODIC,
        ya=-1, yb=1, ny=resolution, bcy=_PERIODIC,
        scheme=CUD.CUD5_MULTIOP,
        s_list=[-1.0, -0.5, 0.5, 1.0],
    )
    # fmt: on

    # Initial condition: off-centre Gaussian, r0=0.2 around (0.3, 0).
    phi0_flat = gaussian_pulse(X, Y, x0=0.3, y0=0.0, sigma=0.2).ravel()

    # Rotational velocity field
    U_flat, V_flat = rotational_field(X, Y, omega)

    # Compute time step based on CFL condition
    dt = cfl * h / float(np.max(np.sqrt(U_flat**2 + V_flat**2)))

    # Advection right-hand side
    def advection_rhs(phi_flat):
        # dphi/dt = -(u phi_x + v phi_y)
        Dx_phi = g.Dx_linop_2d @ phi_flat
        Dy_phi = g.Dy_linop_2d @ phi_flat
        return -(U_flat * Dx_phi + V_flat * Dy_phi)

    frames_scalar_state_flat = Driver4Animation(
        rk_scheme=RK4(),
        initial_state=phi0_flat,
        initial_time=0.0,
        final_time=simulation_time,
        initial_time_step=dt,
        rhs_fn=advection_rhs,
        number_of_frames=frames_per_unit_time * simulation_time,
    )
    # return reshaped frames
    frames = [
        (t, scalar_state_flat.reshape(X.shape))
        for t, scalar_state_flat in frames_scalar_state_flat
    ]
    return X, Y, frames


# ----------------------------- animation rendering ----------------------------- #
def make_animation(
    X,
    Y,
    frames,
    title,
    cmap,
    vmin,
    vmax,
    out_basename,
    fps=24,
    dpi=300,
):
    """Build a FuncAnimation, save it to both .gif and .mp4."""
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(
        frames[0][1],
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(title.format(t=0.0), fontsize=10)
    fig.tight_layout()

    def update(idx):
        t, field = frames[idx]
        im.set_data(field)
        title.set_text(title.format(t=t))
        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        blit=False,  # title text excludes blit
        interval=1000 / fps,
    )

    # ---- save as GIF ----
    gif_path = ANIMATIONS_DIR / f"{out_basename}.gif"
    time_start = time.perf_counter()
    anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    time_processing = time.perf_counter() - time_start
    print(f"Animation saved to: {gif_path}, processing time: {time_processing:.1f}s")
    plt.close(fig)


# ----------------------------- main ----------------------------- #
def main():
    p = argparse.ArgumentParser(description="2D animations in 300+ dpi, saved as gif.")
    p.add_argument(
        "--only",
        choices=("acoustics", "swirling"),
        help="run only the named scenario (default: both)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=48,
        help="number of frames per unit time (default 24)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="grid resolution (default: 100)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="per-frame DPI (default 300)",
    )
    args = p.parse_args()

    if args.only is None or args.only == "acoustics":
        print("Acoustics scenario")
        X, Y, frames = simulate_acoustics(
            simulation_time=0.4,
            cfl=0.5,
            frames_per_unit_time=args.frames,
            resolution=args.resolution,
        )
        vmax = float(np.max(np.abs(frames[0][1]))) * 0.7
        make_animation(
            X,
            Y,
            frames,
            title=r"2D acoustics, skew CUD-5  —  $t = {t:.2f}$",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=+vmax,
            out_basename="anim_acoustics_2d",
            fps=24,
            dpi=args.dpi,
        )
        print()

    if args.only is None or args.only == "swirling":
        print("Swirling-vortex scenario")
        X, Y, frames = simulate_swirling(
            simulation_time=1.0,
            cfl=0.5,
            omega=2 * np.pi,
            frames_per_unit_time=args.frames,
            resolution=args.resolution,
        )
        vmax = float(np.max(frames[0][1]))
        make_animation(
            X,
            Y,
            frames,
            title=r"2D rotation, CUD-5 multiop M=4  —  $t = {t:.3f}$",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            out_basename="anim_swirling_vortex_2d",
            fps=24,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
