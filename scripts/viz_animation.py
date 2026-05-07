"""
Animated divergence field visualization.

Computes div(v) for:
    vx = sin(xy + t)
    vy = cos(x - y - t)

Saves field data to HDF5, then animates with pcolormesh + quiver overlay.
"""

import os

import h5py as h5
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fdx import finite_differences_grid as Ω

# ── Typography ────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "text.usetex": True,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)

π = np.pi

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "./figures/animation"
OUTPUT_FILE = "fields.h5"
T_FINAL = 2 * π
N_FRAMES = 48
GRID_N = 500  # grid points per axis
GRID_L = 5.0  # domain half-length
QUIVER_STEP = 30  # subsample stride for quiver arrows
FPS = 30


# ── Field ─────────────────────────────────────────────────────────────────────
def vector_field(
    x: np.ndarray, y: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vx, vy) at time t on the spatial grid (x, y)."""
    return np.sin(x * y + t), np.cos(x - y - t)


# ── I/O ───────────────────────────────────────────────────────────────────────
def compute_and_save(path: str, t_final: float, frames: int) -> None:
    """Evaluate vx, vy, div(v) on a time sequence and write to HDF5."""
    t_frames = np.linspace(0, t_final, frames, endpoint=True)

    FD = Ω.FiniteDifferenceScheme.COMPACT
    grid = Ω.Grid2d(
        xa=-GRID_L, xb=GRID_L, nx=GRID_N, ya=-GRID_L, yb=GRID_L, ny=GRID_N, scheme=FD
    )
    x, y = np.meshgrid(grid.x, grid.y)

    shape = (GRID_N, GRID_N)
    ds_opts = dict(
        dtype=np.float32, chunks=(1, *shape), compression="gzip", shuffle=True
    )

    with h5.File(path, "w") as f:
        f.attrs.update(
            frames=frames, nx=GRID_N, ny=GRID_N, lx=GRID_L, ly=GRID_L, scheme=str(FD)
        )

        grp = f.create_group("grid")
        grp.create_dataset("x", data=grid.x, compression="gzip")
        grp.create_dataset("y", data=grid.y, compression="gzip")
        f.create_dataset("t", data=t_frames, compression="gzip")

        # Pre-allocate resizable datasets
        div_ds = f.create_dataset(
            "div", shape=(0, *shape), maxshape=(None, *shape), **ds_opts
        )
        vx_ds = f.create_dataset(
            "vx", shape=(0, *shape), maxshape=(None, *shape), **ds_opts
        )
        vy_ds = f.create_dataset(
            "vy", shape=(0, *shape), maxshape=(None, *shape), **ds_opts
        )

        for i, t in enumerate(t_frames):
            print(f"  frame {i + 1:>3}/{frames}")
            vx, vy = vector_field(x, y, float(t))
            div = grid.Div([vx, vy]).astype(np.float32, copy=False)

            for ds, data in ((div_ds, div), (vx_ds, vx), (vy_ds, vy)):
                ds.resize((i + 1, *shape))
                ds[i] = data

    print(f"Saved → {path}")


def load_fields(path: str) -> dict:
    """Load all field arrays from HDF5 into memory."""
    with h5.File(path, "r") as f:
        return {
            "t": f["t"][:],
            "div": f["div"][:],
            "vx": f["vx"][:],
            "vy": f["vy"][:],
            "x": f["grid/x"][:],
            "y": f["grid/y"][:],
        }


# ── Plotting ──────────────────────────────────────────────────────────────────
def build_figure(fields: dict, i0: int = 0):
    """
    Construct the initial figure; return (fig, artists) for later animation.
    artists = (background, quiver, frame_text)
    """
    t, div, vx, vy, x, y = (fields[k] for k in ("t", "div", "vx", "vy", "x", "y"))

    s = QUIVER_STEP
    xq = x[::s]
    yq = y[::s]

    vmin = div[i0].min()
    vmax = div[i0].max()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Divergence heatmap
    bg = ax.pcolormesh(x, y, div[i0], cmap="jet", vmin=vmin, vmax=vmax)

    # Vector arrows
    qv = ax.quiver(xq, yq, vx[i0, ::s, ::s], vy[i0, ::s, ::s], pivot="middle")

    # Colorbar
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.05)
    fig.colorbar(bg, cax=cax).set_label("Divergence")

    # Labels & annotations
    ax.set(xlabel="x", ylabel="y", aspect="equal")
    fig.suptitle(r"\textbf{Unsteady Vector Field}", fontsize=20, y=0.97)
    fig.text(
        0.5,
        0.93,
        r"$\vec{v}(x,y,t) \equiv (v_x, v_y) = (\sin(xy + t), \cos(x - y - t))$",
        ha="center",
        va="top",
        fontsize=15,
    )
    time_label = fig.text(
        0.99, 0.98, rf"$t = {t[i0]:.2f}$", ha="right", va="center", fontsize=12
    )

    return fig, ax, (bg, qv, time_label), (vmin, vmax)


def make_update(fields: dict, artists, clim):
    """Return a FuncAnimation-compatible update callable (closure)."""
    t, div, vx, vy = (fields[k] for k in ("t", "div", "vx", "vy"))
    bg, qv, time_label = artists
    vmin, vmax = clim
    s = QUIVER_STEP

    def update(frame: int):
        bg.set_array(div[frame])
        bg.set_clim(vmin, vmax)
        qv.set_UVC(vx[frame, ::s, ::s], vy[frame, ::s, ::s])
        time_label.set_text(rf"$t = {t[frame]:.2f}$")
        return bg, qv

    return update


def save_frames(fig: plt.Figure, anim: animation.FuncAnimation, out_dir: str) -> None:
    """Render each animation frame to a numbered PNG."""
    for i, frame in enumerate(anim.new_frame_seq()):
        anim._draw_frame(frame)
        fig.savefig(
            os.path.join(out_dir, f"frame_{i:04d}.png"),
            dpi=fig.dpi,
            bbox_inches="tight",
        )
    print(f"Frames saved → {out_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h5_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    compute_and_save(h5_path, t_final=T_FINAL, frames=N_FRAMES)

    fields = load_fields(h5_path)
    fig, ax, artists, clim = build_figure(fields)

    anim = animation.FuncAnimation(
        fig,
        make_update(fields, artists, clim),
        frames=N_FRAMES,
        blit=False,
        interval=1000 / FPS,
        repeat=True,
    )
    plt.show()

    # Save frames as PNG
    # save_frames(fig, anim, OUTPUT_DIR)

    # Save as GIF
    anim.save(
        os.path.join(OUTPUT_DIR, "animation.gif"), writer="pillow", fps=12, dpi=100
    )

    # Close figure
    plt.close()
