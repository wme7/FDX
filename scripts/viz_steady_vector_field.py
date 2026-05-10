"""
Animate a steady vector field with it divergence visualized as a background.

Capture frames with FieldAnimation and save as PNG files.

Use convert, magick or ffmpeg to create an animation.
"""

import numpy as np

from fdx import finite_differences_grid as Ω
from fdx.utils import save_animation_frames

if __name__ == "__main__":
    # Create a 2D grid and compute the vector field and its divergence
    GRID_N = 100
    GRID_L = 5.0
    FD = Ω.FiniteDifferenceScheme.COMPACT
    grid = Ω.Grid2d(
        xa=-GRID_L,
        xb=GRID_L,
        nx=GRID_N,
        ya=-GRID_L,
        yb=GRID_L,
        ny=GRID_N,
        scheme=FD,
    )
    x, y = np.meshgrid(grid.x, grid.y)
    vx, vy, name = np.sin(x * y), np.cos(x - y), "trigonometric_field"
    # vx, vy, name = x, y, "positive_div"
    # vx, vy, name = -x, -y, "negative_div"
    # vx, vy, name = -y, x, "negative_curl"
    # vx, vy, name = y, -x, "positive_curl"
    v_field = np.stack((vx, vy), axis=-1)
    assert v_field.shape == (GRID_N, GRID_N, 2)

    # Save animation frames
    save_animation_frames(
        numpy_field=v_field,
        num_frames=100,
        output_dir=f"frames_{name}",
        width=500,
        height=500,
        speed_factor=2.0,
        decay=0.003,
        fade_opacity=0.996,
        tracers_count=1000,
        point_size=2.0,
        draw_field=False,
    )
