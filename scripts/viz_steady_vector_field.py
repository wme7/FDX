"""
Animate a steady vector field with it divergence visualized as a background.

Capture frames with FieldAnimation and save as PNG files.

Use convert, magick or ffmpeg to create an animation from frames:

```bash
magick -delay 5 -loop 0 frames_trigonometric_field/frame_*.png animation.gif
# or
ffmpeg -framerate 24 -i frames_trigonometric_field/frame_%04d.png -y animation.gif
```
"""

import numpy as np

from fdx import finite_differences_grid as Ω
from fdx import viz

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

    # Show vector field
    viz.show_vector_field(
        field=v_field, 
        width=500, 
        height=500,
        tracers_count=1000,
        speed=0.5,
        point_size=3,
        decay=0.001,
    )

    # # Save animation frames
    # paths = viz.save_animation_frames(
    #     field=v_field,
    #     width=500,
    #     height=500,
    #     tracers_count=1000,
    #     speed=0.5,
    #     point_size=3,
    #     decay=0.001,
    #     # ---- Parameters for saving animation frames -----
    #     num_frames=100,
    #     warmup_frames=80,  # tune for fuller trails
    #     output_dir=f"frames_{name}",
    # )
    # print(paths)