from pathlib import Path

import glfw
import numpy as np
import OpenGL.GL as gl
from fieldanimation import FieldAnimation
from fieldanimation.examples.glfwBackend import glfwApp, glInfo
from PIL import Image

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _load_background_image(
    background_image: np.ndarray | str | Path | None,
) -> np.ndarray | None:
    if background_image is None:
        return None
    if isinstance(background_image, (str, Path)):
        return np.flipud(np.asarray(Image.open(background_image)))
    return np.asarray(background_image)


def _read_framebuffer(width: int, height: int) -> np.ndarray:
    """Read RGB pixels from the current OpenGL framebuffer (bottom-left origin)."""
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    return np.flipud(rgb)


def _make_vector_field_viewer(
    field: np.ndarray,
    *,
    width: int,
    height: int,
    title: str,
    background_image: np.ndarray | str | Path | None,
    draw_field: bool,
    speed: float,
    decay: float,
    decay_boost: float,
    opacity: float,
    color: tuple[float, float, float],
    palette: bool,
    point_size: float,
    tracers_count: int,
    periodic: bool,
    bg_color: tuple[float, float, float, float],
    use_fragment_shader: bool,
) -> type[glfwApp]:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("field must have shape (m, n, 2)")

    if glInfo() is None:
        raise RuntimeError("Cannot initialize OpenGL")

    info = glInfo()
    use_compute = (not use_fragment_shader) and info["glversion"] >= 4.3
    image = _load_background_image(background_image)

    class Viewer(glfwApp):
        def __init__(self):
            super().__init__(title, width, height)
            self.bg_color = bg_color
            self._fa = FieldAnimation(
                self.fwidth, self.fheight, field, use_compute, image
            )
            fa = self._fa
            fa.drawField = draw_field
            fa.speedFactor = speed
            fa.decay = decay
            fa.decayBoost = decay_boost
            fa.fadeOpacity = opacity
            fa.color = color
            fa.palette = palette
            fa.pointSize = point_size
            fa.tracersCount = tracers_count
            fa.periodic = periodic
            fa.setField(field)

        def renderScene(self):
            super().renderScene()
            self._fa.draw()

        def onResize(self, window, w, h):
            gl.glViewport(0, 0, w, h)
            self._fa.setSize(w, h)

    return Viewer


def show_vector_field(
    field: np.ndarray,
    *,
    width: int = 800,
    height: int = 600,
    title: str = "Vector field",
    background_image: np.ndarray | str | Path | None = None,
    draw_field: bool = False,
    speed: float = 0.25,
    decay: float = 0.003,
    decay_boost: float = 0.01,
    opacity: float = 0.996,
    color: tuple[float, float, float] = (0.5, 1.0, 1.0),
    palette: bool = True,
    point_size: float = 1.0,
    tracers_count: int = 10_000,
    periodic: bool = True,
    bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    use_fragment_shader: bool = False,
    show_fps: bool = False,
) -> None:
    """Interactive visualization of a 2D vector field.

    Args:
        field: (m, n, 2) float array, last axis = (U, V).
        width, height: framebuffer size in pixels.
        ...: mirror FieldAnimation / app.py GUI knobs.
    """
    viewer_cls = _make_vector_field_viewer(
        field,
        width=width,
        height=height,
        title=title,
        background_image=background_image,
        draw_field=draw_field,
        speed=speed,
        decay=decay,
        decay_boost=decay_boost,
        opacity=opacity,
        color=color,
        palette=palette,
        point_size=point_size,
        tracers_count=tracers_count,
        periodic=periodic,
        bg_color=bg_color,
        use_fragment_shader=use_fragment_shader,
    )
    viewer_cls().run()


def save_animation_frames(
    field: np.ndarray,
    num_frames: int,
    output_dir: str | Path,
    *,
    width: int = 400,
    height: int = 400,
    warmup_frames: int = 60,
    frame_prefix: str = "frame_",
    title: str = "Vector field",
    background_image: np.ndarray | str | Path | None = None,
    draw_field: bool = False,
    speed: float = 0.25,
    decay: float = 0.003,
    decay_boost: float = 0.01,
    opacity: float = 0.996,
    color: tuple[float, float, float] = (0.5, 1.0, 1.0),
    palette: bool = True,
    point_size: float = 1.0,
    tracers_count: int = 10_000,
    periodic: bool = True,
    bg_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    use_fragment_shader: bool = False,
) -> list[Path]:
    """Advance the field animation and write numbered PNG frames for GIF assembly.

    Each frame is one simulation step of the tracer visualization (same renderer
    as :func:`show_vector_field`). Run ``warmup_frames`` steps first so trails
    reach a steady look before capture begins.

    Args:
        field: (m, n, 2) float array, last axis = (U, V).
        num_frames: number of PNG files to write.
        output_dir: directory created if missing; files named ``{prefix}{i:04d}.png``.
        width, height: framebuffer size in pixels.
        warmup_frames: simulation steps before the first saved frame.
        ...: same visualization knobs as :func:`show_vector_field`.

    Returns:
        Paths of the written PNG files, in frame order.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    viewer_cls = _make_vector_field_viewer(
        field,
        width=width,
        height=height,
        title=title,
        background_image=background_image,
        draw_field=draw_field,
        speed=speed,
        decay=decay,
        decay_boost=decay_boost,
        opacity=opacity,
        color=color,
        palette=palette,
        point_size=point_size,
        tracers_count=tracers_count,
        periodic=periodic,
        bg_color=bg_color,
        use_fragment_shader=use_fragment_shader,
    )
    viewer = viewer_cls()
    glfw.hide_window(viewer._window)

    paths: list[Path] = []
    total_steps = warmup_frames + num_frames
    try:
        for step in range(total_steps):
            gl.glClearColor(*viewer.bg_color)
            glfw.poll_events()
            viewer.renderScene()
            glfw.swap_buffers(viewer._window)

            if step < warmup_frames:
                continue

            frame_idx = step - warmup_frames
            path = out / f"{frame_prefix}{frame_idx:04d}.png"
            Image.fromarray(_read_framebuffer(viewer.fwidth, viewer.fheight)).save(path)
            paths.append(path)
    finally:
        viewer.close()

    return paths
