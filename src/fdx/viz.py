from pathlib import Path
from types import SimpleNamespace

import glfw
import numpy as np
import OpenGL.GL as gl
from fieldanimation import FieldAnimation
from fieldanimation.examples import app as fieldanimation_app
from fieldanimation.examples.glfwBackend import createWindow, glInfo
from PIL import Image

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def capture_frame(width, height):
    """
    Capture the current OpenGL framebuffer as a numpy array.

    Args:
        width (int): Window width
        height (int): Window height

    Returns:
        numpy.ndarray: RGB image data (height, width, 3) as uint8
    """
    # Read pixels from the current framebuffer
    pixels = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    # Reshape and flip (OpenGL reads from bottom-left)
    image_array = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    image_array = np.flipud(image_array)  # Flip vertically

    return image_array


def play_vector_field(width, height, numpy_field):
    """Play an interactive animation of a vector field.

    Opens an interactive GLApp window displaying the provided vector field
    with GUI controls for adjusting visualization parameters (speed, decay,
    color, opacity, point size, etc.).

    Args:
        width (int): Window width in pixels
        height (int): Window height in pixels
        numpy_field (np.ndarray): Vector field as an (m, n, 2) shaped array,
            where the last dimension contains [U, V] components of the field

    Example:
        >>> import numpy as np
        >>> from fdx.utils import play_vector_field
        >>> # Create a simple vector field
        >>> m, n = 64, 64
        >>> Y, X = np.mgrid[-3:3:m*1j, -3:3:n*1j]
        >>> U = Y.copy()
        >>> V = -X
        >>> field = np.dstack((U, V))
        >>> play_vector_field(800, 800, field)
    """
    field = np.asarray(numpy_field, dtype=np.float32)
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError(
            f"numpy_field must be an array with shape (m, n, 2), got {field.shape}"
        )
    if not np.isfinite(field).all():
        raise ValueError("numpy_field must contain only finite values")

    options = SimpleNamespace(
        image=None,
        choose=fieldanimation_app.CHOICES[0],
        use_fragment=False,
        draw_field=False,
        fps=False,
        gui=True,
    )

    app = fieldanimation_app.GLApp("Vector Field", width, height, options)
    fieldanimation_app.app = app
    app._fa.setField(field)
    app.setTitle("Vector Field")
    app.run()


def save_animation_frames(
    numpy_field, num_frames, output_dir, width=400, height=400, **field_params
):
    """
    Capture animation frames and save as PNG files.

    Args:
        field_name (str): Name of the vector field to animate
        num_frames (int): Number of frames to capture
        output_dir (str): Directory to save PNG files
        width (int): Window width in pixels
        height (int): Window height in pixels
        **field_params: Additional parameters for FieldAnimation
                        (speed_factor, decay, fade_opacity, etc.)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize GLFW
    if not glfw.init():
        raise SystemExit("Error initializing GLFW")

    # Create an off-screen window
    glfw.window_hint(glfw.VISIBLE, False)
    window = createWindow(width, height, "Frame Capture", visible=False)

    if not window:
        glfw.terminate()
        raise SystemExit("Could not create OpenGL window")

    glfw.make_context_current(window)

    # Check OpenGL version
    gl_info = glInfo()
    if gl_info is None:
        glfw.terminate()
        raise SystemExit("Could not initialize OpenGL")

    print(f"OpenGL {gl_info['glversion']:.1f} - {gl_info['renderer'].decode()}")

    # Create FieldAnimation instance
    use_compute = gl_info["glversion"] >= 4.3
    fa = FieldAnimation(width, height, numpy_field, computeSahder=use_compute)

    # Apply custom parameters if provided
    if "speed_factor" in field_params:
        fa.speedFactor = field_params["speed_factor"]
    if "decay" in field_params:
        fa.decay = field_params["decay"]
    if "fade_opacity" in field_params:
        fa.fadeOpacity = field_params["fade_opacity"]
    if "tracers_count" in field_params:
        fa.tracersCount = field_params["tracers_count"]
    if "point_size" in field_params:
        fa.pointSize = field_params["point_size"]
    if "draw_field" in field_params:
        fa.drawField = field_params["draw_field"]

    print(f"Capturing {num_frames} frames...")
    print(f"Speed Factor: {fa.speedFactor}")
    print(f"Decay: {fa.decay}")
    print(f"Fade Opacity: {fa.fadeOpacity}")
    print(f"Number of Tracers: {fa.tracersCount}")

    # Capture frames
    for frame_num in range(num_frames):
        # Render frame
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        fa.draw()
        gl.glFlush()

        # Capture frame
        frame_data = capture_frame(width, height)

        # Save as PNG
        filename = output_path / f"frame_{frame_num:04d}.png"
        image = Image.fromarray(frame_data, "RGB")
        image.save(filename)

        if (frame_num + 1) % max(
            1, num_frames // 10
        ) == 0 or frame_num == num_frames - 1:
            print(f"  Saved frame {frame_num + 1}/{num_frames}")

    # Cleanup
    glfw.make_context_current(None)
    glfw.destroy_window(window)
    glfw.terminate()

    name = output_dir.replace("frames_", "").title()
    print(f"\n✓ All frames saved to: {output_path.absolute()}")
    print("\nTo create an animated GIF:")
    print(f"  convert -delay 5 -loop 0 {output_path}/*.png {name}.gif")
    print("\nor with ImageMagick's magick command:")
    print(f"  magick -delay 5 -loop 0 {output_path}/*.png {name}.gif")
    print("\nOr with ffmpeg:")
    print(
        f"  ffmpeg -framerate 20 -i {output_path}/frame_%04d.png \
            -c:v libx264 {name}.mp4"
    )
