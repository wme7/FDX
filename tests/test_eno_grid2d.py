import numpy as np

from fdx import essentially_nonoscillatory_grid as Ω


def test_grid2d_threaded_derivatives_match_serial():
    nx, ny = 64, 64
    g0 = Ω.Grid2d(nx=nx, ny=ny)
    u = np.sin(g0.X) * np.cos(g0.Y)

    g_serial = Ω.Grid2d(nx=nx, ny=ny, workers=None)
    g_parallel = Ω.Grid2d(nx=nx, ny=ny, workers=4)

    for method in ("Dx_upwind", "Dx_downwind", "Dy_upwind", "Dy_downwind"):
        fn_serial = getattr(g_serial, method)
        fn_parallel = getattr(g_parallel, method)
        u_serial = fn_serial(u)
        u_parallel = fn_parallel(u)
        np.testing.assert_allclose(u_parallel, u_serial, rtol=0, atol=0)
