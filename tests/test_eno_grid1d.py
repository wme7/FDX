import numpy as np

from fdx import essentially_nonoscillatory_grid as Ω


def _ghost_sine_field(grid: Ω.Grid1d) -> np.ndarray:
    return np.sin(2 * np.pi * grid.nodes)


def test_crweno5_ghost_points_matches_weno5_on_smooth_data():
    """CRWENO5 GHOST_POINTS must agree with WENO5 (same boundary rows/stencils)."""
    grid_w = Ω.Grid1d(
        0.0,
        1.0,
        n=64,
        bc=Ω.BoundaryCondition.GHOST_POINTS,
        scheme=Ω.NonOscillatoryScheme.WENO5,
        verbose=False,
    )
    grid_c = Ω.Grid1d(
        0.0,
        1.0,
        n=64,
        bc=Ω.BoundaryCondition.GHOST_POINTS,
        scheme=Ω.NonOscillatoryScheme.CRWENO5,
        verbose=False,
    )
    u = _ghost_sine_field(grid_w)
    ng = grid_w.n_gps
    phys = slice(ng, -ng)

    du_w_up = grid_w.Dx_upwind(u)
    du_c_up = grid_c.Dx_upwind(u)
    du_w_dn = grid_w.Dx_downwind(u)
    du_c_dn = grid_c.Dx_downwind(u)

    np.testing.assert_allclose(du_c_up[phys], du_w_up[phys], rtol=0, atol=1e-3)
    np.testing.assert_allclose(du_c_dn[phys], du_w_dn[phys], rtol=0, atol=1e-3)
