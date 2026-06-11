from fdx import finite_differences_grid as Ω


def test_grid1d_dirichlet_explicit():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        r_width=1,
        bc=Ω.BoundaryCondition.DIRICHLET,
        n_ghost_points=0,
        verbose=False,
    )
    size = 100
    assert grid.n_gps == 0
    assert grid.Dx_central_operator.shape == (size, size)
    assert grid.Dx2_central_operator.shape == (size, size)


def test_grid1d_dirichlet_central():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        bc=Ω.BoundaryCondition.DIRICHLET,
        n_ghost_points=0,
        r_width=1,
        verbose=False,
    )
    size = 100
    assert grid.n_gps == 0
    assert grid.Dx_central_operator.shape == (size, size)
    assert grid.Dx2_central_operator.shape == (size, size)


def test_grid1d_periodic_explicit():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        bc=Ω.BoundaryCondition.PERIODIC,
        n_ghost_points=0,
        r_width=1,
        verbose=False,
    )
    size = 100
    assert grid.n_gps == 0
    assert grid.Dx_central_operator.shape == (size, size)
    assert grid.Dx2_central_operator.shape == (size, size)


def test_grid1d_ghost_points_explicit():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        bc=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points=2,
        r_width=1,
        verbose=False,
    )
    size = 100 + 2 * grid.n_gps
    assert grid.n_gps == 2
    assert grid.Dx_central_operator.shape == (size, size)
    assert grid.Dx2_central_operator.shape == (size, size)


def test_grid1d_ghost_points_upwind():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        bc=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points=2,
        r_width=1,
        verbose=False,
    )
    size = 100 + 2 * grid.n_gps
    assert grid.n_gps == 2
    assert grid.Dx_upwind_operator.shape == (size, size)


def test_grid1d_ghost_points_downwind():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        bc=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points=2,
        r_width=1,
        verbose=False,
    )
    size = 100 + 2 * grid.n_gps
    assert grid.n_gps == 2
    assert grid.Dx_downwind_operator.shape == (size, size)
