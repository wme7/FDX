from fdx import finite_differences_grid as Ω


def test_grid1d_dirichlet_explicit():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        r=1,
        bc=Ω.BoundaryCondition.DIRICHLET,
        scheme=Ω.FiniteDifferenceScheme.EXPLICIT,
        verbose=False,
    )
    size = 100
    assert grid.Dx.shape == (size, size)
    assert grid.Dx2.shape == (size, size)


def test_grid1d_periodic_explicit():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        r=1,
        bc=Ω.BoundaryCondition.PERIODIC,
        scheme=Ω.FiniteDifferenceScheme.EXPLICIT,
        verbose=False,
    )
    size = 100
    assert grid.Dx.shape == (size, size)
    assert grid.Dx2.shape == (size, size)


def test_grid1d_ghost_points_explicit():
    grid = Ω.Grid1d(
        a=0.0,
        b=1.0,
        n=100,
        r=1,
        bc=Ω.BoundaryCondition.GHOST_POINTS,
        scheme=Ω.FiniteDifferenceScheme.EXPLICIT,
        verbose=False,
    )
    r_width = 1
    size = 100 + 2 * r_width
    assert grid.Dx.shape == (size, size)
    assert grid.Dx2.shape == (size, size)
