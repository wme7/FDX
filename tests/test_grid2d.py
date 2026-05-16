from fdx import finite_differences_grid as Ω


def test_grid2d_dirichlet_central():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=100,
        ry=1,
        bcx=Ω.BoundaryCondition.DIRICHLET,
        bcy=Ω.BoundaryCondition.DIRICHLET,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        verbose=False,
    )
    size = 100 * 100
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_dirichlet_explicit():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=100,
        ry=1,
        bcx=Ω.BoundaryCondition.DIRICHLET,
        bcy=Ω.BoundaryCondition.DIRICHLET,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        verbose=False,
    )
    size = 100 * 100
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_periodic_explicit():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=100,
        ry=1,
        bcx=Ω.BoundaryCondition.PERIODIC,
        bcy=Ω.BoundaryCondition.PERIODIC,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        verbose=False,
    )
    size = 100 * 100
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_ghost_points_explicit():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=100,
        ry=1,
        bcx=Ω.BoundaryCondition.GHOST_POINTS,
        bcy=Ω.BoundaryCondition.GHOST_POINTS,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        verbose=False,
    )
    r_width = 1
    size = (100 + 2 * r_width) * (100 + 2 * r_width)
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_ghost_points_upwind():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=100,
        ry=1,
        bcx=Ω.BoundaryCondition.GHOST_POINTS,
        bcy=Ω.BoundaryCondition.GHOST_POINTS,
        scheme=Ω.FiniteDifferenceScheme.UPWIND,
        verbose=False,
    )
    r_width = 1
    size = (100 + r_width) * (100 + r_width)
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)


def test_grid2d_ghost_points_downwind():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=100,
        ry=1,
        bcx=Ω.BoundaryCondition.GHOST_POINTS,
        bcy=Ω.BoundaryCondition.GHOST_POINTS,
        scheme=Ω.FiniteDifferenceScheme.DOWNWIND,
        verbose=False,
    )
    r_width = 1
    size = (100 + r_width) * (100 + r_width)
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)


def test_grid2d_dirichlet_tridiagonal():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=10,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=10,
        ry=1,
        bcx=Ω.BoundaryCondition.DIRICHLET,
        bcy=Ω.BoundaryCondition.DIRICHLET,
        scheme=Ω.FiniteDifferenceScheme.PADE,
        verbose=False,
    )
    size = 10 * 10
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_periodic_tridiagonal():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=10,
        rx=1,
        ya=0.0,
        yb=1.0,
        ny=10,
        ry=1,
        bcx=Ω.BoundaryCondition.PERIODIC,
        bcy=Ω.BoundaryCondition.PERIODIC,
        scheme=Ω.FiniteDifferenceScheme.PADE,
        verbose=False,
    )
    size = 10 * 10
    assert grid.Dx.shape == (size, size)
    assert grid.Dy.shape == (size, size)
    assert grid.Dxy.shape == (size, size)
    assert grid.Dyx.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)
