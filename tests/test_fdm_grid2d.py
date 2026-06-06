from fdx import finite_differences_grid as Ω


def test_grid2d_dirichlet_central():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        ya=0.0,
        yb=1.0,
        ny=100,
        bc_x=Ω.BoundaryCondition.DIRICHLET,
        bc_y=Ω.BoundaryCondition.DIRICHLET,
        n_ghost_points_x=0,
        n_ghost_points_y=0,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = 100 * 100
    assert grid.n_gps_x == 0
    assert grid.n_gps_y == 0
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_dirichlet_explicit():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        ya=0.0,
        yb=1.0,
        ny=100,
        bc_x=Ω.BoundaryCondition.DIRICHLET,
        bc_y=Ω.BoundaryCondition.DIRICHLET,
        n_ghost_points_x=0,
        n_ghost_points_y=0,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = 100 * 100
    assert grid.n_gps_x == 0
    assert grid.n_gps_y == 0
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_periodic_explicit():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        ya=0.0,
        yb=1.0,
        ny=100,
        bc_x=Ω.BoundaryCondition.PERIODIC,
        bc_y=Ω.BoundaryCondition.PERIODIC,
        n_ghost_points_x=0,
        n_ghost_points_y=0,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = 100 * 100
    assert grid.n_gps_x == 0
    assert grid.n_gps_y == 0
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_ghost_points_explicit():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        ya=0.0,
        yb=1.0,
        ny=100,
        bc_x=Ω.BoundaryCondition.GHOST_POINTS,
        bc_y=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points_x=2,
        n_ghost_points_y=2,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = (100 + 2 * grid.n_gps_x) * (100 + 2 * grid.n_gps_y)
    assert grid.n_gps_x == 2
    assert grid.n_gps_y == 2
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_ghost_points_upwind():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        ya=0.0,
        yb=1.0,
        ny=100,
        bc_x=Ω.BoundaryCondition.GHOST_POINTS,
        bc_y=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points_x=2,
        n_ghost_points_y=2,
        scheme=Ω.FiniteDifferenceScheme.UPWIND,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = (100 + 2 * grid.n_gps_x) * (100 + 2 * grid.n_gps_y)
    assert grid.n_gps_x == 2
    assert grid.n_gps_y == 2
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)


def test_grid2d_ghost_points_downwind():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=100,
        ya=0.0,
        yb=1.0,
        ny=100,
        bc_x=Ω.BoundaryCondition.GHOST_POINTS,
        bc_y=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points_x=2,
        n_ghost_points_y=2,
        scheme=Ω.FiniteDifferenceScheme.DOWNWIND,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = (100 + 2 * grid.n_gps_x) * (100 + 2 * grid.n_gps_y)
    assert grid.n_gps_x == 2
    assert grid.n_gps_y == 2
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)


def test_grid2d_dirichlet_tridiagonal():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=10,
        ya=0.0,
        yb=1.0,
        ny=10,
        bc_x=Ω.BoundaryCondition.DIRICHLET,
        bc_y=Ω.BoundaryCondition.DIRICHLET,
        n_ghost_points_x=0,
        n_ghost_points_y=0,
        scheme=Ω.FiniteDifferenceScheme.PADE,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = 10 * 10
    assert grid.n_gps_x == 0
    assert grid.n_gps_y == 0
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)


def test_grid2d_periodic_tridiagonal():
    grid = Ω.Grid2d(
        xa=0.0,
        xb=1.0,
        nx=10,
        ya=0.0,
        yb=1.0,
        ny=10,
        bc_x=Ω.BoundaryCondition.PERIODIC,
        bc_y=Ω.BoundaryCondition.PERIODIC,
        n_ghost_points_x=0,
        n_ghost_points_y=0,
        scheme=Ω.FiniteDifferenceScheme.PADE,
        r_width_x=1,
        r_width_y=1,
        verbose=False,
    )
    size = 10 * 10
    assert grid.n_gps_x == 0
    assert grid.n_gps_y == 0
    assert grid.Dx_operator.shape == (size, size)
    assert grid.Dy_operator.shape == (size, size)
    assert grid.Dxy_operator.shape == (size, size)
    assert grid.Dyx_operator.shape == (size, size)
    assert grid.grad.shape == (2 * size, size)
    assert grid.div.shape == (size, 2 * size)
    assert grid.curl.shape == (size, 2 * size)
    assert grid.laplacian.shape == (size, size)
