import numpy as np

from fdx import finite_differences_grid as Ω

# Constants
π = np.pi
TOL = 1e-10


def test_operators_identities_periodic_tridiagonal():
    grid = Ω.Grid2d(
        nx=64,
        ny=64,
        bc_x=Ω.BoundaryCondition.PERIODIC,
        bc_y=Ω.BoundaryCondition.PERIODIC,
    )
    # Test function
    u = np.sin(2 * π * grid.X) * np.sin(2 * π * grid.Y)
    assert u.shape == (64, 64)

    # Test Operators
    Du = grid.Grad_pade(u)
    Laplacian = grid.Laplacian_pade(u)
    assert Du[0].shape == (64, 64)
    assert Du[1].shape == (64, 64)
    assert Laplacian.shape == (64, 64)

    errors = {}
    errors["div_grad_eq_laplacian"] = (
        np.linalg.norm(grid.Div_pade(Du) - Laplacian) < TOL
    )
    assert errors["div_grad_eq_laplacian"] == np.False_


def test_operators_identities_periodic_explicit():
    grid = Ω.Grid2d(
        bc_x=Ω.BoundaryCondition.PERIODIC,
        bc_y=Ω.BoundaryCondition.PERIODIC,
    )
    # Test function
    u = np.sin(2 * π * grid.X) * np.sin(2 * π * grid.Y)
    assert u.shape == (100, 100)

    # Test Operators
    Du = grid.Grad_central(u)
    Laplacian = grid.Laplacian_central(u)
    assert Du[0].shape == (100, 100)
    assert Du[1].shape == (100, 100)
    assert Laplacian.shape == (100, 100)

    errors = {}
    errors["div_grad_eq_laplacian"] = (
        np.linalg.norm(grid.Div_central(Du) - Laplacian) < TOL
    )
    assert errors["div_grad_eq_laplacian"] == np.False_


def test_operators_identities_dirichlet_tridiagonal():
    grid = Ω.Grid2d(
        nx=64,
        ny=64,
        bc_x=Ω.BoundaryCondition.DIRICHLET,
        bc_y=Ω.BoundaryCondition.DIRICHLET,
    )
    # Test function
    u = np.sin(2 * π * grid.X) * np.sin(2 * π * grid.Y)
    assert u.shape == (64, 64)

    # Test Operators
    Du = grid.Grad_pade(u)
    Laplacian = grid.Laplacian_pade(u)
    assert Du[0].shape == (64, 64)
    assert Du[1].shape == (64, 64)
    assert Laplacian.shape == (64, 64)

    errors = {}
    errors["div_grad_eq_laplacian"] = (
        np.linalg.norm(grid.Div_pade(Du) - Laplacian) < TOL
    )
    assert errors["div_grad_eq_laplacian"] == np.False_


def test_operators_identities_dirichlet_explicit():
    grid = Ω.Grid2d(
        nx=64,
        ny=64,
        bc_x=Ω.BoundaryCondition.DIRICHLET,
        bc_y=Ω.BoundaryCondition.DIRICHLET,
    )
    # Test function
    u = np.sin(2 * π * grid.X) * np.sin(2 * π * grid.Y)
    assert u.shape == (64, 64)

    # Test Operators
    Du = grid.Grad_central(u)
    Laplacian = grid.Laplacian_central(u)
    assert Du[0].shape == (64, 64)
    assert Du[1].shape == (64, 64)
    assert Laplacian.shape == (64, 64)

    errors = {}
    errors["div_grad_eq_laplacian"] = (
        np.linalg.norm(grid.Div_central(Du) - Laplacian) < TOL
    )
    assert errors["div_grad_eq_laplacian"] == np.False_


def test_operators_identities_ghost_points_explicit():
    grid = Ω.Grid2d(
        bc_x=Ω.BoundaryCondition.GHOST_POINTS,
        bc_y=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points_x=2,
        n_ghost_points_y=2,
        r_width_x=1,
        r_width_y=1,
    )
    # Test function
    u = np.sin(2 * π * grid.X) * np.sin(2 * π * grid.Y)
    assert u.shape == (104, 104)

    # Test Operators
    Du = grid.Grad_central(u)
    Laplacian = grid.Laplacian_central(u)
    assert Du[0].shape == (104, 104)
    assert Du[1].shape == (104, 104)
    assert Laplacian.shape == (104, 104)

    errors = {}
    errors["div_grad_eq_laplacian"] = (
        np.linalg.norm(grid.Div_central(Du) - Laplacian) < TOL
    )
    assert errors["div_grad_eq_laplacian"] == np.False_
