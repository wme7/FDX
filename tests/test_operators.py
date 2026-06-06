import numpy as np
import scipy as sp

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
        scheme=Ω.FiniteDifferenceScheme.PADE,
    )
    # Test function
    X, Y = np.meshgrid(grid.x, grid.y)
    u = np.sin(2 * π * X) * np.sin(2 * π * Y)

    # Test Operators
    Du = grid.Grad(u)
    Laplacian = grid.Laplacian(u)

    errors = {}
    errors["div_grad_eq_laplacian"] = np.linalg.norm(grid.Div(Du) - Laplacian) < TOL
    errors["curl_grad_eq_zero"] = sp.sparse.linalg.norm(grid.curl @ grid.grad) < TOL
    errors["adjoint_consistency"] = sp.sparse.linalg.norm(grid.div + grid.grad.T) < TOL
    assert errors["div_grad_eq_laplacian"] == np.False_
    assert errors["curl_grad_eq_zero"] == np.True_
    assert errors["adjoint_consistency"] == np.True_


def test_operators_identities_periodic_explicit():
    grid = Ω.Grid2d(
        bc_x=Ω.BoundaryCondition.PERIODIC,
        bc_y=Ω.BoundaryCondition.PERIODIC,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
    )
    # Test function
    X, Y = np.meshgrid(grid.x, grid.y)
    u = np.sin(2 * π * X) * np.sin(2 * π * Y)

    # Test Operators
    Du = grid.Grad(u)
    Laplacian = grid.Laplacian(u)

    errors = {}
    errors["div_grad_eq_laplacian"] = np.linalg.norm(grid.Div(Du) - Laplacian) < TOL
    errors["curl_grad_eq_zero"] = sp.sparse.linalg.norm(grid.curl @ grid.grad) < TOL
    errors["adjoint_consistency"] = sp.sparse.linalg.norm(grid.div + grid.grad.T) < TOL
    assert errors["div_grad_eq_laplacian"] == np.False_
    assert errors["curl_grad_eq_zero"] == np.True_
    assert errors["adjoint_consistency"] == np.True_


def test_operators_identities_dirichlet_tridiagonal():
    grid = Ω.Grid2d(
        nx=64,
        ny=64,
        bc_x=Ω.BoundaryCondition.DIRICHLET,
        bc_y=Ω.BoundaryCondition.DIRICHLET,
        scheme=Ω.FiniteDifferenceScheme.PADE,
    )
    # Test function
    X, Y = np.meshgrid(grid.x, grid.y)
    u = np.sin(2 * π * X) * np.sin(2 * π * Y)

    # Test Operators
    Du = grid.Grad(u)
    Laplacian = grid.Laplacian(u)

    errors = {}
    errors["div_grad_eq_laplacian"] = np.linalg.norm(grid.Div(Du) - Laplacian) < TOL
    errors["curl_grad_eq_zero"] = sp.sparse.linalg.norm(grid.curl @ grid.grad) < TOL
    errors["adjoint_consistency"] = sp.sparse.linalg.norm(grid.div + grid.grad.T) < TOL
    assert errors["div_grad_eq_laplacian"] == np.False_
    assert errors["curl_grad_eq_zero"] == np.True_
    assert errors["adjoint_consistency"] == np.False_


def test_operators_identities_dirichlet_explicit():
    grid = Ω.Grid2d(
        nx=64,
        ny=64,
        bc_x=Ω.BoundaryCondition.DIRICHLET,
        bc_y=Ω.BoundaryCondition.DIRICHLET,
        scheme=Ω.FiniteDifferenceScheme.PADE,
    )
    # Test function
    X, Y = np.meshgrid(grid.x, grid.y)
    u = np.sin(2 * π * X) * np.sin(2 * π * Y)

    # Test Operators
    Du = grid.Grad(u)
    Laplacian = grid.Laplacian(u)

    errors = {}
    errors["div_grad_eq_laplacian"] = np.linalg.norm(grid.Div(Du) - Laplacian) < TOL
    errors["curl_grad_eq_zero"] = sp.sparse.linalg.norm(grid.curl @ grid.grad) < TOL
    errors["adjoint_consistency"] = sp.sparse.linalg.norm(grid.div + grid.grad.T) < TOL
    assert errors["div_grad_eq_laplacian"] == np.False_
    assert errors["curl_grad_eq_zero"] == np.True_
    assert errors["adjoint_consistency"] == np.False_


def test_operators_identities_ghost_points_explicit():
    grid = Ω.Grid2d(
        bc_x=Ω.BoundaryCondition.GHOST_POINTS,
        bc_y=Ω.BoundaryCondition.GHOST_POINTS,
        n_ghost_points_x=2,
        n_ghost_points_y=2,
        scheme=Ω.FiniteDifferenceScheme.CENTRAL,
        r_width_x=1,
        r_width_y=1,
    )
    # Test function
    X, Y = np.meshgrid(grid.x, grid.y)
    u = np.sin(2 * π * X) * np.sin(2 * π * Y)

    # Test Operators
    Du = grid.Grad(u)
    Laplacian = grid.Laplacian(u)

    errors = {}
    errors["div_grad_eq_laplacian"] = np.linalg.norm(grid.Div(Du) - Laplacian) < TOL
    errors["curl_grad_eq_zero"] = sp.sparse.linalg.norm(grid.curl @ grid.grad) < TOL
    errors["adjoint_consistency"] = sp.sparse.linalg.norm(grid.div + grid.grad.T) < TOL
    assert errors["div_grad_eq_laplacian"] == np.False_
    assert errors["curl_grad_eq_zero"] == np.True_
    assert errors["adjoint_consistency"] == np.False_
