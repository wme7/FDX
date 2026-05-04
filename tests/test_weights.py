import numpy as np

from fdx.fornberg_weights import fd_explicit_weights
from fdx.taylor_table_weights import fd_central_weights


# -----------------------------------------------------------------------
# Interpolation schemes
# -----------------------------------------------------------------------
def test_Fornberg_central_schemes_interpolation():
    coefs = fd_explicit_weights(m=0, x=-0.5, alpha=[-1, 0, 1])
    assert len(coefs) == 3
    assert np.allclose(coefs, [0.375, 0.75, -0.125])
    coefs = fd_explicit_weights(m=0, x=-0.5, alpha=[-2, -1, 0, 1, 2])
    assert len(coefs) == 5
    assert np.allclose(coefs, [-5 / 128, 15 / 32, 45 / 64, -5 / 32, 3 / 128])


# -----------------------------------------------------------------------
# Central difference schemes through Fornberg's weights
# -----------------------------------------------------------------------
def test_Fornberg_central_schemes_1st_derivative():
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[-1, 0, 1])
    assert len(coefs) == 3
    assert np.allclose(coefs, [-1 / 2, 0, 1 / 2])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[-2, -1, 0, 1, 2])
    assert len(coefs) == 5
    assert np.allclose(coefs, [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[-3, -2, -1, 0, 1, 2, 3])
    assert len(coefs) == 7
    assert np.allclose(coefs, [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    assert len(coefs) == 9
    assert np.allclose(
        coefs, [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
    )


def test_Fornberg_central_schemes_2nd_derivative():
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[-1, 0, 1])
    assert len(coefs) == 3
    assert np.allclose(coefs, [1, -2, 1])
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[-2, -1, 0, 1, 2])
    assert len(coefs) == 5
    assert np.allclose(coefs, [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[-3, -2, -1, 0, 1, 2, 3])
    assert len(coefs) == 7
    assert np.allclose(
        coefs, [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
    )
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    assert len(coefs) == 9
    assert np.allclose(
        coefs,
        [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
    )


def test_Fornberg_biased_schemes_1st_derivative():
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[0, 1])
    assert len(coefs) == 2
    assert np.allclose(coefs, [-1, 1])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[0, 1, 2])
    assert len(coefs) == 3
    assert np.allclose(coefs, [-3 / 2, 2, -1 / 2])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[0, 1, 2, 3])
    assert len(coefs) == 4
    assert np.allclose(coefs, [-11 / 6, 3, -3 / 2, 1 / 3])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[0, 1, 2, 3, 4])
    assert len(coefs) == 5
    assert np.allclose(coefs, [-25 / 12, 4, -3, 4 / 3, -1 / 4])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[0, 1, 2, 3, 4, 5])
    assert len(coefs) == 6
    assert np.allclose(coefs, [-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[0, 1, 2, 3, 4, 5, 6])
    assert len(coefs) == 7
    assert np.allclose(coefs, [-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6])


def test_Fornberg_biased_schemes_2nd_derivative():
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[0, 1, 2])
    assert len(coefs) == 3
    assert np.allclose(coefs, [1, -2, 1])
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[0, 1, 2, 3])
    assert len(coefs) == 4
    assert np.allclose(coefs, [2, -5, 4, -1])
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[0, 1, 2, 3, 4])
    assert len(coefs) == 5
    assert np.allclose(coefs, [35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12])
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[0, 1, 2, 3, 4, 5])
    assert len(coefs) == 6
    assert np.allclose(coefs, [15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6])
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[0, 1, 2, 3, 4, 5, 6])
    assert len(coefs) == 7
    assert np.allclose(
        coefs, [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180]
    )
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[0, 1, 2, 3, 4, 5, 6, 7])
    assert len(coefs) == 8
    assert np.allclose(
        coefs,
        [469 / 90, -223 / 10, 879 / 20, -949 / 18, 41, -201 / 10, 1019 / 180, -7 / 10],
    )


# -----------------------------------------------------------------------
# Central difference schemes through Taylor table
# -----------------------------------------------------------------------
def test_Taylor_table_central_schemes_1st_derivative():
    alpha_coefs, beta_coefs = fd_central_weights(m=1, alpha=[0], beta=[-1, 0, 1])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 3
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-1 / 2, 0, 1 / 2])
    alpha_coefs, beta_coefs = fd_central_weights(m=1, alpha=[0], beta=[-2, -1, 0, 1, 2])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 5
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=1, alpha=[0], beta=[-3, -2, -1, 0, 1, 2, 3]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 7
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=1, alpha=[0], beta=[-4, -3, -2, -1, 0, 1, 2, 3, 4]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 9
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(
        beta_coefs,
        [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280],
    )


def test_Taylor_table_central_schemes_2nd_derivative():
    alpha_coefs, beta_coefs = fd_central_weights(m=2, alpha=[0], beta=[-1, 0, 1])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 3
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [1, -2, 1])
    alpha_coefs, beta_coefs = fd_central_weights(m=2, alpha=[0], beta=[-2, -1, 0, 1, 2])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 5
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=2, alpha=[0], beta=[-3, -2, -1, 0, 1, 2, 3]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 7
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(
        beta_coefs, [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
    )
    alpha_coefs, beta_coefs = fd_central_weights(
        m=2, alpha=[0], beta=[-4, -3, -2, -1, 0, 1, 2, 3, 4]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 9
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(
        beta_coefs,
        [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
    )


def test_Taylor_table_biased_schemes_1st_derivative():
    alpha_coefs, beta_coefs = fd_central_weights(m=1, alpha=[0], beta=[0, 1])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 2
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-1, 1])
    alpha_coefs, beta_coefs = fd_central_weights(m=1, alpha=[0], beta=[0, 1, 2])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 3
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-3 / 2, 2, -1 / 2])
    alpha_coefs, beta_coefs = fd_central_weights(m=1, alpha=[0], beta=[0, 1, 2, 3])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 4
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-11 / 6, 3, -3 / 2, 1 / 3])
    alpha_coefs, beta_coefs = fd_central_weights(m=1, alpha=[0], beta=[0, 1, 2, 3, 4])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 5
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-25 / 12, 4, -3, 4 / 3, -1 / 4])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=1, alpha=[0], beta=[0, 1, 2, 3, 4, 5]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 6
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=1, alpha=[0], beta=[0, 1, 2, 3, 4, 5, 6]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 7
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(
        beta_coefs, [-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]
    )


def test_Taylor_table_biased_schemes_2nd_derivative():
    alpha_coefs, beta_coefs = fd_central_weights(m=2, alpha=[0], beta=[0, 1, 2])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 3
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [1, -2, 1])
    alpha_coefs, beta_coefs = fd_central_weights(m=2, alpha=[0], beta=[0, 1, 2, 3])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 4
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [2, -5, 4, -1])
    alpha_coefs, beta_coefs = fd_central_weights(m=2, alpha=[0], beta=[0, 1, 2, 3, 4])
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 5
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=2, alpha=[0], beta=[0, 1, 2, 3, 4, 5]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 6
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(beta_coefs, [15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=2, alpha=[0], beta=[0, 1, 2, 3, 4, 5, 6]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 7
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(
        beta_coefs, [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180]
    )
    alpha_coefs, beta_coefs = fd_central_weights(
        m=2, alpha=[0], beta=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    assert len(alpha_coefs) == 1
    assert len(beta_coefs) == 8
    assert np.allclose(alpha_coefs, [1])
    assert np.allclose(
        beta_coefs,
        [469 / 90, -223 / 10, 879 / 20, -949 / 18, 41, -201 / 10, 1019 / 180, -7 / 10],
    )


# -----------------------------------------------------------------------
# Examples from Fornberg 1998, From Table 1
# -----------------------------------------------------------------------
def test_Fornberg_1998_Table_1_irregular_grids():
    # 1. 2nd derivative centered regular grid (α = [-2, -1, 0, 1, 2])
    coefs = fd_explicit_weights(m=2, x=0.0, alpha=[-2, -1, 0, 1, 2])
    assert len(coefs) == 5
    assert np.allclose(coefs, [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
    # 2. Staggered regular grid (α = [-3/2, -1/2, 1/2, 3/2])
    coefs = fd_explicit_weights(m=1, x=0.0, alpha=[-3 / 2, -1 / 2, 1 / 2, 3 / 2])
    assert len(coefs) == 4
    assert np.allclose(coefs, [1 / 24, -9 / 8, 9 / 8, -1 / 24])
    # 3. One-sided regular grid (α = [-3, -2, -1, 0, 1])
    coefs = fd_explicit_weights(m=1, x=1.0, alpha=[-3, -2, -1, 0, 1])
    assert len(coefs) == 5
    assert np.allclose(coefs, [1 / 4, -4 / 3, 3, -4, 25 / 12])
    # 4. 3rd derivative partly one-sided irregular grid
    coefs = fd_explicit_weights(m=3, x=0.5, alpha=[0, 1 / 3, 1, 2, 7 / 2, 6])
    assert len(coefs) == 6
    assert np.allclose(
        coefs, [-195 / 14, 26.1808050, -408 / 25, 89 / 20, -0.394586466, 21 / 1700]
    )  # Errata in Fornberg 1998
    # 5. Standard compact scheme (α = [-1, 0, 1], β = [-1, 0, 1])
    alpha_coefs, beta_coefs = fd_central_weights(m=1, alpha=[-1, 0, 1], beta=[-1, 0, 1])
    assert len(alpha_coefs) == 3
    assert len(beta_coefs) == 3
    assert np.allclose(alpha_coefs, [1 / 4, 1, 1 / 4])
    assert np.allclose(beta_coefs, [-3 / 4, 0, 3 / 4])
    # 6. Adams-Bashforth scheme (α = [-3, -2, -1, 0], β = [0, 1])
    alpha_coefs, beta_coefs = fd_central_weights(
        m=1, alpha=[-3, -2, -1, 0], beta=[0, 1]
    )
    assert len(alpha_coefs) == 4
    assert len(beta_coefs) == 2
    assert np.allclose(alpha_coefs, [-9 / 55, 37 / 55, -59 / 55, 1])
    assert np.allclose(beta_coefs, [-24 / 55, 24 / 55])


# -----------------------------------------------------------------------
# Examples from Fornberg 1998, From Table 4
# -----------------------------------------------------------------------
def test_Fornberg_1998_Table_4_2nd_order_compact_scheme():
    # 2nd order compact scheme (α = [-1, 0, 1], β = [-1, 0, 1])
    alpha_coefs, beta_coefs = fd_central_weights(m=2, alpha=[-1, 0, 1], beta=[-1, 0, 1])
    assert len(alpha_coefs) == 3
    assert len(beta_coefs) == 3
    assert np.allclose(alpha_coefs, [1 / 10, 1, 1 / 10])
    assert np.allclose(beta_coefs, [12 / 10, -24 / 10, 12 / 10])


# -----------------------------------------------------------------------
# WENO interpolation stencisl and sub-stencil weights from http://dx.doi.org/10.4249/scholarpedia.9709
# -----------------------------------------------------------------------
def test_WENO_interpolation_stencils_to_right_face():
    coefs = fd_explicit_weights(m=0, x=0.5, alpha=[-2, -1, 0])
    assert len(coefs) == 3
    assert np.allclose(coefs, [3 / 8, -5 / 4, 15 / 8])
    coefs = fd_explicit_weights(m=0, x=0.5, alpha=[-1, 0, 1])
    assert len(coefs) == 3
    assert np.allclose(coefs, [-1 / 8, 3 / 4, 3 / 8])
    coefs = fd_explicit_weights(m=0, x=0.5, alpha=[0, 1, 2])
    assert len(coefs) == 3
    assert np.allclose(coefs, [3 / 8, 3 / 4, -1 / 8])
    coefs = fd_explicit_weights(m=0, x=0.5, alpha=[-2, -1, 0, 1, 2])
    assert len(coefs) == 5
    assert np.allclose(coefs, [3 / 128, -5 / 32, 45 / 64, 15 / 32, -5 / 128])


def test_WENO_interpolation_stencils_to_left_face():
    coefs = fd_explicit_weights(m=0, x=-0.5, alpha=[-2, -1, 0])
    assert len(coefs) == 3
    assert np.allclose(coefs, [-1 / 8, 3 / 4, 3 / 8])
    coefs = fd_explicit_weights(m=0, x=-0.5, alpha=[-1, 0, 1])
    assert len(coefs) == 3
    assert np.allclose(coefs, [3 / 8, 3 / 4, -1 / 8])
    coefs = fd_explicit_weights(m=0, x=-0.5, alpha=[0, 1, 2])
    assert len(coefs) == 3
    assert np.allclose(coefs, [15 / 8, -5 / 4, 3 / 8])
    coefs = fd_explicit_weights(m=0, x=-0.5, alpha=[-2, -1, 0, 1, 2])
    assert len(coefs) == 5
    assert np.allclose(coefs, [-5 / 128, 15 / 32, 45 / 64, -5 / 32, 3 / 128])
