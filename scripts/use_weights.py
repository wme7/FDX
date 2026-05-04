# %% Run this script to compute the weights for the Explicit and Implicit schemes

import sympy as sb

from fdx.fornberg_weights import fd_explicit_weights
from fdx.taylor_table_weights import fd_central_weights

# Compute weights for Explicit schemes
coefs = fd_explicit_weights(m=0, x=0.5, alpha=[0, 1, 2])
rationals = [sb.Rational(sb.nsimplify(v)) for v in coefs]
sb.pprint(f"α_coefs: {rationals}")

coefs = fd_explicit_weights(m=0, x=0.5, alpha=[-1, 0, 1])
rationals = [sb.Rational(sb.nsimplify(v)) for v in coefs]
sb.pprint(f"α_coefs: {rationals}")

coefs = fd_explicit_weights(m=0, x=0.5, alpha=[-2, -1, 0])
rationals = [sb.Rational(sb.nsimplify(v)) for v in coefs]
sb.pprint(f"α_coefs: {rationals}")

coefs = fd_explicit_weights(m=0, x=0.5, alpha=[-2, -1, 0, 1, 2])
rationals = [sb.Rational(sb.nsimplify(v)) for v in coefs]
sb.pprint(f"α_coefs: {rationals}")

# Compute weights for Implicit schemes
a_coefs, b_coefs = fd_central_weights(
    m=1, alpha=[-1, 0, 1], beta=[i for i in range(-2, 3)]
)
a_rationals = [sb.Rational(sb.nsimplify(v)) for v in a_coefs]
b_rationals = [sb.Rational(sb.nsimplify(v)) for v in b_coefs]
sb.pprint(f"α_coefs: {a_rationals}")
sb.pprint(f"β_coefs: {b_rationals}")

a_coefs, b_coefs = fd_central_weights(m=1, alpha=[0, 1], beta=[0, 1, 2])
a_rationals = [sb.Rational(sb.nsimplify(v)) for v in a_coefs]
b_rationals = [sb.Rational(sb.nsimplify(v)) for v in b_coefs]
sb.pprint(f"α_coefs: {a_rationals}")
sb.pprint(f"β_coefs: {b_rationals}")

a_coefs, b_coefs = fd_central_weights(m=1, alpha=[-1, 0], beta=[-2, -1, 0])
a_rationals = [sb.Rational(sb.nsimplify(v)) for v in a_coefs]
b_rationals = [sb.Rational(sb.nsimplify(v)) for v in b_coefs]
sb.pprint(f"α_coefs: {a_rationals}")
sb.pprint(f"β_coefs: {b_rationals}")

a_coefs, b_coefs = fd_central_weights(m=1, alpha=[-1, 0, 1], beta=[-1, 0, 1])
a_rationals = [sb.Rational(sb.nsimplify(v)) for v in a_coefs]
b_rationals = [sb.Rational(sb.nsimplify(v)) for v in b_coefs]
sb.pprint(f"α_coefs: {a_rationals}")
sb.pprint(f"β_coefs: {b_rationals}")
