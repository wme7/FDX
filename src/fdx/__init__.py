import argparse
from importlib.metadata import requires, version

from .eno_weights import fd_eno_weights
from .fornberg_weights import fd_explicit_weights
from .taylor_table_weights import fd_pade_weights
from .utils import RationalFloat

__version__ = version("fdx")
__requires__ = requires("fdx")


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument(
        "-p",
        "--pade",
        action="store_true",
        help="compute Padé-type weights(k_order, alphas, betas)",
    )
    parser.add_argument(
        "-k",
        "--k_order",
        type=int,
        default=1,
        help="build k-order derivative, k = 1, 2, 3, ...",
    )
    parser.add_argument(
        "-a",
        "--alphas",
        type=float,
        nargs="+",
        default=[-1, 0, 1],
        help="α indexes (e.g.: α_i = -1, 0, 1, ...)",
    )
    parser.add_argument(
        "-b",
        "--betas",
        type=float,
        nargs="+",
        default=[-1, 0, 1],
        help="β .indexes (e.g.: β_i = -1, 0, 1, ...)",
    )
    parser.add_argument(
        "-f",
        "--fornberg",
        action="store_true",
        help="compute Fornberg weights(k_order, alphas, x_bar)",
    )
    parser.add_argument(
        "-x",
        "--x_bar",
        type=float,
        default=0.0,
        help="Fornberg weights evaluation point x_bar",
    )
    parser.add_argument(
        "-e",
        "--eno",
        action="store_true",
        help="compute ENO/WENO substencil weights(r_order)",
    )
    parser.add_argument(
        "-r",
        "--r_order",
        type=int,
        default=1,
        help="build r-order ENO/WENO substencils weights",
    )

    # Parse arguments
    args = parser.parse_args()

    # Filter unique and sort alphas and betas
    alphas = sorted(list(set(args.alphas)))
    betas = sorted(list(set(args.betas)))

    if args.fornberg:
        a_coefs = fd_explicit_weights(args.k_order, args.x, alphas)
        str_alphas = [RationalFloat(alpha) for alpha in alphas]
        str_x = RationalFloat(args.x)
        str_a_coefs = [RationalFloat(coef) for coef in a_coefs]
        print(
            f"α [{', '.join(f'{x:r}' for x in str_alphas)}] "
            f"x = {str_x} "
            f"-> weights: [{', '.join(f'{x:r}' for x in str_a_coefs)}]"
        )
    elif args.pade:
        a_coefs, b_coefs = fd_pade_weights(args.k_order, alphas, betas)
        str_alphas = [RationalFloat(alpha) for alpha in alphas]
        str_betas = [RationalFloat(beta) for beta in betas]
        str_a_coefs = [RationalFloat(coef) for coef in a_coefs]
        str_b_coefs = [RationalFloat(coef) for coef in b_coefs]
        print(
            f"α [{', '.join(f'{x:r}' for x in str_alphas)}] "
            f"β [{', '.join(f'{x:r}' for x in str_betas)}] "
            f"-> weights: [{', '.join(f'{x:r}' for x in str_a_coefs)}] "
            f"[{', '.join(f'{x:r}' for x in str_b_coefs)}]"
        )
    elif args.eno:
        s_coefs = fd_eno_weights(args.r_order).tolist()
        for s in range(args.r_order + 1):
            str_s_coefs = [RationalFloat(coef) for coef in s_coefs[s]]
            print(
                f"r = {args.r_order}, s = {s} -> eno/weno weights:"
                f"[{', '.join(f'{x:r}' for x in str_s_coefs)}]"
            )
    else:
        # print version and usage
        print("-" * 80)
        print("Finite-Difference eXtentions (FDX) library")
        print(f" version: {__version__}")
        print("-" * 80)
        parser.print_help()


if __name__ == "__main__":
    main()
