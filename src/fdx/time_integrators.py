"""
Runge-Kutta time integrators: RK1 (Euler) through RK4.

Each scheme is a callable that advances a state vector by one timestep,
accepting an arbitrary RHS function and optional extra arguments — matching
the ergonomics of the standalone rk4() helper you already have.
"""

from __future__ import annotations

from typing import Callable

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RungeKutta:
    """Base class for explicit Runge-Kutta integrators.

    Subclasses implement `_step`, which advances *state* by *dt* given a
    right-hand-side callable `rhs_fn(state, *args) -> same shape as state`.

    Usage
    -----
    >>> rk = RungeKutta.from_order(4)       # factory: order 1-4
    >>> rk = RK4()                          # or directly
    >>> u_next = rk(rhs_fn, u, dt, *args)   # __call__ mirrors your rk4()
    >>> u_next = rk.step(rhs_fn, u, dt)     # same thing, no extra args
    """

    #: Human-readable label used in __repr__ and match/case dispatch
    name: str = "RungeKutta"

    def __call__(self, rhs_fn: Callable, state, dt: float, *args):
        return self._step(rhs_fn, state, dt, *args)

    def step(self, rhs_fn: Callable, state, dt: float, *args):
        """Explicit alias for __call__ — useful when passing the integrator
        as an argument so the call site reads more clearly."""
        return self._step(rhs_fn, state, dt, *args)

    def _step(self, rhs_fn: Callable, state, dt: float, *args):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_order(order: int) -> "RungeKutta":
        """Return the canonical RK scheme for *order* ∈ {1, 2, 3, 4}."""
        schemes = {1: RK1, 2: RK2, 3: RK3, 4: RK4}
        if order not in schemes:
            raise ValueError(f"order must be 1–4, got {order}")
        return schemes[order]()

    @staticmethod
    def from_name(name: str) -> "RungeKutta":
        """Return a scheme by string name: 'Euler'/'RK1', 'RK2', 'RK3', 'RK4'."""
        mapping = {
            "euler": RK1,
            "rk1": RK1,
            "rk2": RK2,
            "rk3": RK3,
            "rk4": RK4,
        }
        key = name.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unknown scheme '{name}'. Choose from: Euler, RK1–RK4.")
        return mapping[key]()


# ---------------------------------------------------------------------------
# Concrete schemes
# ---------------------------------------------------------------------------


class RK1(RungeKutta):
    """Forward Euler — first-order accurate."""

    name = "Euler"

    def _step(self, rhs_fn, state, dt, *args):
        return state + dt * rhs_fn(state, *args)


class RK2(RungeKutta):
    """Explicit trapezoidal (Heun) — second-order accurate."""

    name = "RK2"

    def _step(self, rhs_fn, state, dt, *args):
        k1 = rhs_fn(state, *args)
        k2 = rhs_fn(state + dt * k1, *args)
        return state + 0.5 * dt * (k1 + k2)


class RK3(RungeKutta):
    """Shu-Osher SSP RK3 — third-order, strong-stability-preserving."""

    name = "RK3"

    def _step(self, rhs_fn, state, dt, *args):
        u0 = state + dt * rhs_fn(state, *args)
        us = (3 / 4) * state + (1 / 4) * (u0 + dt * rhs_fn(u0, *args))
        return (1 / 3) * state + (2 / 3) * (us + dt * rhs_fn(us, *args))


class RK4(RungeKutta):
    """Classic four-stage RK4 — fourth-order accurate."""

    name = "RK4"

    def _step(self, rhs_fn, state, dt, *args):
        k1 = rhs_fn(state, *args)
        k2 = rhs_fn(state + 0.5 * dt * k1, *args)
        k3 = rhs_fn(state + 0.5 * dt * k2, *args)
        k4 = rhs_fn(state + dt * k3, *args)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    # Simple ODE: dy/dt = -y  →  exact solution y(t) = exp(-t)
    def rhs(y, lam=-1.0):
        return lam * y

    y0, dt, T = 1.0, 0.1, 1.0
    steps = int(T / dt)
    exact = math.exp(-T)

    print(f"{'Scheme':<8}  {'y(T)':>12}  {'error':>12}")
    print("-" * 36)
    for order in range(1, 5):
        rk = RungeKutta.from_order(order)
        y = y0
        for _ in range(steps):
            y = rk(rhs, y, dt)
        print(f"{rk.name:<8}  {y:>12.8f}  {abs(y - exact):>12.2e}")

    print(f"\n{'exact':<8}  {exact:>12.8f}")
