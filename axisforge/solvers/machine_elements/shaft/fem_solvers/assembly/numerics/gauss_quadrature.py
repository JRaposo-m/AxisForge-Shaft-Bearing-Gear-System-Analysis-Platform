"""
axisforge/solvers/machine_elements/shaft/fem_solvers/assembly/numerics/gauss_quadrature.py

Gauss-Legendre quadrature rules, and estimation of the minimum
integration order needed to (near-)exactly integrate one element's
distributed-load work integral:

    f_i = ∫ N_i(zeta) · q(x(zeta)) · [cos/sin theta(x(zeta))] · J dzeta

q_degree/theta_degree stay free functions — they sample an arbitrary
user-supplied Callable and don't depend on beam theory at all.
shape_function_degree is a per-theory constant (Hermite cubic = 3,
Timoshenko linear = 1), fixed once for the whole shaft by
BeamModelSettings — not something to re-derive from an Elem instance
on every call. QuadratureOrderEstimator holds that constant so callers
building the load vector never have to pass an Elem in just to read
one attribute that's the same for every element in the mesh.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def gauss_points_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre points and weights on [-1, 1] for arbitrary n.
    Closed form for n <= 3; numpy.polynomial.legendre for n > 3.
    """
    if n < 1:
        raise ValueError(f"gauss_points_weights: n must be >= 1, got {n}")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    elif n == 2:
        s = 1.0 / np.sqrt(3)
        return np.array([-s, s]), np.array([1.0, 1.0])
    elif n == 3:
        s = np.sqrt(3 / 5)
        return np.array([-s, 0.0, s]), np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        return np.polynomial.legendre.leggauss(n)


def q_degree(q: Callable[[float], float], x_lo: float, x_hi: float) -> int:
    """Estimate the polynomial degree of q(x) over [x_lo, x_hi]."""
    xs = np.linspace(x_lo, x_hi, 6)
    ys = np.array([q(x) for x in xs])
    for d in range(4):
        coeffs = np.polyfit(xs, ys, d)
        residual = np.max(np.abs(np.polyval(coeffs, xs) - ys))
        if residual < 1e-8:
            return d
    return 3


def theta_degree(theta_fn: Callable[[float], float], x_lo: float, x_hi: float) -> int:
    """
    Estimate the effective polynomial degree of cos(theta(x)) over
    [x_lo, x_hi] by sampling cos(theta(x)) directly. theta_fn returns
    degrees, matching DistributedRadialLoad.theta_at().
    """
    xs = np.linspace(x_lo, x_hi, 6)
    ys = np.array([np.cos(np.radians(theta_fn(x))) for x in xs])
    for d in range(4):
        coeffs = np.polyfit(xs, ys, d)
        residual = np.max(np.abs(np.polyval(coeffs, xs) - ys))
        if residual < 1e-8:
            return d
    return 3


class QuadratureOrderEstimator:
    """
    Decides the Gauss order needed for distributed-load integration
    over an entire shaft's worth of elements, all sharing the same
    beam_theory (fixed once by BeamModelSettings — see
    mesh/shaft/beam_model_settings.py).

    Instantiate once per analysis; reuse .gauss_order() for every
    distributed load / element pair. No Elem instance is ever needed
    here — shape function degree is a property of the theory, not of
    any particular element.
    """

    # Known analytically, not estimated:
    #   euler_bernoulli: Hermite cubics -> degree 3
    #   timoshenko:      linear N       -> degree 1
    # A third beam_theory must be entered here explicitly — deliberate
    # lookup, not a fallback.
    _SHAPE_FUNCTION_DEGREE = {
        "euler_bernoulli": 3,  # Hermite cubic shape functions 
        "timoshenko": 1,
    }

    def __init__(self, beam_theory: str):
        try:
            self._deg_N = self._SHAPE_FUNCTION_DEGREE[beam_theory]
        except KeyError:
            raise ValueError(
                f"QuadratureOrderEstimator: unknown beam_theory '{beam_theory}', "
                f"expected one of {tuple(self._SHAPE_FUNCTION_DEGREE)}"
            ) from None
        self.beam_theory = beam_theory

    def gauss_order(self, q: Callable[[float], float], x_lo: float, x_hi: float,
                     theta_fn: Callable[[float], float] | None = None) -> int:
        """
        Minimum Gauss order to (near-)exactly integrate this element's
        distributed-load contribution, from the combined polynomial
        degree of q(x), cos/sin(theta(x)) (if direction varies), and
        this estimator's fixed shape-function degree.
        """
        deg_q = q_degree(q, x_lo, x_hi)
        deg_theta = theta_degree(theta_fn, x_lo, x_hi) if theta_fn else 0
        p = deg_q + deg_theta + self._deg_N
        return max(1, math.ceil((p + 1) / 2))