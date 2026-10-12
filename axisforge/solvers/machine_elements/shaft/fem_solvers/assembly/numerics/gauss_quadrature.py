"""Gauss-Legendre quadrature rules and the choice of the integration order needed to integrate the
distributed-load work integral of one beam element (nodal load = integral of shape function times
the load intensity of the plane, over the element).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def gauss_points_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre points and weights on [-1, 1].

    Closed form for n <= 3, ``numpy.polynomial.legendre.leggauss`` for n > 3.

    Parameters
    ----------
    n : int
        Number of points, n >= 1.

    Returns
    -------
    points : ndarray, shape (n,)
        Abscissae xi on [-1, 1] [-].
    weights : ndarray, shape (n,)
        Weights [-]; they sum to 2.

    Raises
    ------
    ValueError
        If n < 1.

    Notes
    -----
    n = 1: (0, 2); n = 2: (+-1/sqrt(3), 1, 1); n = 3: (+-sqrt(3/5), 0) with weights (5/9, 8/9, 5/9).
    Values as in Onate2013_vol2, Appendix C, Table C.1, except that the table prints the n = 3
    abscissa as 0.774596697 (sqrt(3/5) = 0.7745966692). A rule of n points integrates a polynomial
    of degree 2n - 1 exactly (Onate2013_vol2, Appendix C.1).
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
    """Estimate the polynomial degree (0 to 3) of the load intensity q(x) over [x_lo, x_hi].

    Parameters
    ----------
    q : callable
        Load intensity ``q(x)`` [N/mm], ``x`` in [mm].
    x_lo, x_hi : float
        Interval [mm].

    Returns
    -------
    int
        Lowest degree d in 0..3 for which a least-squares polynomial of degree d reproduces q at 6
        equally spaced samples with a maximum absolute residual below 1e-8 [N/mm]; 3 if none does.

    Notes
    -----
    The probe (6 samples, degree up to 3, tolerance 1e-8) is a choice of this code, not of a source.
    """
    xs = np.linspace(x_lo, x_hi, 6)
    ys = np.array([q(x) for x in xs])
    for d in range(4):
        coeffs = np.polyfit(xs, ys, d)
        residual = np.max(np.abs(np.polyval(coeffs, xs) - ys))
        if residual < 1e-8:
            return d
    return 3


def theta_degree(theta_fn: Callable[[float], float], x_lo: float, x_hi: float) -> int:
    """Estimate the polynomial degree (0 to 3) of cos(theta(x)) over [x_lo, x_hi].

    Parameters
    ----------
    theta_fn : callable
        Load direction ``theta(x)`` [deg], ``x`` in [mm] (same unit as
        ``DistributedRadialLoad.theta_at``).
    x_lo, x_hi : float
        Interval [mm].

    Returns
    -------
    int
        Lowest degree d in 0..3 for which a least-squares polynomial of degree d reproduces
        ``cos(radians(theta_fn(x)))`` at 6 equally spaced samples with a maximum absolute residual
        below 1e-8 [-]; 3 if none does.

    Notes
    -----
    Only the cosine is probed; the sine of the direction is not sampled. The probe is a choice of
    this code.
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
    """Gauss order needed for the distributed-load integration of the elements of one shaft.

    All elements share the same beam theory, fixed once by ``BeamModelSettings``; the degree of the
    shape functions is a property of the theory (known analytically), not of an element, so no
    ``Elem`` is needed. Create one per analysis and reuse ``gauss_order`` for every load/element pair.

    Parameters
    ----------
    beam_theory : str
        ``"euler_bernoulli"`` (Hermite cubic shape functions, degree 3) or ``"timoshenko"`` (linear
        shape functions, degree 1).

    Raises
    ------
    ValueError
        If ``beam_theory`` is unknown (a third theory must be added to the lookup table explicitly).
    """

    # Known analytically, not estimated:
    #   euler_bernoulli: Hermite cubics -> degree 3
    #   timoshenko:      linear N       -> degree 1
    # A third beam_theory must be entered here explicitly -- deliberate
    # lookup, not a fallback.
    _SHAPE_FUNCTION_DEGREE = {
        "euler_bernoulli": 3,  # Hermite cubic shape functions 
        "timoshenko": 1,
    }

    def __init__(self, beam_theory: str):
        """Look up the shape-function degree of ``beam_theory``; see the class docstring."""
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
        """Smallest Gauss order that integrates the element load work exactly for polynomial data.

        Parameters
        ----------
        q : callable
            Load intensity ``q(x)`` [N/mm].
        x_lo, x_hi : float
            Integration interval [mm] (overlap of the load with the element).
        theta_fn : callable or None, optional
            Load direction ``theta(x)`` [deg]; if given, the degree of cos(theta) is added.

        Returns
        -------
        int
            ``max(1, ceil((p + 1) / 2))`` with ``p = deg(q) + deg(cos theta) + deg(N)``, where ``deg(N)``
            is 3 (Euler-Bernoulli) or 1 (Timoshenko).

        Notes
        -----
        A rule of n points integrates a polynomial of degree 2n - 1 exactly (Onate2013_vol2,
        Appendix C.1), so n = ceil((p + 1) / 2) is exact when the integrand is a polynomial of degree p.
        The degree probes of ``q`` and ``theta`` are a choice of this code.

        The callers pass ``q`` already projected on the bending plane (it contains the cosine or sine
        of the direction) and also ``theta_fn``, so a variable direction raises the order twice. Each
        probe stops at degree 3: the order is at most 5 (Euler-Bernoulli) or 4 (Timoshenko), and a
        non-polynomial load is integrated approximately.
        """
        deg_q = q_degree(q, x_lo, x_hi)
        deg_theta = theta_degree(theta_fn, x_lo, x_hi) if theta_fn else 0
        p = deg_q + deg_theta + self._deg_N
        return max(1, math.ceil((p + 1) / 2))