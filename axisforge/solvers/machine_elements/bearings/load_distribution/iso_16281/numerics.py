"""Nonlinear-system driver of the ISO/TS 16281 load-distribution solvers.

Thin layer over SciPy that solves the contact-equilibrium residual of the
ball and roller solvers (clause 4.2 and 5.2 of ISO_TS_16281_2008); it knows
nothing about bearings and imports nothing from the solvers.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import root


__all__ = ["run_root"]


def run_root(fun: Callable, x0: Sequence[float],
             tol: float) -> tuple[np.ndarray, int, float, bool]:
    """Find the zero of a nonlinear residual vector with SciPy ``root``.

    Numerical method: Powell hybrid (MINPACK ``hybr``, a dogleg Newton-type
    method with a finite-difference Jacobian). If ``hybr`` reports failure,
    the system is solved again from the same ``x0`` with Levenberg-Marquardt
    (MINPACK ``lm``) and the attempt with the smaller residual norm is kept,
    since a failed ``hybr`` run can end closer to the root than a nominally
    successful ``lm`` run. In the ISO/TS 16281 solvers the unknowns are the
    relative radial (and axial) displacement of the bearing rings,
    ``delta_r`` (and ``delta_a``), found from the force equilibrium.

    Parameters
    ----------
    fun : callable
        Residual ``f(x) -> array_like`` with the same length as ``x``; zero at
        the solution. In ``contact_solver.py`` ``x`` is ``[delta_r, delta_a]``
        [mm] (ball, point contact, single row and multi-row) or ``[delta_r]``
        [mm] (roller, line contact), and ``f`` is the force residual [N]
        (radial, and axial for balls).
    x0 : sequence of float
        Initial guess, same units as ``x`` [mm] in the callers.
    tol : float
        Tolerance passed unchanged to ``scipy.optimize.root`` (``tol``
        argument, which SciPy maps to the method's own tolerance; relative
        step of the solution for ``hybr``, see the SciPy documentation). The
        callers pass ``self.tol`` of the solver, default ``SOLVER_TOLERANCE`` =
        1e-6 (general numerical tolerance of ``axisforge.config``, not from
        the standard) [-]. It is not compared with the force residual.

    Returns
    -------
    x : np.ndarray
        Solution vector, always at least 1-D; same units as ``x0`` [mm] in the
        callers.
    nfev : int
        Residual evaluations reported by the accepted attempt (0 if the
        solver did not report it) [-].
    residual_norm : float
        Euclidean norm of the residual at ``x``, in the units of ``fun``
        ([N] in the callers). This is the value to check against a physical
        tolerance, since ``success`` alone does not guarantee convergence to
        the intended root.
    success : bool
        The accepted attempt's own success flag.

    Notes
    -----
    ``nfev`` and ``success`` describe the ACCEPTED attempt only; the cost of
    the discarded attempt is not included in ``nfev``. The fallback rule
    (keep the smaller residual norm) is a heuristic of this code, not from
    the standard. ISO_TS_16281_2008 prescribes the equilibrium equations, not
    the solution algorithm.

    References
    ----------
    SciPy ``scipy.optimize.root`` (software documentation, not in the
    library: source to find).
    """
    sol = root(fun, x0, method="hybr", tol=tol)
    if not sol.success:
        sol_lm = root(fun, x0, method="lm", tol=tol)
        if (np.linalg.norm(np.atleast_1d(sol_lm.fun)) <
                np.linalg.norm(np.atleast_1d(sol.fun))):
            sol = sol_lm

    return (np.atleast_1d(sol.x),
            int(getattr(sol, "nfev", 0)),
            float(np.linalg.norm(np.atleast_1d(sol.fun))),
            bool(sol.success))