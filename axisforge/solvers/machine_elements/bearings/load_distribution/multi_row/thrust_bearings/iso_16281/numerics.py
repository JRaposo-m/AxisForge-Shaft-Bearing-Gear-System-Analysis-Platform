"""
axisforge/solvers/machine_elements/bearings/ISO_16281/numerics.py

Generic numerical helpers shared across solvers.

Nothing in this module knows about bearings, shafts or gears -- it is a thin,
dependency-free layer over SciPy that any solver in the project may import.
Keeping it at the `solvers/` root (rather than inside a per-element package)
is deliberate: `run_root` is the same nonlinear-system driver whether the
residual comes from ISO/TS 16281 contact equilibrium, a fan-out power split,
or anything added later.

Import direction
----------------
    solvers/**  ---->  numerics.py

Never the reverse. This module must not import from any solver package,
core element, or results registry.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import root


__all__ = ["run_root"]


def run_root(fun: Callable, x0: Sequence[float],
             tol: float) -> tuple[np.ndarray, int, float, bool]:
    """
    Solve a nonlinear system with scipy.optimize.root, hybr with lm fallback.

    Tries 'hybr' first. If it reports failure, retries with 'lm' and keeps
    whichever attempt left the smaller residual norm -- an unsuccessful hybr
    run is not discarded blindly, since it is not unusual for it to land
    closer than a nominally successful lm run.

    Parameters
    ----------
    fun : callable
        Residual function, f(x) -> array_like. Zero at the solution.
    x0 : sequence of float
        Initial guess.
    tol : float
        Tolerance passed through to scipy.optimize.root.

    Returns
    -------
    x : np.ndarray
        Solution vector, always at least 1-D.
    nfev : int
        Residual evaluations reported by the accepted attempt (0 if the
        solver did not report it).
    residual_norm : float
        Euclidean norm of the residual at `x` -- the value to check against
        a physical tolerance, since `success` alone is not a guarantee of
        convergence to the intended root.
    success : bool
        The accepted attempt's own success flag.

    Notes
    -----
    `nfev` and `success` describe the ACCEPTED attempt only; the discarded
    attempt's cost is not included in `nfev`.
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