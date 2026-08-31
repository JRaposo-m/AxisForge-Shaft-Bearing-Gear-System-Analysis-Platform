"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_multirow_solver.py

this file is maintained simply to show the analysis on the combination_comparison where the 
diference in the anaylysis/ approach so the problem can be seen and the diference in results
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.numerics import run_root
from axisforge.solvers.machine_elements.bearings.ISO_16281.validation import check_bearing_ready, warn_if_floating_loaded

from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.single_row_solver import (
    ISO16281BallSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.results import (
    BallLoadDistributionResult,
    BallBearingResult,
)
from axisforge.config import SOLVER_TOLERANCE


class ISO16281MultiRowBallSolver:
    """
    Orchestration-only multi-row thrust ball bearing solver -- see module
    docstring for the idealization and method. Every row's contact problem
    is solved by ISO16281BallSolver.solve_contact(), unmodified.

    Registers itself onto ISO16281BallSolver.MULTIROW_SOLVER at the bottom
    of this module. Routing upstream (RollingBearingSolver) is structural:
    _is_multirow() picks out a multi-row bearing by `.rows` count, then the
    row's own attribute set is matched against the single-row registry
    (dispatch.resolve_solver_cls_for_attrs()) to find which single-row
    solver its rows belong to, and `.MULTIROW_SOLVER` on that class gives
    this one.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, outer_tol: float = 1e-6):
        self.tol         = tol         # passed through to each row's inner solve
        self.outer_tol   = outer_tol   # convergence tol on the cross-row residual
        self._row_solver = ISO16281BallSolver(tol=tol)

    def solve_bearing(self,
                       bearing: Bearing,
                       Fr_xz: float, Fr_xy: float, Fa: float,
                       psi: float,
                       label: str = "") -> BallBearingResult:
        """
        bearing must expose `.rows` (list[dict], each carrying
        REQUIRED_ATTRS) and `.i`. Fr_xz, Fr_xy, Fa, psi are the TOTAL
        reaction/misalignment for the whole (all-rows) bearing.

        Returns a BallBearingResult via .multirow() -- len(rows) == i,
        f_r/f_a/outer_n_iter/outer_residual/outer_ok all populated.
        """
        rows  = bearing.rows
        i     = len(rows)
        label = label or getattr(bearing, "label", "multirow")
        if i < 2:
            raise ValueError(f"{label}: multi-row solve needs >=2 rows, got {i}")

        # SimpleNamespace views, not Bearing objects -- rows are plain dicts
        # (MultiRowThrustBallFamily never assembles a per-row Bearing/catalog).
        row_views = []
        for j, row in enumerate(rows):
            rv = SimpleNamespace(**row, label=f"{label}-row{j}")
            check_bearing_ready(rv, rv.label, REQUIRED_ATTRS)
            row_views.append(rv)

        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        FR_NEGLIGIBLE_EPS = 1.0e-9   # [N] -- see module docstring
        Fr = float(np.hypot(Fr_xz, Fr_xy))
        split_radial = Fr > FR_NEGLIGIBLE_EPS

        f_r_fixed = np.full(i, 1.0 / i)   # used as-is when split_radial is False

        def fractions(free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if split_radial:
                free_r, free_a = free[: i - 1], free[i - 1:]
                f_r = np.concatenate([[1.0 - free_r.sum()], free_r])
            else:
                f_r, free_a = f_r_fixed, free
            f_a = np.concatenate([[1.0 - free_a.sum()], free_a])
            return f_r, f_a

        def solve_rows(f_r: np.ndarray, f_a: np.ndarray) -> list[BallLoadDistributionResult]:
            return [
                self._row_solver.solve_contact(
                    rv,
                    Fr_xz=f_r[j] * Fr_xz, Fr_xy=f_r[j] * Fr_xy, Fa=f_a[j] * Fa,
                    delta_r_init=0.0, delta_a_init=0.0,
                    psi=psi, phi_Fr=phi_Fr,
                )
                for j, rv in enumerate(row_views)
            ]

        def residual(free: np.ndarray) -> np.ndarray:
            f_r, f_a = fractions(free)
            res = solve_rows(f_r, f_a)
            da0 = res[0].delta_a
            out = [r.delta_a - da0 for r in res[1:]]
            if split_radial:
                dr0 = res[0].delta_r
                out = [r.delta_r - dr0 for r in res[1:]] + out
            return np.array(out)

        n_free = (2 if split_radial else 1) * (i - 1)
        free0  = np.full(n_free, 1.0 / i)   # equal split, every active unknown

        x, nfev, res_norm, ok = run_root(residual, free0, self.outer_tol)
        f_r, f_a    = fractions(np.asarray(x))
        row_results = solve_rows(f_r, f_a)

        return BallBearingResult.multirow(
            rows=row_results, f_r=f_r, f_a=f_a,
            n_iter=nfev, residual=res_norm, ok=ok,
        )


ISO16281BallSolver.MULTIROW_SOLVER = ISO16281MultiRowBallSolver