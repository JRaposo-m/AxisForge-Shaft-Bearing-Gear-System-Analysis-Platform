"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_multirow_solver.py

Multi-row thrust ball bearing load distribution -- ORCHESTRATION ONLY.
Reuses ISO16281BallSolver.solve_contact() per row, once per outer
iteration -- no ISO/TS 16281 kinematics duplicated here.

Physical problem
-----------------
A multi-row thrust ball bearing (MultiRowThrustBallFamily, i >= 2 rows) is
ONE rigid ring/washer. Each row has its own contact geometry (A, alpha_0,
phi_j, Ri, cp, Dpw, Z -- bearing.rows[j]) but ALL rows share the SAME rigid
ring displacement (delta_r, delta_a) -- the coupling a single
ISO16281BallSolver call can't see on its own.

Idealization: rows are treated as co-located (zero axial offset between
contact planes) -- collapses the compatibility condition to "identical
(delta_r, delta_a) at every row". psi is a single FEM-projected input,
shared unchanged by every row (not split or iterated).

Method -- outer fixed-point load split, inner exact solves
--------------------------------------------------------------
Unknowns: i-1 independent radial fractions + i-1 independent axial
fractions (row 0's own fraction makes each set sum to 1, so every
candidate already satisfies Fr = sum(Fr_j) and Fa = sum(Fa_j) by
construction). Each outer iteration: split (Fr_xz, Fr_xy, Fa) across rows
by the current fractions (same phi_Fr direction for every row -- only
magnitude splits), solve each row independently via solve_contact(),
residual = every row's (delta_r, delta_a) minus row 0's.

Fr ~= 0 singularity (this repo's centered-Fa thrust scenarios): when
Fr_xz = Fr_xy = 0, the radial fractions have no effect on the residual at
all -- those Jacobian columns are identically zero, and scipy's hybr can
wander them nonsensically while still reporting a tiny residual norm and
success=False, even though the physically meaningful part (axial split,
inter-row delta_a agreement) is already correct. Fixed by dropping the
radial unknowns from the outer problem below FR_NEGLIGIBLE_EPS and fixing
f_r = 1/i per row instead of leaving singular columns for the optimizer.

Verified locally with real SingleRowThrustBallFamily/MultiRowThrustBallFamily
geometry (identical rows under pure Fa -> exact 50/50 split matching a
direct single-row solve_contact() at Fa/2; heterogeneous Z=14/Z=8 rows ->
unequal split favoring the stiffer row) -- not yet run against this
session's contact_stiffness.py, which is a stub. Treat the solver
mechanics as verified, absolute numbers as shape-of-the-answer only.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    check_bearing_ready,
    run_root,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_solver import (
    ISO16281BallSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
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