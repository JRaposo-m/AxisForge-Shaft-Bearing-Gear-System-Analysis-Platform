"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_multirow_solver.py

Multi-row thrust ball bearing load distribution -- ORCHESTRATION ONLY.

This module does NOT reimplement any ISO/TS 16281 point-contact kinematics.
Every row is solved by the EXISTING ISO16281BallSolver (ball_bearing_solver.py),
called once per row per outer iteration via its `_solve_bearing()` method --
the same private, single-bearing entry point that takes raw scalar loads
(Fr_xz, Fr_xy, Fa, psi, phi_Fr) directly, bypassing the ShaftSystem/library
batch path (`solve()`). Reusing it here, rather than duplicating eq.(12)-(28)
a second time, is the entire point of this file: one Q_j(delta) implementation,
one place it can have a bug.

Physical problem
-----------------
A multi-row thrust ball bearing (MultiRowThrustBallFamily, i >= 2 rows) is
ONE rigid ring/washer. Each row has its own ball set/contact geometry
(A, alpha_0, phi_j, Ri, cp, Dpw, Z -- see bearing.rows[j]) but ALL rows share
the SAME rigid-ring displacement (delta_r, delta_a) -- that sharing is
exactly the coupling a single ISO16281BallSolver call cannot see (it solves
one row per call, by design).

Idealization made here, stated explicitly rather than left implicit:
  - Rows are treated as CO-LOCATED (zero axial offset between their contact
    planes). The real geometry has a small offset between rows; representing
    it would need the offset itself (not reliably available -- see
    design_bearing_combination_comparison.py's RUN LOG) plus a per-row psi
    correction. Dropping the offset collapses the compatibility condition
    from "consistent displacement at each row's own offset position" down to
    "identical (delta_r, delta_a) at every row" -- which is what is actually
    enforced below.
  - psi (misalignment) is a single FEM-projected input, shared by all rows
    unchanged -- it is not split or iterated, only the load fractions are.

Method -- outer Gauss-Seidel / fixed-point load split, inner exact solves
---------------------------------------------------------------------------
Unknowns: i-1 independent radial-load fractions and i-1 independent
axial-load fractions (row 0's own fraction is whatever makes each set sum
to 1 -- so every candidate split this residual ever evaluates already
satisfies Fr = Fr_1 + ... + Fr_i and Fa = Fa_1 + ... + Fa_i by construction,
never as a fitted approximation). Each outer root-find iteration:
  1. splits the bearing's total (Fr_xz, Fr_xy, Fa) across rows using the
     current fraction guess -- SAME phi_Fr direction for every row (only the
     magnitude is split; all rows sit on one ring seeing one applied force
     direction, so splitting the direction itself would not be physical);
  2. solves each row INDEPENDENTLY via ISO16281BallSolver._solve_bearing();
  3. residual = every row's (delta_r, delta_a) minus row 0's (delta_r, delta_a).

At convergence (residual ~ 0) every row reports the same ring displacement --
exactly the compatibility condition a fully coupled multi-row solver would
enforce, reached here through repeated calls to the already-verified
single-row solver instead of a new joint nonlinear system.

Starts from an equal split (Fr_j = Fr/i, Fa_j = Fa/i for every row) --
your "começar com as forças a serem 50/50 [1/i for i rows] e depois
avançando" -- and lets run_root() converge it (the SAME helper
ball_bearing_solver.py already uses for the inner (delta_r, delta_a) solve).
CONFIRMED against the real library.py (pasted after this file was first
written): run_root() is `scipy.optimize.root(fun, x0, method="hybr", tol=tol)`
with an "lm" fallback if "hybr" doesn't converge -- genuinely dimension-
general, x0 can be any length as long as fun(x0) returns a same-length
array (a square system), which is exactly what this file's 2*(i-1)-unknown/
2*(i-1)-residual outer problem is. No longer an assumption.

FIXED, not just flagged -- this bit for real in local testing (see below):
when Fr_xz = Fr_xy = 0 (this repo's centered-Fa thrust-bearing comparison
scenarios), the radial-split unknowns have NO effect on the residual at
all -- every row sees Fr_xz_j = Fr_xy_j = 0 regardless of their fraction,
so those Jacobian columns are identically zero. Reproduced directly against
the real solver: with a 2-unknown (radial + axial) outer problem at Fr=0,
scipy's "hybr" wandered the radial unknown to f_r0 = 49.9 (nonsensical --
fractions must sum to 1) while the residual stayed ~1e-11 (dominated by the
equally tiny, correctly-converged axial residual), and hybr reported
success=False; run_root()'s hybr-vs-lm fallback only swaps to lm's
well-behaved answer (f_r0 = 0.5) when lm's residual norm is STRICTLY
smaller, which at that scale it usually isn't -- so the final `ok` came
back False even though the axial split and inter-row delta_a agreement
(the only physically meaningful part of the answer when Fr=0) were both
correct. Fixed below: when Fr is below a small threshold (1e-9 N), the
radial unknowns are dropped from the outer problem entirely and f_r is
fixed at 1/i per row -- removes the singular columns instead of hoping the
optimizer tolerates them. Re-verified after the fix: identical rows still
converge to the exact expected 50/50 axial split with ok=True; a
heterogeneous case (Z=14 + Z=8 rows, pure Fa) converges to f_a =
[0.636, 0.364] -- the stiffer, more-balls row correctly carries the larger
share to match the softer row's delta_a -- also with ok=True, where before
this fix the same case came back ok=False despite an already-correct answer.

Cross-checked against the real ball_bearing_results.py, library.py and
rolling_bearing_solver.py (all three pasted after this file was first
written) -- _REQUIRED_ATTRS/BallLoadDistributionResult's field set,
check_bearing_ready()'s getattr-based check, and run_root()'s dimension-
generality all match what this file assumed.

Locally verified (not against the real repo -- see caveat below) with a
throwaway script exercising real SingleRowThrustBallFamily/
MultiRowThrustBallFamily geometry: (a) 2 identical rows under pure Fa
converge to an exact 50/50 split, and each row's delta_a matches, to
numerical precision, a DIRECT single-row _solve_bearing() call at Fa/2 --
the correct closed-form answer for that symmetric case; (b) 2 heterogeneous
rows (Z=14 vs Z=8) converge to an unequal split favoring the stiffer
(more-balls) row, the physically expected direction. Still NOT run against
the real repo (this session's contact_stiffness.py is a stub, not the real
Hertzian geometry -- see that file's own docstring) -- treat the SOLVER
MECHANICS above as verified, the absolute numbers as shape-of-the-answer
only until you run it for real.
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
    _REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
    BallLoadDistributionResult,
)
from axisforge.config import SOLVER_TOLERANCE


class MultiRowBallLoadDistributionResult:
    """
    Per-row BallLoadDistributionResult list plus the converged split
    fractions -- NOT a single BallLoadDistributionResult, because there is
    no single (delta_j, alpha_j) array once more than one row is involved:
    each row has its own Z balls. row_results[j] is exactly what
    ISO16281BallSolver would have produced had row j been solved alone with
    its converged share of the total load.

    delta_r / delta_a / psi properties read row 0 -- at convergence every
    row agrees (that agreement IS the convergence criterion), so any row
    would do; row 0 is picked for definiteness, not because it's special.
    """
    __slots__ = ("row_results", "f_r", "f_a", "n_iter", "residual", "ok")

    def __init__(self, row_results: list[BallLoadDistributionResult],
                 f_r: np.ndarray, f_a: np.ndarray,
                 n_iter: int, residual: float, ok: bool):
        self.row_results = row_results
        self.f_r         = f_r
        self.f_a         = f_a
        self.n_iter       = n_iter
        self.residual     = residual
        self.ok           = ok

    @property
    def delta_r(self) -> float:
        return self.row_results[0].delta_r

    @property
    def delta_a(self) -> float:
        return self.row_results[0].delta_a

    @property
    def psi(self) -> float:
        return self.row_results[0].psi


class ISO16281MultiRowBallSolver:
    """
    Orchestration-only multi-row thrust ball bearing solver -- see module
    docstring for the physical idealization and method. Does not
    reimplement any ISO/TS 16281 kinematics; every row's contact problem is
    solved by ISO16281BallSolver._solve_bearing(), unmodified.

    Deliberately does NOT subclass or get registered into
    rolling_bearing_solver.py's _SOLVER_MAP/_POSTPROC_MAP -- its output
    shape (MultiRowBallLoadDistributionResult, a list of per-row results)
    is not interchangeable with BallLoadDistributionResult, so it cannot
    be dropped into that dispatch table unmodified. Call it directly, the
    same way design_bearing_combination_comparison.py's
    capacity_only_multirow_comparison() calls bearing.family.dynamic_capacity()
    directly rather than through RollingBearingSolver.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, outer_tol: float = 1e-6):
        self.tol         = tol         # passed through to each row's inner solve
        self.outer_tol   = outer_tol   # convergence tol on the cross-row residual
        self._row_solver = ISO16281BallSolver(tol=tol)

    def solve_bearing(self,
                       bearing: Bearing,
                       Fr_xz: float, Fr_xy: float, Fa: float,
                       psi: float,
                       label: str = "") -> MultiRowBallLoadDistributionResult:
        """
        bearing must be a MultiRowThrustBallFamily Bearing -- i.e. expose
        `.rows` (list[dict], each already carrying _REQUIRED_ATTRS from
        SingleRowThrustBallFamily.assemble_geometry()) and `.i`.

        Fr_xz, Fr_xy, Fa, psi -- same meaning/units as
        ISO16281BallSolver._solve_bearing()'s own arguments: the TOTAL
        reaction/misalignment for the whole (all-rows) bearing, exactly what
        a SimpleFEMResultsLibrary node already carries. This method is what
        would sit between that node and i separate _solve_bearing() calls
        -- it is the "additional class that handles only the multi-row case
        and orchestrates the other class's calculation" you asked for.
        """
        rows  = bearing.rows
        i     = len(rows)
        label = label or getattr(bearing, "label", "multirow")
        if i < 2:
            raise ValueError(f"{label}: multi-row solve needs >=2 rows, got {i}")

        # SimpleNamespace views, not Bearing objects -- rows are plain dicts
        # (MultiRowThrustBallFamily never assembles a per-row Bearing/catalog,
        # see that family's own docstring), so this is the lightest way to
        # give ISO16281BallSolver._solve_bearing() the attribute access
        # (bearing.A, .alpha_0, ...) it expects. check_bearing_ready() is the
        # SAME readiness check ball_bearing_solver.py runs on a real Bearing
        # before solving -- reused here rather than a hand-rolled duplicate,
        # now that library.py's real signature is confirmed to work by
        # getattr() (so it applies to a SimpleNamespace exactly as it would
        # to a Bearing).
        row_views = []
        for j, row in enumerate(rows):
            rv = SimpleNamespace(**row, label=f"{label}-row{j}")
            check_bearing_ready(rv, rv.label, _REQUIRED_ATTRS)
            row_views.append(rv)

        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        # Fr ~= 0 (this repo's centered-Fa thrust scenarios, in particular)
        # makes the radial-split unknowns genuinely unconstrained: every
        # row sees Fr_xz_j = Fr_xy_j = 0 regardless of f_r, so those
        # Jacobian columns are identically zero. Verified directly against
        # the real solver (not just argued): with both radial and axial
        # unknowns free at Fr=0, scipy's "hybr" wanders the radial unknown
        # to a nonsensical value (seen: f_r0 = 49.9) while still reporting
        # a near-zero residual (~1e-11, dominated by the equally-tiny axial
        # residual) -- and run_root()'s hybr-vs-lm fallback only swaps to
        # lm's well-behaved answer (f_r0 = 0.5) when lm's residual norm is
        # STRICTLY smaller, which it usually isn't at that scale, so `ok`
        # can come back False even though the axial split/delta_a agreement
        # (the only physically meaningful part of the answer here) is
        # correct. Fix: drop the radial unknowns entirely below this
        # threshold and fix f_r at 1/i -- removes the singular columns
        # instead of hoping the optimizer tolerates them.
        FR_NEGLIGIBLE_EPS = 1.0e-9   # [N]
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
                self._row_solver._solve_bearing(
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

        return MultiRowBallLoadDistributionResult(
            row_results=row_results, f_r=f_r, f_a=f_a,
            n_iter=nfev, residual=res_norm, ok=ok,
        )