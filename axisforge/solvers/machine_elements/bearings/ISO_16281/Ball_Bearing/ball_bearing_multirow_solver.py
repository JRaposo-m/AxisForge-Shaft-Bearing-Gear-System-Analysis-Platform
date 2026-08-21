"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_multirow_solver.py

Multi-row thrust ball bearing load distribution -- ORCHESTRATION ONLY.

This module does NOT reimplement any ISO/TS 16281 point-contact kinematics.
Every row is solved by the EXISTING ISO16281BallSolver (ball_bearing_solver.py),
called once per row per outer iteration via its PUBLIC solve_contact()
method -- the same single-bearing entry point that takes raw scalar loads
(Fr_xz, Fr_xy, Fa, psi, phi_Fr) directly, bypassing the ShaftSystem/library
batch path (`solve()`). Reusing it here, rather than duplicating eq.(12)-(28)
a second time, is the entire point of this file: one Q_j(delta) implementation,
one place it can have a bug.

RECONSTRUCTED, this turn -- thin orchestration over a public seam
---------------------------------------------------------------------
Two changes from the previous version of this file, both purely structural
-- the outer/inner iteration itself (physics, convergence behaviour, the
Fr~0 singular-Jacobian fix) is UNCHANGED, see below:

  1. self._row_solver._solve_bearing(...) -> self._row_solver.solve_contact(...).
     That inner call always reached into another class's method; it just
     used to be named with a leading underscore despite being called from
     outside ISO16281BallSolver. ball_bearing_solver.py now exposes
     solve_contact() (and REQUIRED_ATTRS) as public, formalizing exactly
     the contract this file was already relying on.

  2. This file no longer defines its own result container.
     MultiRowBallLoadDistributionResult is retired -- solve_bearing() now
     returns a BallBearingResult (ball_bearing_results.py), built via
     BallBearingResult.multirow(rows=row_results, f_r=f_r, f_a=f_a,
     n_iter=nfev, residual=res_norm, ok=ok). This is the SAME type
     ISO16281BallSolver.solve() now wraps its own single-row output in
     (via BallBearingResult.single()) -- so single-row and multi-row solves
     hand back one uniform result type, differing only in how many rows
     `.rows` holds and whether the outer-iteration fields are populated.
     `.delta_r` / `.delta_a` / `.psi` (reading row 0) are inherited from
     BallBearingResult unchanged from what this file's own
     MultiRowBallLoadDistributionResult used to provide directly.

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
  2. solves each row INDEPENDENTLY via ISO16281BallSolver.solve_contact();
  3. residual = every row's (delta_r, delta_a) minus row 0's (delta_r, delta_a).

At convergence (residual ~ 0) every row reports the same ring displacement --
exactly the compatibility condition a fully coupled multi-row solver would
enforce, reached here through repeated calls to the already-verified
single-row solver instead of a new joint nonlinear system.

Starts from an equal split (Fr_j = Fr/i, Fa_j = Fa/i for every row) --
your "começar com as forças a serem 50/50 [1/i for i rows] e depois
avançando" -- and lets run_root() converge it (the SAME helper
ball_bearing_solver.py already uses for the inner (delta_r, delta_a) solve).
CONFIRMED against the real library.py: run_root() is
`scipy.optimize.root(fun, x0, method="hybr", tol=tol)` with an "lm"
fallback if "hybr" doesn't converge -- genuinely dimension-general, x0 can
be any length as long as fun(x0) returns a same-length array (a square
system), which is exactly what this file's 2*(i-1)-unknown/2*(i-1)-residual
outer problem is.

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

Locally verified (not against the real repo -- see caveat below) with a
throwaway script exercising real SingleRowThrustBallFamily/
MultiRowThrustBallFamily geometry: (a) 2 identical rows under pure Fa
converge to an exact 50/50 split, and each row's delta_a matches, to
numerical precision, a DIRECT single-row solve_contact() call at Fa/2 --
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
    docstring for the physical idealization and method. Does not
    reimplement any ISO/TS 16281 kinematics; every row's contact problem is
    solved by ISO16281BallSolver.solve_contact(), unmodified.

    Deliberately does NOT subclass or get registered into
    rolling_bearing_solver.py's _SOLVER_MAP/_POSTPROC_MAP -- its result is a
    BallBearingResult with len(rows) >= 2, structurally indistinguishable
    at the type level from a single-row one, but the routing decision
    upstream (rolling_bearing_solver.RollingBearingSolver) is made on
    `.rows` count, not on which solver class produced the result -- see
    that module's own docstring for how it dispatches to this class today.
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
        bearing must be a MultiRowThrustBallFamily Bearing -- i.e. expose
        `.rows` (list[dict], each already carrying REQUIRED_ATTRS from
        SingleRowThrustBallFamily.assemble_geometry()) and `.i`.

        Fr_xz, Fr_xy, Fa, psi -- same meaning/units as
        ISO16281BallSolver.solve_contact()'s own arguments: the TOTAL
        reaction/misalignment for the whole (all-rows) bearing, exactly what
        a SimpleFEMResultsLibrary node already carries.

        Returns
        -------
        BallBearingResult, built via .multirow() -- len(rows) == i,
        f_r/f_a/outer_n_iter/outer_residual/outer_ok all populated.
        """
        rows  = bearing.rows
        i     = len(rows)
        label = label or getattr(bearing, "label", "multirow")
        if i < 2:
            raise ValueError(f"{label}: multi-row solve needs >=2 rows, got {i}")

        # SimpleNamespace views, not Bearing objects -- rows are plain dicts
        # (MultiRowThrustBallFamily never assembles a per-row Bearing/catalog,
        # see that family's own docstring), so this is the lightest way to
        # give ISO16281BallSolver.solve_contact() the attribute access
        # (bearing.A, .alpha_0, ...) it expects. check_bearing_ready() is the
        # SAME readiness check ball_bearing_solver.py runs on a real Bearing
        # before solving -- reused here rather than a hand-rolled duplicate.
        row_views = []
        for j, row in enumerate(rows):
            rv = SimpleNamespace(**row, label=f"{label}-row{j}")
            check_bearing_ready(rv, rv.label, REQUIRED_ATTRS)
            row_views.append(rv)

        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        # Fr ~= 0 (this repo's centered-Fa thrust scenarios, in particular)
        # makes the radial-split unknowns genuinely unconstrained: every
        # row sees Fr_xz_j = Fr_xy_j = 0 regardless of f_r, so those
        # Jacobian columns are identically zero. Verified directly against
        # the real solver (not just argued) -- see module docstring. Fix:
        # drop the radial unknowns entirely below this threshold and fix
        # f_r at 1/i -- removes the singular columns instead of hoping the
        # optimizer tolerates them.
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