"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing_multirow_solver.py

Multi-row radial cylindrical roller bearing load distribution --
ORCHESTRATION ONLY. Mirrors Ball_Bearing/ball_bearing_multirow_solver.py's
structure; the physical idealization differs where the two bearing types
genuinely differ -- see below.

This module does NOT reimplement any ISO/TS 16281 lamina-model kinematics.
Every row is solved by the EXISTING ISO16281RollerSolver
(roller_bearing_solver.py), called once per row per outer iteration via its
public solve_contact() method -- the same single-bearing entry point that
takes raw scalar loads (Fr_xz, Fr_xy, psi, phi_Fr) directly, bypassing the
ShaftSystem/library batch path (`solve()`). Reusing it here is the entire
point of this file: one lamina-load implementation, one place it can have
a bug.

Physical problem -- and why it is SIMPLER than the ball thrust case
-----------------------------------------------------------------------
A multi-row radial cylindrical roller bearing (MultiRowCylindricalRollerFamily,
i >= 2 rows) is ONE rigid ring/sleeve. Each row has its own roller
set/lamina geometry (Z, Dwe, Lwe, Dpw, phi_j, s, n_s, x_k, cL, cs, alpha_0 --
see bearing.rows[j]) but ALL rows share the SAME rigid-ring radial
displacement delta_r -- that sharing is exactly the coupling a single
ISO16281RollerSolver call cannot see (it solves one row per call, by
design). This is structurally the same coupling
ball_bearing_multirow_solver.py handles for thrust ball rows -- but with
only ONE degree of freedom (delta_r) instead of two (delta_r, delta_a),
because a radial roller bearing carries no axial load at all (delta_a is
always 0.0 on every RollerLoadDistributionResult, roller or not) -- there
is no axial split to iterate here, unlike the ball case.

IDEALIZATION, deliberately chosen this turn -- FORCE SPLIT ONLY, no moment
-----------------------------------------------------------------------------
Rows are treated as CO-LOCATED (zero axial offset between their contact
planes), exactly like ball_bearing_multirow_solver.py's own idealization
for thrust ball rows -- see that file's module docstring for the same
argument applied here: representing a real axial offset between rows would
need the offset itself plus a per-row psi correction, which this solver
does not attempt. Dropping the offset collapses the compatibility condition
from "consistent displacement at each row's own offset position" down to
"identical delta_r at every row" -- which is what is actually enforced
below.

THIS IS A KNOWN SIMPLIFICATION, NOT A COMPLETE MODEL -- flagged explicitly
per instruction, for follow-up:
A real double-row cylindrical roller bearing (e.g. NNU/NN-type) usually has
its rows genuinely separated along the bearing's axial width, not
co-located. An external moment on the shaft could then load the two rows
UNEVENLY beyond what a pure radial-force split captures -- e.g. one row
carrying more of Fr than the other specifically because of a moment
reaction, not just because of stiffness differences between heterogeneous
rows. Sec 5.2.4's eq.(45)/(46) pairing (radial force balance + moment
balance) is exactly the single-row mechanism that would need a multi-row
counterpart -- a genuine "does the compatibility condition include a
per-row moment share" question -- to model that correctly. This file does
NOT attempt it: only Fr is split across rows below; psi is a single
FEM-projected input shared unchanged by every row (same convention as the
ball side), and no inter-row moment balance is solved or enforced. TODO:
revisit once the row-offset / moment-sharing question has been worked out
-- until then, treat a converged result from this solver as "correct under
the co-located-rows idealization", not as a full moment-aware multi-row
solve.

Method -- outer Gauss-Seidel / fixed-point load split, inner exact solves
---------------------------------------------------------------------------
Unknowns: i-1 independent radial-load fractions (row 0's own fraction is
whatever makes the set sum to 1 -- so every candidate split this residual
ever evaluates already satisfies Fr = Fr_1 + ... + Fr_i by construction,
never as a fitted approximation). Each outer root-find iteration:
  1. splits the bearing's total (Fr_xz, Fr_xy) across rows using the
     current fraction guess -- SAME phi_Fr direction for every row (only
     the magnitude is split; all rows sit on one sleeve seeing one applied
     force direction, so splitting the direction itself would not be
     physical, same argument as the ball side);
  2. solves each row INDEPENDENTLY via ISO16281RollerSolver.solve_contact();
  3. residual = every row's delta_r minus row 0's delta_r.

At convergence (residual ~ 0) every row reports the same ring displacement
-- exactly the compatibility condition a fully coupled multi-row solver
would enforce, reached here through repeated calls to the already-verified
single-row solver instead of a new joint nonlinear system.

Starts from an equal split (Fr_j = Fr/i for every row) and lets run_root()
converge it -- the SAME helper roller_bearing_solver.py already uses for
the inner delta_r solve. Mirrors ball_bearing_multirow_solver.py's own
starting point exactly (see that file for the run_root() dimension-
generality argument, which applies unchanged here).

Fr ~= 0 degenerate case -- simpler than the ball side's, not the same fix
-----------------------------------------------------------------------------
ball_bearing_multirow_solver.py has to handle Fr~=0 specially because ball
rows still have an AXIAL split to solve even when the radial split is
undefined (every row sees Fr_xz_j = Fr_xy_j = 0 regardless of f_r, making
those Jacobian columns identically zero, while the axial residual stays
well-posed). A radial roller bearing has no such second degree of freedom
-- Fr is the ONLY thing ever split here. So when Fr ~= 0, there is no load
at all to divide, and NO outer problem to solve: every row would converge
to the same near-zero-load delta_r regardless of which split fraction is
tried, making the outer residual identically zero for ANY f_r (a fully
singular system, not just some of its columns). Handled below by skipping
run_root() entirely below FR_NEGLIGIBLE_EPS and returning the trivial equal
split (f_r = 1/i for every row) directly -- there is no "well-posed part"
of this problem left to solve when Fr ~= 0, unlike the ball case.

NOT verified against the real repo -- same caveat as
ball_bearing_multirow_solver.py's own: this session's contact_stiffness.py
equivalents are stubs, not the real Hertzian/lamina geometry. Treat the
SOLVER MECHANICS above as the intended design, not yet run end-to-end.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    check_bearing_ready,
    run_root,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_solver import (
    ISO16281RollerSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import (
    RollerLoadDistributionResult,
    RollerBearingResult,
)
from axisforge.config import SOLVER_TOLERANCE


# [N] -- below this, Fr is treated as zero: no load to split across rows,
# outer root-find skipped entirely (see module docstring).
_FR_NEGLIGIBLE_EPS = 1.0e-9


class ISO16281MultiRowRollerSolver:
    """
    Orchestration-only multi-row radial cylindrical roller bearing solver --
    see module docstring for the physical idealization (co-located rows,
    FORCE SPLIT ONLY -- no moment/Mz compatibility between rows, explicitly
    deferred) and method. Does not reimplement any ISO/TS 16281 lamina
    kinematics; every row's contact problem is solved by
    ISO16281RollerSolver.solve_contact(), unmodified.

    Deliberately does NOT get registered into rolling_bearing_solver.py's
    _SOLVER_MAP/_POSTPROC_MAP, mirroring ISO16281MultiRowBallSolver's own
    reasoning -- routing to this class from the orchestrator still needs
    that module to learn to dispatch by contact type instead of always
    reaching for ISO16281MultiRowBallSolver (see rolling_bearing_solver.py's
    own known-breakage notes); not addressed by this file.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, outer_tol: float = 1e-6):
        self.tol         = tol         # passed through to each row's inner solve
        self.outer_tol   = outer_tol   # convergence tol on the cross-row residual
        self._row_solver = ISO16281RollerSolver(tol=tol)

    def solve_bearing(self,
                       bearing: Bearing,
                       Fr_xz: float, Fr_xy: float,
                       psi: float,
                       label: str = "") -> RollerBearingResult:
        """
        bearing must be a MultiRowCylindricalRollerFamily Bearing -- i.e.
        expose `.rows` (list[dict], each already carrying REQUIRED_ATTRS
        from CylindricalRollerFamily.assemble_geometry()) and `.i`.

        Fr_xz, Fr_xy, psi -- same meaning/units as
        ISO16281RollerSolver.solve_contact()'s own arguments: the TOTAL
        reaction/misalignment for the whole (all-rows) bearing, exactly
        what a SimpleFEMResultsLibrary node already carries. No Fa
        argument -- a radial roller bearing carries no axial load, single-
        row or multi-row alike (see roller_bearing_results.py's
        RollerBearingResult.f_a docstring).

        Returns
        -------
        RollerBearingResult, built via .multirow() -- len(rows) == i,
        f_r/outer_n_iter/outer_residual/outer_ok populated, f_a always None.
        """
        rows  = bearing.rows
        i     = len(rows)
        label = label or getattr(bearing, "label", "multirow")
        if i < 2:
            raise ValueError(f"{label}: multi-row solve needs >=2 rows, got {i}")

        # SimpleNamespace views, not Bearing objects -- rows are plain dicts
        # (MultiRowCylindricalRollerFamily never assembles a per-row
        # Bearing/catalog, see that family's own docstring), so this is the
        # lightest way to give ISO16281RollerSolver.solve_contact() the
        # attribute access (bearing.Z, .Dwe, ...) it expects.
        # check_bearing_ready() is the SAME readiness check
        # roller_bearing_solver.py runs on a real Bearing before solving --
        # reused here rather than a hand-rolled duplicate.
        row_views = []
        for j, row in enumerate(rows):
            rv = SimpleNamespace(**row, label=f"{label}-row{j}")
            check_bearing_ready(rv, rv.label, REQUIRED_ATTRS)
            row_views.append(rv)

        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))
        Fr     = float(np.hypot(Fr_xz, Fr_xy))

        def solve_rows(f_r: np.ndarray) -> list[RollerLoadDistributionResult]:
            return [
                self._row_solver.solve_contact(
                    rv,
                    Fr_xz=f_r[j] * Fr_xz, Fr_xy=f_r[j] * Fr_xy,
                    delta_r_init=0.0,
                    psi=psi, phi_Fr=phi_Fr,
                )
                for j, rv in enumerate(row_views)
            ]

        if Fr <= _FR_NEGLIGIBLE_EPS:
            # No load to split -- see module docstring, "Fr ~= 0 degenerate
            # case". Every row would converge to the same near-zero-load
            # delta_r regardless of f_r, so the outer residual is
            # identically zero for ANY split -- skip run_root() entirely
            # and record the trivial equal split.
            f_r = np.full(i, 1.0 / i)
            row_results = solve_rows(f_r)
            return RollerBearingResult.multirow(
                rows=row_results, f_r=f_r, n_iter=0, residual=0.0, ok=True,
            )

        def fractions(free: np.ndarray) -> np.ndarray:
            return np.concatenate([[1.0 - free.sum()], free])

        def residual(free: np.ndarray) -> np.ndarray:
            f_r = fractions(free)
            res = solve_rows(f_r)
            dr0 = res[0].delta_r
            return np.array([r.delta_r - dr0 for r in res[1:]])

        free0 = np.full(i - 1, 1.0 / i)   # equal split, every free unknown

        x, nfev, res_norm, ok = run_root(residual, free0, self.outer_tol)
        f_r         = fractions(np.asarray(x))
        row_results = solve_rows(f_r)

        return RollerBearingResult.multirow(
            rows=row_results, f_r=f_r,
            n_iter=nfev, residual=res_norm, ok=ok,
        )