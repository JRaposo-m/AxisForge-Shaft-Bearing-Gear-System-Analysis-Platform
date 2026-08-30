"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_multirow_solver_shared_displacement.py

Alternative multi-row thrust ball bearing solver -- SAME public contract
as ball_bearing_multirow_solver.py's ISO16281MultiRowBallSolver
(solve_bearing(bearing, Fr_xz, Fr_xy, Fa, psi, label) -> BallBearingResult),
different internal formulation.

UPDATED, this turn -- ADOPTED as ISO16281BallSolver.MULTIROW_SOLVER
-------------------------------------------------------------------
This was written for side-by-side comparison, explicitly NOT to replace
the existing (fraction-based) solver -- that stance has now changed, based
on evidence rather than just the theoretical trade-off discussion below.
Running both solvers side by side on a real gearbox model
(design_bearing_combination_comparison.py's "SOLVER COMPARISON" section)
showed ISO16281MultiRowBallSolver (fraction-based) fails to converge
(outer_ok=False, residual stuck ~6e-3, hits its iteration cap) for a
double-row bearing with Fa~=0 (a purely radially-loaded shaft, the common
case) AND heterogeneous rows (different cp per row) -- most likely the
axial-side counterpart of the Fr~=0 degeneracy that solver's own
FR_NEGLIGIBLE_EPS already handles (when Fa_total~=0, ANY per-row axial
load-split fraction f_a_row satisfies f_a_row*Fa_total=0, so that Jacobian
column goes singular), unhandled on the axial side. This solver does not
share that failure mode by construction -- see "Formulation" below, it
never divides by a possibly-zero total -- and converged cleanly
(outer_ok=True, residual ~1e-6..1e-7) on every case tested, including the
ones where the fraction-based solver failed.

Consequence: this class is now registered as
ISO16281BallSolver.MULTIROW_SOLVER at the bottom of this file (was: not
registered at all). rolling_bearing_solver.py's own import was switched to
import THIS module (was: ball_bearing_multirow_solver.py) so that
registration actually happens at orchestrator start-up -- see that file's
own docstring. ball_bearing_multirow_solver.py is NOT deleted; it stays in
the codebase for reference/comparison, it is simply no longer the default
dispatch reaches.

Also, as a direct consequence of adoption: the "Encapsulation note" below
is now RESOLVED, not just flagged -- ISO16281BallSolver._elements() was
promoted to a public elements() (no leading underscore) in
ball_bearing_solver.py specifically because of this file, and every call
site here was updated to match (ISO16281BallSolver.elements(...), not
._elements(...)). See that module's own "UPDATED this turn" docstring
note.

Physical problem
-----------------
Same idealization as the existing solver: co-located rows (zero axial
offset between contact planes) -- every row of the shared rigid ring sees
the SAME (delta_r, delta_a). Where this file differs is HOW that shared
displacement is found.

Formulation -- one flat root-find, no nested solve
----------------------------------------------------
The existing solver parametrises by PER-ROW LOAD-SPLIT FRACTIONS
(2(i-1) unknowns, growing with row count) and calls
ISO16281BallSolver.solve_contact() -- itself a 2-equation root-find --
once per row, per outer iteration; convergence is checked by requiring
every row's independently-solved (delta_r, delta_a) to agree with row 0's.

This file instead parametrises directly by the SHARED (delta_r, delta_a)
-- always exactly 2 unknowns, regardless of row count -- and reuses
ISO16281BallSolver.elements() (the static, non-optimizing kinematics
primitive: given a displacement, returns delta_j/alpha_j/Q_j-ingredients
directly, no root-find) once per row, per residual evaluation. Total
reaction is the SUM of every row's own contribution, each with its OWN
cp, set equal to the externally applied (Fr, Fa):

    Fr - sum_row [ cp_row * sum_j delta_j,row^1.5 * cos(alpha_j,row) * cos(phi_j,row) ] = 0
    Fa - sum_row [ cp_row * sum_j delta_j,row^1.5 * sin(alpha_j,row) ]                  = 0

IMPORTANT -- this is NOT a literal instance of ISO/TS 16281 eq.(16)-(17):
those are written for ONE raceway with ONE cp (Sec 4.2.2.1). What's above
is each row's OWN eq.(16)-(17) (identical functional form, that row's own
cp/Z/geometry), SUMMED via superposition of independent Hertzian contacts
sharing one rigid ring -- ordinary statics (3rd-law reactions on a common
rigid body), not an extra physical assumption, but genuinely a step beyond
what eq.(16)-(17) states on its own for a single raceway. Say so explicitly
wherever this solver's results get written up -- do not present it as "the
standard already covers multi-row this way."

Per-row cp is NEVER assumed equal across rows -- see the explicit
positivity/presence check in solve_bearing() below, and the
per-row `rv.cp` read inside residual() (never a pooled/shared value).
Each row's own cp is computed at that row's own assemble_geometry() time
(e.g. SingleRowThrustBallFamily.assemble_geometry() ->
bc.hertz_spring_constant(), the exact same mechanism
DeepGrooveBallFamily.assemble_geometry() uses for its own single-row cp --
see that file's `cp = bc.hertz_spring_constant(Dw, ri, re, E, nu, alpha_0,
Dpw)` line). A bearing built from identical row_specs (the common case
shown in MultiRowThrustBallFamily's own usage example, `row_specs=[row,
row]`) will legitimately produce equal cp per row -- that is a property
of the INPUT geometry, not something this solver assumes, enforces, or
special-cases; heterogeneous row_specs (different Dw/E/nu per row) flow
through the exact same code path with each row keeping its own cp.

Naming note: elements() returns a tuple whose 5th element is also called
`cp_j` in the original source -- but that `cp_j` is `cos(phi_j)` (a ball
ANGULAR POSITION cosine), not the Hertz contact stiffness `cp` used here.
Same three letters, two unrelated quantities, both in scope at once in
residual() below -- kept exactly as named in the source rather than
renamed, but flagged here so it is not misread.

No nested Newton: unlike the existing solver (i inner solve_contact()
root-finds per outer iteration), this file's residual() calls only
elements() (no optimization) per row, so there is exactly ONE
scipy.optimize.root call, total, on 2 unknowns, regardless of row count.
No FR_NEGLIGIBLE_EPS-style special case is needed for Fr ~= 0 either -- a
plain 2-unknown/2-equation system degenerates gracefully to delta_r ~= 0
rather than producing zero Jacobian columns the way the fraction
parametrisation's radial unknowns did (see the existing solver's own
module docstring for that issue). The only place Fr~=0/Fa~=0 still needs
a guard here is the back-computed f_r/f_a *reporting* fields (division by
a near-zero total), not the solve itself.

Encapsulation note -- RESOLVED, this turn: ISO16281BallSolver._elements()
used to be named with a leading underscore (an internal implementation
detail of that class), and this file called it directly from outside the
class rather than through solve_contact() -- a deliberate, acknowledged
trade-off at the time for avoiding the nested-solve cost, with "consider
promoting it to non-underscored if this file is adopted" left as an open
follow-up. It has now been adopted (see this module's top docstring
section) and _elements() promoted to a public elements() in
ball_bearing_solver.py -- every call site below uses
ISO16281BallSolver.elements(...), not ._elements(...).

REGISTERED onto ISO16281BallSolver.MULTIROW_SOLVER at the bottom of this
file, this turn -- see this module's top docstring section for why. It was
previously an explicit standalone comparison, not wired into dispatch;
that has changed.

References
----------
ISO/TS 16281:2008 Sec 4.2.2.1 eq.(12)-(18) (equilibrium, per raceway)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6 (superposition
of independent Hertzian contacts on a shared rigid ring)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    check_bearing_ready,
    run_root,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.single_row_solver import (
    ISO16281BallSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.results import (
    BallLoadDistributionResult,
    BallBearingResult,
)
from axisforge.config import SOLVER_TOLERANCE


class ISO16281MultiRowBallSolverSharedDisplacement:
    """
    Alternative to ISO16281MultiRowBallSolver -- same public contract
    (solve_bearing()), shared-displacement formulation instead of
    load-split fractions. See module docstring for the full trade-off
    discussion, and for why this is now the class registered as
    ISO16281BallSolver.MULTIROW_SOLVER (the one rolling_bearing_solver.py's
    dispatch actually reaches for a ball multi-row bearing).
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, outer_tol: float = 1e-6):
        self.tol       = tol        # unused internally (no nested solve_contact() calls
                                     # to pass it to) -- kept only for interface parity
                                     # with ISO16281MultiRowBallSolver's constructor.
        self.outer_tol = outer_tol  # convergence tol on the single (Fr, Fa) residual

    def solve_bearing(self,
                       bearing: Bearing,
                       Fr_xz: float, Fr_xy: float, Fa: float,
                       psi: float,
                       label: str = "") -> BallBearingResult:
        """
        bearing must expose `.rows` (list[dict], each carrying
        REQUIRED_ATTRS -- including its OWN cp, checked explicitly below)
        and `.i`. Fr_xz, Fr_xy, Fa, psi are the TOTAL reaction/misalignment
        for the whole (all-rows) bearing.

        Returns a BallBearingResult via .multirow() -- same shape as
        ISO16281MultiRowBallSolver.solve_bearing(), so callers can swap one
        solver for the other without touching downstream code. f_r/f_a are
        BACK-COMPUTED from the converged solution (each row's own share of
        Fr/Fa) purely for API/reporting compatibility -- this solver never
        treats them as unknowns the way the fraction-based one does; they
        fall out once (delta_r, delta_a) is known, not before.
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
            check_bearing_ready(rv, rv.label, REQUIRED_ATTRS)   # REQUIRED_ATTRS already includes "cp"
            if rv.cp is None or rv.cp <= 0.0:
                raise ValueError(
                    f"{rv.label}: cp = {rv.cp!r} is not a valid (positive) Hertzian "
                    f"contact stiffness. Every row must carry its OWN cp, computed at "
                    f"that row's own assemble_geometry() time (see e.g. "
                    f"SingleRowThrustBallFamily.assemble_geometry() -> "
                    f"bc.hertz_spring_constant(), the same mechanism "
                    f"DeepGrooveBallFamily uses for its own single-row cp) -- this "
                    f"solver never falls back to a shared/default cp."
                )
            row_views.append(rv)

        # Not read anywhere else -- exists so a caller inspecting this
        # solver's behaviour (or a debugger) can see at a glance that rows
        # are NOT assumed to share cp; residual()/the post-solve pass below
        # always read rv.cp per row, independently, never this list.
        _cp_per_row = [rv.cp for rv in row_views]  # noqa: F841

        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))
        Fr     = float(np.hypot(Fr_xz, Fr_xy))

        def row_reactions(rv, delta_r: float, delta_a: float):
            """(Fr_row, Fa_row, Mz_row, delta_j, alpha_j) for ONE row, at a
            GIVEN shared displacement -- no root-find, just elements()
            plus rv's OWN cp applied afterward (cp is never touched inside
            elements() itself -- see module docstring's naming note)."""
            Vpsi = rv.Ri * np.sin(psi) * np.cos(rv.phi_j)
            delta_j, alpha_j, ca, sa, cos_phi_j, d32 = ISO16281BallSolver.elements(
                rv, delta_r, delta_a, Vpsi
            )
            Fr_row = rv.cp * float(np.sum(d32 * ca * cos_phi_j))
            Fa_row = rv.cp * float(np.sum(d32 * sa))
            Mz_row = (rv.Dpw / 2.0) * rv.cp * float(np.sum(d32 * sa * cos_phi_j))
            return Fr_row, Fa_row, Mz_row, delta_j, alpha_j

        def residual(u: np.ndarray) -> np.ndarray:
            delta_r, delta_a = u
            Fr_total = Fa_total = 0.0
            for rv in row_views:
                Fr_row, Fa_row, _, _, _ = row_reactions(rv, delta_r, delta_a)
                Fr_total += Fr_row
                Fa_total += Fa_row
            return np.array([Fr - Fr_total, Fa - Fa_total])

        dr0 = _initial_delta_r(row_views, Fr)
        da0 = _initial_delta_a(row_views, Fa)
        x, nfev, res_norm, ok = run_root(residual, [dr0, da0], self.outer_tol)
        delta_r, delta_a = float(x[0]), float(x[1])

        row_results, Fr_rows, Fa_rows = [], [], []
        for rv in row_views:
            Fr_row, Fa_row, Mz_row, delta_j, alpha_j = row_reactions(rv, delta_r, delta_a)
            Fr_rows.append(Fr_row)
            Fa_rows.append(Fa_row)
            # n_iter/residual/ok are the SAME single outer solve's diagnostics
            # for every row here -- unlike the fraction-based solver, there is
            # no separate per-row convergence to report (there was only ever
            # one root-find, not i of them).
            row_results.append(BallLoadDistributionResult(
                delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
                delta_j=delta_j, alpha_j=alpha_j,
                Mz=Mz_row, n_iter=nfev, residual=res_norm, ok=ok,
            ))

        # Guarded against dividing by a near-zero total -- the SOLVE itself
        # needed no such guard (see module docstring), only this reporting
        # step, when the corresponding external load component is ~0 and a
        # "share of it" is therefore undefined -- equal split is reported in
        # that case, same convention as the fraction-based solver's
        # f_r_fixed = 1/i for the Fr~=0 case.
        f_r = (np.array(Fr_rows) / Fr) if Fr > 1e-9 else np.full(i, 1.0 / i)
        f_a = (np.array(Fa_rows) / Fa) if abs(Fa) > 1e-9 else np.full(i, 1.0 / i)

        return BallBearingResult.multirow(
            rows=row_results, f_r=f_r, f_a=f_a,
            n_iter=nfev, residual=res_norm, ok=ok,
        )


def _initial_delta_r(row_views, Fr: float) -> float:
    """
    Seed for delta_r -- same Hertz-scale idea as
    ISO16281BallSolver._initial_delta_r(), but pooling Z and cp across ALL
    rows (mean cp, total Z) since there is one shared displacement to seed,
    not one per row. Only a starting guess for Newton -- does not need to
    be exact, and is not used anywhere the positivity/presence of each
    row's own cp isn't already guaranteed by solve_bearing()'s own check.
    """
    if Fr <= 1e-6:
        return 0.0
    total_Z = sum(rv.Z for rv in row_views)
    mean_cp = float(np.mean([rv.cp for rv in row_views]))
    return (Fr / (mean_cp * total_Z)) ** (2.0 / 3.0)


def _initial_delta_a(row_views, Fa: float) -> float:
    """Seed for delta_a -- mirrors _initial_delta_r()."""
    if Fa == 0.0:
        return 0.0
    total_Z = sum(rv.Z for rv in row_views)
    mean_cp = float(np.mean([rv.cp for rv in row_views]))
    hertz   = (abs(Fa) / (mean_cp * total_Z)) ** (2.0 / 3.0)
    return float(np.copysign(hertz, Fa))


# ---------------------------------------------------------------------------
# Registration -- runs at import time. rolling_bearing_solver.py imports
# this module (not ball_bearing_multirow_solver.py) specifically to trigger
# this side effect; see this module's top docstring section and that
# file's own docstring for why the switch was made this turn.
# ---------------------------------------------------------------------------
ISO16281BallSolver.MULTIROW_SOLVER = ISO16281MultiRowBallSolverSharedDisplacement