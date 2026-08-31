"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing_multirow_solver_shared_displacement.py

Roller-side counterpart of
Ball_Bearing/ball_bearing_multirow_solver_shared_displacement.py --
SAME shared-displacement idea (one flat root-find on the SHARED unknown,
each row's own reaction summed via superposition on a common rigid ring,
each row keeping its own contact stiffness, verified explicitly, never
pooled), adapted to line contact: ONE shared unknown (delta_r) instead of
two (delta_r, delta_a) -- radial roller bearings carry no axial load at
all (delta_a is fixed at 0.0 on every row, see
roller_bearing_solver.py/roller_bearing_results.py), so there is no axial
equilibrium equation here, unlike the ball side.

READ THIS BEFORE USING -- unlike the ball side, this is NOT validated
-----------------------------------------------------------------------
The ball-side shared-displacement solver was adopted as
ISO16281BallSolver.MULTIROW_SOLVER on the strength of concrete evidence:
a real family (MultiRowThrustBallFamily) that produces genuine multi-row
bearings, and a real comparison script
(design_bearing_combination_comparison.py) that ran both solvers side by
side and showed the fraction-based one failing to converge in a real case.

NONE of that exists here. No family in core/ today builds a
CYLINDRICAL_ROLLER Bearing with a genuine `rows` list of length >= 2:
CylindricalRollerFamily's own multi-row case (i >= 2) is the SAME
single-raceway REDUCTION_FACTOR_BY_ROWS multiplier DGBB uses on the ball
side -- one delta_r, one raceway, `i` only ever entering the closed-form
Cr formula, never this solver. MultiRowCylindricalRollerFamily (the family
that WOULD have produced a genuine rows list) was tried and retired --
see roller_bearing_results.py's own module docstring for that history. So
this file, and the roller_bearing_solver.py private->public promotion it
needed, exist for STRUCTURAL PARITY with the ball side, written at the
user's explicit request after being told exactly this caveat -- NOT
because any real multi-row roller bearing has been solved by it and found
correct. It has been sanity-checked only against a standalone,
hand-written re-implementation of its own row-aggregation logic (see
test_roller_shared_displacement_solver.py, same pattern used for the ball
file's own test) -- never against the real axisforge lamina kinematics,
because there is no real multi-row roller bearing to run those kinematics
on. Treat this as a structural placeholder ready for the day a genuine
multi-row roller case exists (most plausibly a future thrust roller
family -- see below for why even that would need MORE than a rename of
this file), not as validated production code the way the ball one is.

Formulation
-----------
Reuses ISO16281RollerSolver.elements() (the static, non-optimizing lamina
kinematics primitive -- given a shared delta_r and each row's own psi,
returns delta_j/psi_j/delta_jk/q_jk directly, no root-find) once per row,
per residual evaluation. Total radial reaction is the SUM of every row's
own contribution, each with its OWN cs (Sec 5.2.1 line-contact stiffness
constant, eq.36) and its own lamina geometry, set equal to the externally
applied Fr:

    Fr - sum_row [ sum_j cos(phi_j,row) * sum_k q_j,k,row ] = 0

Exactly ONE unknown (delta_r), exactly like ISO16281RollerSolver.
solve_contact() itself -- the "shared" part is that every row's elements()
call is evaluated at the SAME delta_r (co-located rows, identical to the
ball side's idealization), not that there is a second unknown to share.
psi is likewise a single shared INPUT scalar, passed identically to every
row's elements() call, mirroring solve_contact()'s own treatment of psi as
prescribed rather than solved (see roller_bearing_solver.py's own module
docstring for why).

IMPORTANT -- same caveat as the ball side's own solver, restated for line
contact: this is NOT a literal instance of ISO/TS 16281 eq.(45), which is
written for ONE raceway. What's above is each row's OWN eq.(45) (identical
functional form, that row's own cs/geometry), SUMMED via superposition of
independent lamina contacts sharing one rigid ring -- ordinary statics, a
step beyond what eq.(45) states on its own for a single raceway. Say so
explicitly wherever this solver's results get written up.

Per-row cs (and cL, used only for the initial-guess seed) is NEVER assumed
equal across rows -- see the explicit positivity/presence check in
solve_bearing() below, and the per-row `rv.cs` read inside residual()
(never a pooled/shared value). Each row's own cs is computed at that row's
own assemble_geometry() time, the line-contact analogue of the ball side's
cp -- see CylindricalRollerFamily.assemble_geometry() for the single-row
mechanism this would reuse per row, the same reuse pattern the ball side's
per-row cp already follows via SingleRowThrustBallFamily.assemble_geometry().

Why this file alone does NOT solve the "future thrust roller" case
--------------------------------------------------------------------
roller_bearing_results.py's own module docstring already flags that a
genuine multi-row roller case would most plausibly come from a future
THRUST roller family, where Fa (not Fr) is the primary load -- mirroring
how the ball side's genuine multi-row case (MultiRowThrustBallFamily) is
also thrust duty, sharing delta_a, not delta_r. This file does NOT cover
that case: it shares delta_r (radial), because that is the only unknown
ISO16281RollerSolver.elements() knows how to compute a lamina engagement
from (eq.38 is written entirely in terms of delta_r; there is no delta_a
term anywhere in it). A thrust roller version would need its OWN lamina
kinematics (axial engagement, not radial) before a shared-displacement
solver could be written for it -- guessing at that now, with no thrust
roller family or geometry to check it against, is exactly the "unsourced
physics" this codebase has been consistently avoiding; not attempted here.

No nested Newton, no FR_NEGLIGIBLE_EPS-equivalent needed
----------------------------------------------------------
Mirrors the ball side: residual() calls only elements() (no optimization)
per row, so there is exactly ONE scipy.optimize.root call, on a single
unknown, regardless of row count -- versus a fraction-based approach that
would nest i inner solve_contact() root-finds per outer iteration. Because
there is only ONE force component here (Fr; no Fa to speak of), there is
also no axial-fraction degeneracy of the kind found on the ball side
(Fa_total~=0 making f_a meaningless) to worry about at all -- the one
fraction this file reports (f_r) is always well-defined whenever Fr>0,
which every real load case here has (a floating/radial roller carrying
zero radial load is already flagged elsewhere by warn_if_floating_loaded).

Encapsulation note -- same story as the ball side, already resolved:
ISO16281RollerSolver.elements() was promoted from a private _elements() to
a public, static elements() in roller_bearing_solver.py specifically so
this file could call it directly -- see that module's own "UPDATED this
turn" docstring section.

REGISTERED onto ISO16281RollerSolver.MULTIROW_SOLVER at the bottom of this
file. rolling_bearing_solver.py's own import was switched to import THIS
module (was: roller_bearing_multirow_solver.py) so that registration
actually happens at orchestrator start-up.
roller_bearing_multirow_solver.py is NOT deleted -- it stays in the
codebase, it is simply no longer what dispatch reaches by default. Per
roller_bearing_results.py's own docstring, that file was itself framed
around MultiRowCylindricalRollerFamily, the now-retired family -- so
neither the old solver nor this new one currently has a real family to
serve; this one is preferred going forward purely because it mirrors the
ball side's (validated) architecture, not because it has been proven
correct against a real case the way the ball one was.

References
----------
ISO/TS 16281:2008 Sec 5.2.4.1 eq.(45) (radial equilibrium, per raceway)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6 (superposition
of independent line contacts on a shared rigid ring)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.numerics import run_root
from axisforge.solvers.machine_elements.bearings.ISO_16281.validation import check_bearing_ready, warn_if_floating_loaded

from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.single_row_solver import (
    ISO16281RollerSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.results import (
    RollerLoadDistributionResult,
    RollerBearingResult,
)
from axisforge.config import SOLVER_TOLERANCE

# Sec 5.2.2 -- "the number of laminae shall be at least n_s = 30". Same
# value as roller_bearing_solver.py's own _MIN_LAMINAE, kept as a separate
# local constant rather than importing that one -- it is named with a
# leading underscore there (module-private), and this file only needed
# elements() promoted to public, not that constant too; duplicating one
# integer is cheaper than widening a second seam for it.
_MIN_LAMINAE = 30


class ISO16281MultiRowRollerSolverSharedDisplacement:
    """
    Shared-displacement multi-row solver for line-contact (radial
    cylindrical roller) bearings -- structural counterpart of
    ISO16281MultiRowBallSolverSharedDisplacement, one shared unknown
    (delta_r) instead of two. See module docstring for the important
    caveat that, unlike the ball side, this has no real multi-row roller
    bearing to be validated against yet.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, outer_tol: float = 1e-6):
        self.tol       = tol        # unused internally (no nested solve_contact() calls
                                     # to pass it to) -- kept only for interface parity.
        self.outer_tol = outer_tol  # convergence tol on the single Fr residual

    def solve_bearing(self,
                       bearing: Bearing,
                       Fr_xz: float, Fr_xy: float,
                       psi: float,
                       label: str = "") -> RollerBearingResult:
        """
        bearing must expose `.rows` (list[dict], each carrying
        REQUIRED_ATTRS -- including its OWN cs, checked explicitly below)
        and `.i`. Fr_xz, Fr_xy, psi are the TOTAL reaction/misalignment for
        the whole (all-rows) bearing. No Fa parameter -- radial roller rows
        carry no axial load (mirrors solve_contact()'s own signature, which
        also has no Fa); rolling_bearing_solver.py's own dispatch only ever
        passes Fa when row_cls is ISO16281BallSolver, so this signature
        matches what it actually sends.

        Returns a RollerBearingResult via .multirow() -- same shape as
        ISO16281MultiRowRollerSolver.solve_bearing() (the fraction-based,
        no-longer-registered solver), so callers can swap one for the
        other without touching downstream code. f_r is BACK-COMPUTED from
        the converged solution (each row's own share of Fr) purely for
        API/reporting compatibility. f_a is always None here (see
        RollerBearingResult.multirow()'s own docstring) -- there is no
        axial load to split.
        """
        rows  = bearing.rows
        i     = len(rows)
        label = label or getattr(bearing, "label", "multirow")
        if i < 2:
            raise ValueError(f"{label}: multi-row solve needs >=2 rows, got {i}")

        row_views = []
        for j, row in enumerate(rows):
            rv = SimpleNamespace(**row, label=f"{label}-row{j}")
            check_bearing_ready(rv, rv.label, REQUIRED_ATTRS)   # REQUIRED_ATTRS already includes "cs"

            if rv.cs is None or rv.cs <= 0.0:
                raise ValueError(
                    f"{rv.label}: cs = {rv.cs!r} is not a valid (positive) line-contact "
                    f"stiffness. Every row must carry its OWN cs, computed at that row's "
                    f"own assemble_geometry() time (the line-contact analogue of the ball "
                    f"side's cp) -- this solver never falls back to a shared/default cs."
                )
            if rv.cL is None or rv.cL <= 0.0:
                raise ValueError(
                    f"{rv.label}: cL = {rv.cL!r} is not a valid (positive) lumped line-"
                    f"contact stiffness. cL is only used to seed the initial delta_r guess "
                    f"(eq.34) but is still required per REQUIRED_ATTRS, so it is checked "
                    f"here too rather than failing obscurely inside _initial_delta_r()."
                )
            if rv.n_s < _MIN_LAMINAE:
                raise ValueError(
                    f"{rv.label}: n_s = {rv.n_s} < {_MIN_LAMINAE} -- ISO/TS 16281 Sec 5.2.2 "
                    f"requires at least {_MIN_LAMINAE} laminae. (Re-implemented here rather "
                    f"than reaching into ISO16281RollerSolver._check_lamina_count(), which "
                    f"stays private -- this is a two-line check, not worth widening that "
                    f"seam for.)"
                )
            if len(rv.x_k) != rv.n_s:
                raise ValueError(
                    f"{rv.label}: len(x_k) = {len(rv.x_k)} != n_s = {rv.n_s} -- x_k must "
                    f"have exactly one lamina midpoint per n_s."
                )
            row_views.append(rv)

        # Not read anywhere else -- exists so a caller inspecting this
        # solver's behaviour (or a debugger) can see at a glance that rows
        # are NOT assumed to share cs; residual()/the post-solve pass below
        # always read rv.cs per row, independently, never this list.
        _cs_per_row = [rv.cs for rv in row_views]  # noqa: F841

        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))
        Fr     = float(np.hypot(Fr_xz, Fr_xy))

        def row_reaction(rv, delta_r: float):
            """(Fr_row, Mz_row, delta_j, psi_j, delta_jk, q_jk) for ONE row,
            at a GIVEN shared delta_r -- no root-find, just elements() plus
            rv's OWN cs applied inside elements() itself (unlike the ball
            side, cs IS used inside elements() -- eq.36 -- not applied
            afterward; still always rv's own value, never pooled, since
            elements() is called once per row with that row's own `rv`)."""
            delta_j, psi_j, delta_jk, q_jk, cp_j = ISO16281RollerSolver.elements(
                rv, delta_r, psi
            )
            Fr_row = float(np.sum(cp_j * np.sum(q_jk, axis=1)))
            x_k    = rv.x_k
            Mz_row = float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1)))
            return Fr_row, Mz_row, delta_j, psi_j, delta_jk, q_jk

        def residual(u: np.ndarray) -> np.ndarray:
            delta_r = u[0]
            Fr_total = 0.0
            for rv in row_views:
                Fr_row, _, _, _, _, _ = row_reaction(rv, delta_r)
                Fr_total += Fr_row
            return np.array([Fr - Fr_total])

        dr0 = _initial_delta_r(row_views, Fr)
        x, nfev, res_norm, ok = run_root(residual, [dr0], self.outer_tol)
        delta_r = float(x[0])

        row_results, Fr_rows = [], []
        for rv in row_views:
            Fr_row, Mz_row, delta_j, psi_j, delta_jk, q_jk = row_reaction(rv, delta_r)
            Fr_rows.append(Fr_row)
            alpha_j = np.full(rv.Z, rv.alpha_0, dtype=float)
            # n_iter/residual/ok are the SAME single outer solve's diagnostics
            # for every row here -- there was only ever one root-find, not i.
            row_results.append(RollerLoadDistributionResult(
                delta_r=delta_r, delta_a=0.0, psi=psi, phi_Fr=phi_Fr,
                delta_j=delta_j, alpha_j=alpha_j,
                Mz=Mz_row, n_iter=nfev, residual=res_norm, ok=ok,
                x_k=rv.x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
            ))

        # Guarded against dividing by a near-zero total, same convention as
        # the ball side's f_r/f_a reporting -- equal split reported if Fr~=0.
        f_r = (np.array(Fr_rows) / Fr) if Fr > 1e-9 else np.full(i, 1.0 / i)

        return RollerBearingResult.multirow(
            rows=row_results, f_r=f_r,
            n_iter=nfev, residual=res_norm, ok=ok,
        )


def _initial_delta_r(row_views, Fr: float) -> float:
    """
    Seed for delta_r -- same idea as ISO16281RollerSolver._initial_delta_r(),
    but pooling Z and cL across ALL rows (mean cL, total Z, mean s/2 gap)
    since there is one shared delta_r to seed, not one per row. Only a
    starting guess for Newton -- does not need to be exact.
    """
    if Fr <= 1e-6:
        return 0.0
    total_Z = sum(rv.Z for rv in row_views)
    mean_cL = float(np.mean([rv.cL for rv in row_views]))
    mean_gap = float(np.mean([rv.s for rv in row_views])) / 2.0
    line_hertz = (abs(Fr) / (mean_cL * total_Z)) ** (9.0 / 10.0)
    return mean_gap + line_hertz


# ---------------------------------------------------------------------------
# Registration -- runs at import time. rolling_bearing_solver.py imports
# this module (not roller_bearing_multirow_solver.py) specifically to
# trigger this side effect; see this module's top docstring section and
# that file's own docstring for the important caveat about validation.
# ---------------------------------------------------------------------------
ISO16281RollerSolver.MULTIROW_SOLVER = ISO16281MultiRowRollerSolverSharedDisplacement