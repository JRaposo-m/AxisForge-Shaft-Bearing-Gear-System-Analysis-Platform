"""
design_bearing_combination_comparison.py

Compares bearing-pair COMBINATIONS on the same load-lifting gearbox topology
(same shaft/gear pipeline as design_load_distribution_mixed_bearings.py):
loops over exactly 3 SCENARIOS --
  1. single-row locating (6204) + single-row floating roller (NU204)
     [baseline]
  2. DOUBLE-row locating (6204) + DOUBLE-row floating roller (NU204),
     IDENTICAL rows on both sides
  3. DOUBLE-row locating (6204) + DOUBLE-row floating roller (NU204),
     HETEROGENEOUS rows on both sides (differ only in Dw/Dwe per row)
-- see this module's changelog for the restructuring that arrived at
exactly these 3 (earlier turns had 4, including two now-superseded
"double-ball-only" scenarios).

Duplicated from design_load_distribution_mixed_bearings.py rather than
edited in place, per fixtures/README.md ("do not modify for a specific
case -- duplicate and rename").

Output: one comparison table per shaft, grouped by scenario, PLUS a SOLVER
COMPARISON section for whichever scenario's LOCATING bearing is double-row
ball (see "Known gaps" below for why that section stays ball-only).

Changelog (most recent first):
  - FIX, this turn -- both double-row scenarios crashed on shaft1 the first
    time they were actually run, AFTER the earlier floating-side fix below:
      ValueError: ShaftSystem 'shaft1(motor)' validation failed:
        - shaft1(motor): bearing 'brg1a' [6.000, 34.000] mm overlaps
          shoulder at x=30.000 mm
    Root cause: this is the LOCATING-side twin of the floating-side crash
    already fixed below, surfaced only now because of an upstream core fix
    (Shaft.shoulders(), in axisforge/core/machine_elements/shaft/shaft.py --
    not in this file) that started reporting EVERY transition instead of
    only the one that used to live on shoulder_right. Before that fix,
    ShaftSystem.validate_or_raise() never saw the seatA/body shoulder at
    x=30 at all, so a b=28 double-row locating bearing (DR_CATALOG) centered
    at the single-row default position=20 within shaft1's 30mm seatA
    silently passed validation despite genuinely overlapping the shoulder
    by 4mm ([6,34] vs. x=30). The "arrangement='locating' bearings are
    evidently not subject to the same strict shoulder-overlap check" note
    in the FIX entry below was describing behaviour observed against that
    buggy shoulders() -- no longer true now that it's fixed.

    Fixed the same way the floating side already was: new
    DR_LOCATING_POSITIONS constant (midpoint of each shaft's locating seat,
    seatA), threaded through build_systems() as a new locating_positions
    parameter (default None -> the original single-row positions 20.0/
    25.0/25.0, so BASELINE_SCENARIO is unaffected) and a 5th element on
    each SCENARIOS entry. See DR_LOCATING_POSITIONS' own comment for the
    margin arithmetic per shaft -- mirrors DR_ROLLER_FLOATING_POSITIONS'
    reasoning exactly.

  - roller L10r/Pref_r WIRED IN -- new _bearing_life_roller(): q_kci/q_kce
    via RollingElementCapacity.per_lamina() (uniform per lamina, eq.56-57,
    core/.../families/roller/radial/functions/capacity.py), q_kei/q_kee via
    LaminaDynamicEquivalentLoad.from_distribution() (per row, avoids the
    untested multi-row branch of from_bearing_result()). Applies to both
    single-row and double-row roller bearings.

  - L10r/Pref_r ADDED -- eq.(29)-(31), via ball_bearing_postprocessing.py.
    Ball rows only at first; roller followed in the entry above.

  - RESTRUCTURED, this turn -- exactly 3 scenarios now (was 4), and the
    SOLVER COMPARISON section is fixed to compare two GENUINELY different
    solvers again. Both requested together ("corrigas isso e faças 3
    cenarios, single ball, single roller / double ball, double roller com
    rows iguais, e depois as duas double mas diferentes rows").

    (a) SOLVER COMPARISON fix -- since MULTIROW_SOLVER was switched to the
    shared-displacement class (see the older "UPDATED this turn" entries
    below), bundle.extra["multirow_result"] -- what this file's SOLVER
    COMPARISON section used to read as the "fraction-based" column -- IS
    the shared-displacement result now (RollingBearingSolver's own
    dispatch reaches whatever is currently registered). Using it as
    "fraction-based" silently turned the comparison into shared-
    displacement vs. itself -- which is exactly why every single row, in
    every scenario, showed diff=0.000e+00 the first time this ran after
    the switch: not agreement between two solvers, just the same solver
    counted twice. Fixed by instantiating ISO16281MultiRowBallSolver (the
    true fraction-based class, still importable, still not deleted) and
    calling IT directly too, on the identical reconstructed inputs,
    exactly the way shared_solver was already being called directly. Both
    "fraction" and "shared" in solver_comparison are now independent,
    freshly-computed solve_bearing() calls -- neither reuses
    bundle.extra["multirow_result"] (still used, correctly, for this
    file's own PRODUCTION numbers -- the main comparison table -- since
    that genuinely is whatever RollingBearingSolver actually solved with).

    (b) Scenario restructuring -- the two "double-ball-only" scenarios
    (double-row locating + SINGLE-row floating, homogeneous and
    heterogeneous) are removed; DOUBLE_BALL_DOUBLE_ROLLER_SCENARIO is kept
    but renamed DOUBLE_HOMO_SCENARIO, and a NEW DOUBLE_HETERO_SCENARIO adds
    heterogeneous rows on the FLOATING (roller) side too -- previously
    heterogeneity was ball-only (DOUBLE_ROW_HETERO_SCENARIO). New constant
    DR_ROLLER_ROW_GEOM_HETERO mirrors DR_ROW_GEOM_HETERO's approach: the 2
    roller rows differ ONLY in Dwe (row0 keeps CR_GEOM's 6.5mm, row1 uses a
    smaller 5.5mm), same Lwe/Dpw/Z/s/n_s, so ROLLER_REQUIRED_ATTRS stays
    internally consistent while the two rows' line-contact stiffness (cs)
    genuinely differs -- built via the existing _build_dr_rows_roller().
    DR_ROLLER_CATALOG/DR_ROLLER_FLOATING_POSITIONS are UNCHANGED and reused
    by both double scenarios -- the mounting envelope (b/d/D) and seat
    position don't depend on the individual rows' Dwe.

    MULTIROW_SLOTS_BY_SCENARIO and SCENARIOS were rewritten for the new 3-
    scenario set; BASELINE_SCENARIO/DOUBLE_HOMO_SCENARIO/
    DOUBLE_HETERO_SCENARIO are the new named constants (replacing the old
    inline baseline string and DOUBLE_ROW_SCENARIO/DOUBLE_ROW_HETERO_
    SCENARIO/DOUBLE_BALL_DOUBLE_ROLLER_SCENARIO). plot_qmax_comparison()'s
    color palette reverted from 4 back to 3 entries.

  - FIX, upstream (roller_bearing_solver.py), not in this file -- after
    the shoulder-overlap fix below, the SAME scenario crashed again one
    step later with AttributeError: 'types.SimpleNamespace' object has no
    attribute 'P_xk', raised inside ISO16281RollerSolver.elements() (eq.41's
    profile-correction term) the first time it ran against a genuinely
    minimal REQUIRED_ATTRS-shaped row view instead of a real Bearing.
    Root cause: roller_bearing_solver.py's REQUIRED_ATTRS tuple never
    listed "P_xk", even though elements() always needed it -- invisible
    until now because every previous caller of elements() was a real
    Bearing (always carries P_xk regardless of what REQUIRED_ATTRS checks).
    _row_attrs_roller() below reads REQUIRED_ATTRS by name (imported as
    ROLLER_REQUIRED_ATTRS), so fixing the tuple upstream was enough --
    nothing in this file needed to change for this particular bug. See
    roller_bearing_solver.py's own module docstring for the full account.

  - FIX, this turn -- DOUBLE_BALL_DOUBLE_ROLLER_SCENARIO crashed the first
    time it was actually run:
      ValueError: ShaftSystem 'shaft1(motor)' validation failed:
        - shaft1(motor): bearing 'brg1b' [86.000, 114.000] mm overlaps
          shoulder at x=90.000 mm
    The illustrative double-row NU204 (DR_ROLLER_CATALOG, b=28.0 -- copied
    from the ball side's DR_CATALOG) was placed at the SAME position as the
    single-row NU204 (100.0 for shaft1), which was only ever tuned for a
    14mm-wide bearing (~3mm clearance to the shoulder at x=90); a 28mm-wide
    bearing there is nowhere near centered in shaft1's 30mm floating seat
    and crosses the shoulder by 4mm. (The locating side has the
    geometrically analogous situation -- b=28 double-row ball reusing the
    single-row's position=20 within a 30mm seat -- and does NOT crash;
    arrangement="locating" bearings are evidently not subject to the same
    strict shoulder-overlap check as arrangement="floating" ones, which
    makes physical sense: locating bearings are meant to seat against a
    shoulder, floating ones are not.) Fixed by lowering DR_ROLLER_CATALOG's
    b to 24.0 and adding DR_ROLLER_FLOATING_POSITIONS -- the floating
    bearing is now recentered on each shaft's own floating seat (105.0/
    132.5/132.5) instead of reusing the single-row default (100.0/125.0/
    125.0). build_systems() gained a floating_positions parameter (default
    None -> the original single-row positions, so scenarios 1-3 are
    unaffected) and SCENARIOS entries grew a 4th element for it. See the
    DR_ROLLER_CATALOG/DR_ROLLER_FLOATING_POSITIONS constants' own comments
    for the margin arithmetic per shaft.

    NOTE (superseded): the "locating bearings are evidently not subject to
    the same strict check" observation above turned out to be an artifact
    of a since-fixed core bug (Shaft.shoulders() only reporting
    shoulder_right) -- see the FIX entry at the top of this changelog. It
    is left here, unedited, as the historical record of what was actually
    observed at the time.

  - NEW SCENARIO, this turn -- DOUBLE_BALL_DOUBLE_ROLLER_SCENARIO: BOTH
    slots are now double-row simultaneously -- locating is the existing
    double-row 6204 (homogeneous rows, DR_ROW_GEOM_HOMO), floating is a
    NEW double-row NU204 (2 identical roller rows, DR_ROLLER_ROW_GEOM_HOMO)
    -- illustrative/synthetic, same "-illustrative" status as DR_CATALOG,
    no such catalog double-row roller bearing exists. Requested explicitly
    ("podes colocar mais um cenario em que e' double ball e double row?")
    specifically to exercise ISO16281MultiRowRollerSolverSharedDisplacement
    end-to-end for the FIRST time -- until now it had only been sanity-
    checked against a standalone toy reimplementation (see
    test_roller_shared_displacement_solver.py and that solver's own module
    docstring's "READ THIS BEFORE USING" section, which still applies: no
    REAL multi-row roller family exists, so this scenario's floating
    bearing is illustrative geometry run through real solver machinery,
    not a real part).

    This required generalizing several places that used to hardcode "the
    locating slot ('a') is the only possible multi-row slot":
      - DR_ROWS_BY_SCENARIO (locating-only) -> MULTIROW_SLOTS_BY_SCENARIO
        (scenario_name -> {"a": (...), "b": (...)}, either or both slots),
        now also carrying each slot's bearing_type so the main loop can
        dispatch formatting correctly.
      - _make_dr_multirow_view() gained bearing_type/arrangement params
        (were hardcoded to DEEP_GROOVE_BALL/"locating").
      - SCENARIOS entries are now (name, locating_builder, floating_builder)
        3-tuples (later widened to 4, see the FIX entry above); build_
        systems() takes floating_builder (default CRNU204) instead of
        hardcoding CRNU204(...) at every add_bearing() call.
      - The main loop's shaft-level block (bearings_for_solver, the
        capacity-omission catalog logic, the post-solve manual capacity
        patch) now loops over a dict of {label: (row_bearings, dr_rows,
        bearing_type)} instead of assuming at most one multirow label.
      - New DR_NU204() FEM-facing builder (mirrors DR6204()) and new
        _build_dr_rows_roller()/_row_attrs_roller() helpers (mirror
        _build_dr_rows()/_row_attrs(), reading ROLLER_REQUIRED_ATTRS
        instead of ball's REQUIRED_ATTRS).
      - New _format_double_row_result_roller(): roller's
        RollerLoadDistributionResult has no scalar-per-element field
        analogous to ball's delta_j/cp (one Hertzian point contact per
        ball) -- a roller instead carries n_s lamina forces (q_jk, shape
        (Z, n_s)), so per-roller total load is Q_j = sum_k q_jk (already
        computed by elements(), no extra combination needed the way the
        ball side needs cp*delta_j**1.5).

    Also fixed as a direct consequence: the final comparison table's print
    loop did `", ".join(f"{v:.4f}" for v in r["mr_f_a"])` unguarded, which
    would crash for a roller multi-row row -- RollerBearingResult.multirow()
    always sets f_a=None (radial rollers carry no axial load at all, so
    there is nothing to split). Now prints "n/a (no axial load -- radial
    roller)" when mr_f_a is None instead of iterating over it.

    plot_qmax_comparison()'s bar-color palette extended from 3 to 4 colors
    for the 4th scenario.

    NOT added: an analogous roller-side SOLVER COMPARISON section -- there
    is no second live roller multi-row solver to compare against (the
    fraction-based roller_bearing_multirow_solver.py is no longer
    registered and, per its own docstring, was itself built around the
    now-retired MultiRowCylindricalRollerFamily) -- see "Known gaps" below.

  - FIX, this turn -- print_solver_comparison() crashed with KeyError('Dw')
    the first time the heterogeneous scenario was actually run: dr_rows
    entries are built via REQUIRED_ATTRS (_row_attrs()), which does NOT
    include "Dw" (only A/alpha_0/phi_j/Ri/cp/Dpw/Z -- whatever the solvers
    themselves need) -- Dw only ever existed on the real per-row Bearing
    objects. solver_comparison entries now also carry dr_row_bearings, and
    the per-row header line reads Dw off THAT instead of off dr_rows.

    Also added: an explicit WARNING line printed whenever either solver's
    outer solve did not converge (outer_ok=False) for a given row/scenario,
    so a non-converged result can't be misread as a normal small numeric
    mismatch further down the table. This surfaced something real the
    first time the heterogeneous scenario ran end-to-end: with Fa=0 (every
    shaft in this script is purely radially loaded) AND heterogeneous rows,
    ISO16281MultiRowBallSolver's fraction-based outer solve failed to
    converge (ok=False, residual ~6e-3, hit its iteration cap) and reported
    f_a values like [113.0, -112.0] -- nonsensical as "fractions" of
    anything. Root cause, most likely: the same kind of degeneracy already
    known and handled for Fr~=0 (FR_NEGLIGIBLE_EPS, see
    ball_bearing_multirow_solver.py's own module docstring) but on the
    AXIAL side instead, and apparently unhandled there -- when Fa_total=0,
    f_a_row * Fa_total = 0 for ANY f_a_row, so the Jacobian column for the
    axial fraction unknown goes to zero/singular exactly like the radial
    one does at Fr~=0. This never showed up before because every previous
    scenario had either identical rows (so, by symmetry, no row-to-row
    axial force is needed even under misalignment) or nonzero Fa. It is
    NOT something introduced by this file or by
    ISO16281MultiRowBallSolverSharedDisplacement -- it's an existing gap in
    the production fraction-based solver, only now exercised for the first
    time by a case (heterogeneous rows + Fa=0 + nonzero misalignment psi)
    that genuinely needs an internal, self-equilibrating axial load split
    between rows even though the EXTERNAL Fa is zero.
    ISO16281MultiRowBallSolverSharedDisplacement should not share this
    specific degeneracy in principle -- it solves for the shared
    (delta_r, delta_a) directly from the sum of each row's OWN Hertzian
    reaction, never through a "fraction of a possibly-zero total" -- but
    that has not yet been confirmed against real output (the KeyError
    crashed the script before printing the shared-displacement numbers for
    this row). Re-run after this fix and read the WARNING lines / residual
    columns before trusting either solver's numbers for the heterogeneous
    scenario.

  - HETEROGENEOUS double-row scenario added
    (DOUBLE_ROW_HETERO_SCENARIO), specifically to stress-test the "each row
    carries its own cp" requirement the SOLVER COMPARISON section only
    exercised trivially before now (the original double-row scenario's two
    rows share identical geometry by construction, so f_r=f_a=0.5 for both
    solvers was the expected -- and not very informative -- outcome). The
    heterogeneous scenario's two rows differ ONLY in ball diameter Dw (row0
    keeps the standard 6204 ball, row1 uses a smaller one) -- same Dpw/Z/
    s/E/nu, so REQUIRED_ATTRS stays internally consistent (same pitch
    circle, same ball count, same material) while the two rows' Hertzian
    cp genuinely differs. Illustrative/synthetic, same spirit as
    DR_CATALOG's own "-illustrative" designation -- not a real catalog
    double-row bearing.

    This required threading `dr_rows` (and, for capacity, the underlying
    per-row Bearing objects) through as an explicit parameter everywhere
    that used to reach for the single module-level DR_ROWS global
    (_make_dr_multirow_view, _format_double_row_result,
    _solver_comparison_metrics) -- see DR_ROWS_BY_SCENARIO /
    DR_ROW_BEARINGS_BY_SCENARIO below. solver_comparison is now keyed by
    (scenario_name, shaft_name) instead of shaft_name alone, since both
    double-row scenarios now populate it and would otherwise collide.

    Also fixed, as a direct consequence of this: _format_double_row_result()
    used to report Q_ci/Q_ce as a SINGLE whole-bearing pair (patched from
    the row0-representative DR6204() Bearing, "numerically identical"
    across rows -- true only because the two rows WERE identical). That
    stops being correct once rows differ, so Q_ci/Q_ce in the printed table
    are now computed per row, directly from each row's own Bearing object
    (DR_ROW_BEARINGS_BY_SCENARIO[scenario_name][j]), the same per-row
    treatment Q_ei/Q_ee already had. bundle.capacity itself (used
    elsewhere, e.g. by whatever else reads the BearingResultsLibrary
    bundle) is UNCHANGED -- still the row0-representative single value,
    still patched in the same way; only this script's own printed table
    was updated to stop hiding the row1 difference.

  - SOLVER COMPARISON section. For the double-row
    scenario's locating bearing, the SAME (Fr_xz, Fr_xy, Fa, psi) problem
    that ISO16281MultiRowBallSolver already solved (via
    RollingBearingSolver's internal dispatch) is now ALSO solved directly
    with ISO16281MultiRowBallSolverSharedDisplacement, and the two
    results are printed side by side (delta_r/delta_a, per-row Q_max,
    f_r/f_a per row, outer-solve diagnostics). See
    _reconstruct_solver_inputs() and _solver_comparison_metrics() below,
    and the "SOLVER COMPARISON" print block in __main__.

    IMPORTANT caveat, stated once here rather than repeated at every call
    site: this script's Fr_xz/Fr_xy are NOT read directly off the FEM node
    (ShaftResultsReader only exposes the RESULTANT Fr and Fa here, see
    node.Fr/node.Fa throughout this file) -- they are RECONSTRUCTED from
    the already-solved fraction-based result's own phi_Fr
    (bundle.load_distribution[0].phi_Fr) via
    Fr_xz = Fr*cos(phi_Fr), Fr_xy = Fr*sin(phi_Fr). This is the exact
    inverse of phi_Fr = atan2(Fr_xy, Fr_xz), the same convention both
    solvers use internally (confirmed against
    ISO16281MultiRowBallSolverSharedDisplacement.solve_bearing()'s own
    `phi_Fr = np.arctan2(Fr_xy, Fr_xz)` line) -- so this reconstruction is
    an exact round-trip, not an approximation, PROVIDED
    RollingBearingSolver builds phi_Fr the same way internally (it is not
    re-derived independently here; it is read back from the already-
    produced result). psi is read the same way, straight off
    mr_result.psi. Because of this, the comparison is only as good as that
    one shared assumption -- if it turns out to not hold, the two solvers
    would be getting genuinely different inputs and any mismatch in
    outputs would not (only) reflect a difference in solver correctness.

    Also NOT independently verified end-to-end against the real axisforge
    package in this environment (that package's modules are not
    importable here) -- syntax was not re-checked beyond what each
    individual file was already checked against on its own turn. Run it
    for real before trusting the printed comparison numbers.

  - synced to the BallBearingResult/RollerBearingResult unification:
    REQUIRED_ATTRS (was _REQUIRED_ATTRS), families.ball/families.roller
    import paths, mr_result.outer_ok/.outer_n_iter/.outer_residual (was
    .ok/.n_iter/.residual), _contact_distribution() wraps a ball row in
    BallBearingResult(rows=[row]) before calling contact_distribution()
    (roller's own contact_distribution() still takes the raw row directly
    -- see roller_bearing_postprocessing.py's module docstring for why
    that side stays row-level).
  - capacity / Q_ei/Q_ee / Ka now computed via RollingBearingSolver.
    postprocess_and_record() for both single-row and double-row bearings,
    instead of ad hoc calls (_stiffness() removed). Two behavior changes:
    double-row Ka is now a real value (was hardcoded None), and double-row
    Q_ci/Q_ce is now a single whole-bearing value (was a per-row list of
    2, numerically identical). One manual step remains: capacity for the
    double-row label is patched in afterward via the real DR6204()
    Bearing, since the duck-typed multi-row view has no .family.
  - double-row scenario solved through RollingBearingSolver's own
    multi-row dispatch (_make_dr_multirow_view()) instead of a separate
    hand-rolled ISO16281MultiRowBallSolver call -- one solve() per shaft.
  - RollingBearingSolver.solve() returns a LIST per label (Option A) --
    every value indexed [0] for a single-row bearing.
  - Q_ci/Q_ce switched from bearing.family.dynamic_capacity() (whole-
    bearing Ca) to per_element_dynamic_capacity() (Q_ci/Q_ce) -- the
    correct per-element pairing for Q_ei/Q_ee.
  - Q_ei/Q_ee added, per row, via multirow_dynamic_equivalent_load()
    (generalized to any row count i, not hardcoded to 2).
  - illustrative AxialLoad on shaft3 removed -- was silently giving
    shaft3 Fa != 0 even in the "baseline" scenario.
  - thrust single-row scenario removed; the multi-row scenario switched
    from thrust (row-0-only probe) to a real 2-row radial 6204 double-row
    bearing, solved for real via ISO16281MultiRowBallSolver.
  - comparison table regrouped: scenario outer, shaft/slot inner.

Known gaps (not implemented here):
  - roller (line-contact) Q_ei/Q_ee -- LaminaDynamicEquivalentLoad exists
    now (roller_bearing_postprocessing.py), but it's per-lamina (q_kei/
    q_kee arrays, not a scalar per row like ball's Q_ei/Q_ee), so it
    doesn't fit this table's per-row-scalar format as-is -- left None,
    not wired in. Still true for the new double-row roller (floating)
    bearing -- its Q_ei/Q_ee are None in the printed table too, same as
    every other roller row in this script, single- or multi-row.
  - double-row "global" stiffness beyond row-0's representative Ka.
  - ISO 281's SIMPLIFIED L10 life (P = X*Fr + Y*Fa) -- still out of scope,
    and not the same quantity as L10r below (see DynamicEquivalentReference
    Load's own docstring for why Pref is not a drop-in substitute for X*Fr
    +Y*Fa). The RIGOROUS L10r/Pref (ball eq.29-31, roller eq.65-67) IS now
    computed for every row, both bearing types -- see _bearing_life()/
    _bearing_life_roller() and this module's changelog.
  - SOLVER COMPARISON section covers double-row BALL locating bearings
    only -- ISO16281MultiRowRollerSolverSharedDisplacement is now
    exercised for real (both double scenarios' floating slot), but there
    is no second, independently-implemented roller multi-row solver
    currently registered to compare it against (roller_bearing_multirow_
    solver.py, the fraction-based one, was itself built around the now-
    retired MultiRowCylindricalRollerFamily -- see that file's own
    docstring) -- its numbers are printed in the main comparison table
    like any other row, just not cross-checked against an alternative
    solver the way the ball side is (which now genuinely does cross-check
    two independent solvers again -- see this turn's changelog entry).
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.deep_groove import DeepGrooveBallFamily
from axisforge.core.machine_elements.bearings.families.roller_bearing.radial.subtypes.cylindrical_roller import CylindricalRollerFamily
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.loads import RadialLoad, TorqueLoad
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    ShaftResultsReader,
    SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.single_row_solver import (
    ISO16281BallSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.results import (
    BallBearingResult,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.multirow_solver import (
    ISO16281MultiRowBallSolverSharedDisplacement,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_multirow_solver import (
    ISO16281MultiRowBallSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.single_row_solver import (
    REQUIRED_ATTRS as ROLLER_REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.results import (
    RollerBearingResult,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    BearingResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.postprocessing import (
    contact_distribution as ball_contact_distribution,
    basic_reference_rating_life,
    DynamicEquivalentReferenceLoad,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.postprocessing import (
    contact_distribution as roller_contact_distribution,
    LaminaDynamicEquivalentLoad,
    basic_reference_rating_life as roller_basic_reference_rating_life,
    DynamicEquivalentReferenceLoad as RollerDynamicEquivalentReferenceLoad,
)
from axisforge.core.machine_elements.bearings.families.roller_bearing.radial.functions.capacity import (
    RollingElementCapacity,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.rolling_bearing_solver import (
    RollingBearingSolver,
)
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.shaft.mesh_generation.mesh_grade import Grader


# ===========================================================================
# PARAMETERS
# ===========================================================================

B_STUDY  = 20.0
P_W      = 1000.0
RPM_IN   = 450.0
r_pulley = 0.050          # [m]

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

SOLVER_TOL = 1.0e-6
GEAR_GRADE = "grade_3"

# Every locating bearing here is shaft-mounted (inner ring rotates, outer
# stationary) -- one fixed pair of flags for every bearing/row.
INNER_ROTATING = True
OUTER_ROTATING = False

# --- radial deep-groove ball, 6204 (d=20, D=47, B=14) ---
BB_GEOM = dict(Dw=7.94, Dpw=33.5, Z=8, s=0.010, E=206_000.0, nu=0.3)
CR_6204 = 12_700.0

# --- floating cylindrical roller, NU204 ---
CR_GEOM = dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40)
CR_NU204 = 28_500.0

# --- double-row deep-groove ball -- illustrative placeholder, 2 rows,
# tested against this system's real Fr. Two variants:
#   DR_ROW_GEOM_HOMO   -- both rows identical (same as BB_GEOM). Original
#                         scenario; f_r=f_a=0.5 for both solvers is the
#                         expected, uninformative-by-design outcome.
#   DR_ROW_GEOM_HETERO -- rows differ ONLY in ball diameter Dw (row0 keeps
#                         the standard 6204 ball, row1 uses a smaller one)
#                         -- same Dpw/Z/s/E/nu, so the two rows' Hertzian
#                         cp genuinely differs while everything else about
#                         the bearing stays a coherent, single-envelope
#                         6204-family double-row. Added specifically to
#                         stress-test "each row carries its own cp" with
#                         something other than the trivial identical-rows
#                         case -- see this module's changelog.
DR_ROW_GEOM_HOMO = [
    dict(Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
         E=BB_GEOM["E"], nu=BB_GEOM["nu"], s=BB_GEOM["s"])
    for _ in range(2)
]
DR_ROW_GEOM_HETERO = [
    dict(Dw=7.94, Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
         E=BB_GEOM["E"], nu=BB_GEOM["nu"], s=BB_GEOM["s"]),
    dict(Dw=6.35, Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
         E=BB_GEOM["E"], nu=BB_GEOM["nu"], s=BB_GEOM["s"]),
]
DR_CATALOG = dict(d=20.0, D=47.0, b=28.0, C=CR_6204, C0=6_550.0,
                  designation="DR6204-illustrative")

# --- double-row cylindrical roller -- illustrative placeholder, FLOATING
# side, mirrors DR_ROW_GEOM_HOMO/DR_ROW_GEOM_HETERO above. Two variants:
#   DR_ROLLER_ROW_GEOM_HOMO   -- both rows identical (same as CR_GEOM).
#   DR_ROLLER_ROW_GEOM_HETERO -- rows differ ONLY in roller diameter Dwe
#                         (row0 keeps CR_GEOM's standard 6.5mm, row1 uses a
#                         smaller 5.5mm) -- same Lwe/Dpw/Z/s/n_s, so
#                         ROLLER_REQUIRED_ATTRS stays internally consistent
#                         (same effective length, same pitch circle, same
#                         roller count) while the two rows' line-contact
#                         stiffness (cs) genuinely differs. Mirrors
#                         DR_ROW_GEOM_HETERO's own "differ in one diameter
#                         only" approach on the ball side -- added this
#                         turn so heterogeneity is exercised on BOTH sides
#                         of the same scenario at once, not just the ball
#                         locating side -- see this module's changelog.
DR_ROLLER_ROW_GEOM_HOMO = [
    dict(Dwe=CR_GEOM["Dwe"], Lwe=CR_GEOM["Lwe"], Dpw=CR_GEOM["Dpw"],
         Z=CR_GEOM["Z"], s=CR_GEOM["s"], n_s=CR_GEOM["n_s"])
    for _ in range(2)
]
DR_ROLLER_ROW_GEOM_HETERO = [
    dict(Dwe=CR_GEOM["Dwe"], Lwe=CR_GEOM["Lwe"], Dpw=CR_GEOM["Dpw"],
         Z=CR_GEOM["Z"], s=CR_GEOM["s"], n_s=CR_GEOM["n_s"]),
    dict(Dwe=5.5, Lwe=CR_GEOM["Lwe"], Dpw=CR_GEOM["Dpw"],
         Z=CR_GEOM["Z"], s=CR_GEOM["s"], n_s=CR_GEOM["n_s"]),
]
# FIX, this turn -- catalog width b was originally 28.0 (matching
# DR_CATALOG's own illustrative ball width) and the floating bearing was
# left at the SAME position as the single-row NU204 (100.0/125.0/125.0).
# That combination crashed the first real run:
#   ValueError: ShaftSystem 'shaft1(motor)' validation failed:
#     - shaft1(motor): bearing 'brg1b' [86.000, 114.000] mm overlaps
#       shoulder at x=90.000 mm
# Root cause: shaft1's floating seat (seatB, between the body/seatB
# shoulder at x=90 and the shaft end at x=120) is only 30mm long -- the
# single-row NU204 (b=14mm) fits it with ~3mm clearance to the shoulder at
# position=100 (100-7-90=3), but a 28mm-wide bearing at that SAME position
# is nowhere near centered in the 30mm seat (100 is 10mm from the left
# shoulder, only 20mm from the right end) and crosses the shoulder by 4mm.
# Fixed two ways together: b lowered to 24.0 (still clearly "double-row"
# vs. the single row's 14mm, but leaves enough seat length to recenter with
# margin -- see DR_ROLLER_FLOATING_POSITIONS below) AND the floating
# bearing is recentered on each shaft's own floating seat instead of
# reusing the single-row bearing's position. (shaft1's locating side has
# the analogous geometry -- b=28 double-row ball at the single-row's
# position=20 within a 30mm seatA -- and did NOT crash at the time; see the
# newest changelog entry at the top of this docstring for why that turned
# out to be an artifact of a since-fixed core bug, not a real difference
# between locating and floating arrangements.)
DR_ROLLER_CATALOG = dict(d=20.0, D=47.0, b=24.0, C=CR_NU204, C0=22_000.0,
                         designation="DR-NU204-illustrative")

# Recentered floating position per shaft, used ONLY for the double-row-
# roller scenario (single-row scenarios keep the original 100.0/125.0/125.0
# via build_systems()'s own default -- unaffected). Computed as the midpoint
# of each shaft's floating seat (seatB, between the body/seatB shoulder and
# the shaft's free end) -- see make_stepped_shaft()/build_systems() for the
# underlying shaft geometry:
#   shaft1: seatB = [90, 120]  (l_seat_b=30) -> center 105.0, margin
#           (30-24)/2 = 3.0mm each side at b=24 -- matches the ~3mm
#           clearance convention the single-row default already uses.
#   shaft2: seatB = [115, 150] (l_seat_b=35) -> center 132.5, margin 5.5mm.
#   shaft3: seatB = [115, 150] (l_seat_b=35, same shaft params as shaft2)
#           -> center 132.5, margin 5.5mm.
DR_ROLLER_FLOATING_POSITIONS = (105.0, 132.5, 132.5)

# Recentered locating position per shaft, used ONLY for the double-row-ball
# scenarios (single-row scenarios keep the original 20.0/25.0/25.0 via
# build_systems()'s own default -- unaffected). Computed as the midpoint of
# each shaft's locating seat (seatA, [0, l_seat_a]) -- mirrors
# DR_ROLLER_FLOATING_POSITIONS' own reasoning on the floating side. This
# only became necessary once Shaft.shoulders() (core) was fixed to report
# every transition instead of only the one that used to live on
# shoulder_right -- before that fix, the seatA/body shoulder was invisible
# to ShaftSystem.validate_or_raise(), so a b=28 double-row locating bearing
# centered at position=20 within a 30/35mm seatA never tripped the overlap
# check even though it genuinely doesn't fit there. Now it does, correctly
# -- see this module's changelog, newest entry.
#   shaft1: seatA = [0, 30]  -> center 15.0, margin (30-28)/2 = 1.0mm each side
#   shaft2: seatA = [0, 35]  -> center 17.5, margin 3.5mm
#   shaft3: seatA = [0, 35]  -> center 17.5, margin 3.5mm
DR_LOCATING_POSITIONS = (15.0, 17.5, 17.5)


# ===========================================================================
# BEARING BUILDERS -- one per bearing kind, each returning an assembled
# Bearing via Bearing.assemble(). Locating builders are swapped per
# scenario; CRNU204 (floating) is constant across every scenario.
# ===========================================================================

def BB6204(position, label) -> Bearing:
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(d=20.0, D=47.0, designation="6204", b=14.0,
                               C=CR_6204, C0=6_550.0, arrangement="locating",
                               label=label, position=position),
        geometry=dict(Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
                      E=BB_GEOM["E"], nu=BB_GEOM["nu"], s=BB_GEOM["s"]),
        analyses={"point_contact": True},
    )


def CRNU204(position, label) -> Bearing:
    return Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(d=20.0, D=47.0, designation="NU204", b=14.0,
                               C=CR_NU204, C0=22_000.0, arrangement="floating",
                               label=label, position=position),
        geometry=dict(Dwe=CR_GEOM["Dwe"], Lwe=CR_GEOM["Lwe"], Dpw=CR_GEOM["Dpw"],
                      Z=CR_GEOM["Z"], s=CR_GEOM["s"], n_s=CR_GEOM["n_s"]),
        analyses={"line_contact": True},
    )


# DR_ROW_BEARINGS -- the 2 rows' full Bearing objects (position-independent,
# built once), so per_element_dynamic_capacity() can be called on each row
# directly rather than guessed at via a stripped SimpleNamespace.
def _row_attrs(b: Bearing) -> dict:
    return {a: getattr(b, a) for a in REQUIRED_ATTRS}


def _build_dr_rows(row_geoms: list[dict], label_prefix: str) -> tuple[list[Bearing], list[dict]]:
    """
    (dr_row_bearings, dr_rows) for a given list of per-row geometry dicts --
    dr_row_bearings are the real per-row Bearing objects (used for
    per-row capacity), dr_rows are their REQUIRED_ATTRS dicts (used by
    both multi-row solvers, `cp` included).
    """
    bearings = [
        Bearing.assemble(
            family=DeepGrooveBallFamily(),
            catalog=BearingCatalog(arrangement="locating", label=f"{label_prefix}-row{j}",
                                   position=0.0, **DR_CATALOG),
            geometry=geom,
            analyses={"point_contact": True},
        )
        for j, geom in enumerate(row_geoms)
    ]
    return bearings, [_row_attrs(b) for b in bearings]


DR_ROW_BEARINGS_HOMO,   DR_ROWS_HOMO   = _build_dr_rows(DR_ROW_GEOM_HOMO,   "dr-homo")
DR_ROW_BEARINGS_HETERO, DR_ROWS_HETERO = _build_dr_rows(DR_ROW_GEOM_HETERO, "dr-hetero")


def _row_attrs_roller(b: Bearing) -> dict:
    return {a: getattr(b, a) for a in ROLLER_REQUIRED_ATTRS}


def _build_dr_rows_roller(row_geoms: list[dict], label_prefix: str) -> tuple[list[Bearing], list[dict]]:
    """
    Roller counterpart of _build_dr_rows() -- (dr_row_bearings, dr_rows) for
    a double-row cylindrical roller bearing. Same illustrative/synthetic
    status as DR_CATALOG (no such catalog part exists); built specifically
    to exercise ISO16281MultiRowRollerSolverSharedDisplacement end-to-end
    for the first time -- see this module's changelog and that solver's
    own module docstring for the "not validated against a real bearing"
    caveat, which still applies to this scenario's data, not just the
    solver code.
    """
    bearings = [
        Bearing.assemble(
            family=CylindricalRollerFamily(),
            catalog=BearingCatalog(arrangement="floating", label=f"{label_prefix}-row{j}",
                                   position=0.0, **DR_ROLLER_CATALOG),
            geometry=geom,
            analyses={"line_contact": True},
        )
        for j, geom in enumerate(row_geoms)
    ]
    return bearings, [_row_attrs_roller(b) for b in bearings]


DR_ROLLER_ROW_BEARINGS_HOMO,   DR_ROLLER_ROWS_HOMO   = _build_dr_rows_roller(
    DR_ROLLER_ROW_GEOM_HOMO,   "dr-roller-homo")
DR_ROLLER_ROW_BEARINGS_HETERO, DR_ROLLER_ROWS_HETERO = _build_dr_rows_roller(
    DR_ROLLER_ROW_GEOM_HETERO, "dr-roller-hetero")


def DR6204(position, label, row_geom: dict) -> Bearing:
    """
    Double-row 6204's FEM-facing Bearing -- single-row-shaped (one node;
    FEM has no notion of rows). Uses `row_geom` (row 0's geometry, by
    convention -- bound via functools.partial in SCENARIOS below) as the
    REPRESENTATIVE single geometry for the FEM node; NOT used for capacity
    reporting any more (that's now genuinely per-row, see
    _format_double_row_result()). The solver does NOT see this object for
    the double-row scenario's locating slot -- it's handed a separate
    duck-typed multi-row view instead, with BOTH rows' real geometry, see
    _make_dr_multirow_view().
    """
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(arrangement="locating", label=label,
                               position=position, **DR_CATALOG),
        geometry=row_geom,
        analyses={"point_contact": True},
    )


def DR_NU204(position, label, row_geom: dict) -> Bearing:
    """
    Double-row NU204's FEM-facing Bearing -- roller counterpart of DR6204(),
    same convention: single-row-shaped (FEM has no notion of rows), row_geom
    is row 0's geometry (bound via functools.partial in SCENARIOS below).
    The solver does NOT see this object for a double-row-roller scenario's
    floating slot -- swapped for a duck-typed multi-row view instead, same
    as the locating side -- see _make_dr_multirow_view().
    """
    return Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(arrangement="floating", label=label,
                               position=position, **DR_ROLLER_CATALOG),
        geometry=row_geom,
        analyses={"line_contact": True},
    )


# ===========================================================================
# SCENARIOS -- (name, locating_builder, floating_builder, floating_positions,
# locating_positions). Exactly 3 scenarios -- see this module's changelog
# ("RESTRUCTURED, this turn") for why the earlier 4 (which included two
# double-ball-only variants with a single-row floating side) collapsed to
# these 3:
#   1. BASELINE_SCENARIO      -- single-row both sides.
#   2. DOUBLE_HOMO_SCENARIO   -- double-row both sides, IDENTICAL rows.
#   3. DOUBLE_HETERO_SCENARIO -- double-row both sides, HETEROGENEOUS rows
#                                 (ball rows differ in Dw, roller rows in
#                                 Dwe -- independently of each other).
# ===========================================================================

BASELINE_SCENARIO      = "single-row locating (6204) + single-row floating roller (NU204)  [baseline]"
DOUBLE_HOMO_SCENARIO   = "DOUBLE-row locating (2x 6204-row) + DOUBLE-row floating roller (2x NU204-row)  [REAL multi-row solve, BOTH slots, IDENTICAL rows]"
DOUBLE_HETERO_SCENARIO = "DOUBLE-row locating (2x 6204-row) + DOUBLE-row floating roller (2x NU204-row)  [REAL multi-row solve, BOTH slots, HETEROGENEOUS rows]"

# scenario_name -> {slot_suffix: (dr_row_bearings, dr_rows, bearing_type)},
# looked up by scenario_name in the main loop below -- slot_suffix is "a"
# (locating) or "b" (floating), matching the label suffix build_systems()
# already uses (brgNa/brgNb). BASELINE_SCENARIO has no entry (single-row
# both slots, nothing multi-row); the two double scenarios populate BOTH
# slots -- the main loop does not assume at most one multirow label per
# shaft. Was DR_ROWS_BY_SCENARIO (locating-only, no bearing_type), then
# briefly a locating+floating-but-mixed-width version -- see this module's
# changelog for both restructurings.
MULTIROW_SLOTS_BY_SCENARIO: dict[str, dict[str, tuple[list[Bearing], list[dict], BearingType]]] = {
    DOUBLE_HOMO_SCENARIO: {
        "a": (DR_ROW_BEARINGS_HOMO, DR_ROWS_HOMO, BearingType.DEEP_GROOVE_BALL),
        "b": (DR_ROLLER_ROW_BEARINGS_HOMO, DR_ROLLER_ROWS_HOMO, BearingType.CYLINDRICAL_ROLLER),
    },
    DOUBLE_HETERO_SCENARIO: {
        "a": (DR_ROW_BEARINGS_HETERO, DR_ROWS_HETERO, BearingType.DEEP_GROOVE_BALL),
        "b": (DR_ROLLER_ROW_BEARINGS_HETERO, DR_ROLLER_ROWS_HETERO, BearingType.CYLINDRICAL_ROLLER),
    },
}

# floating_positions/locating_positions are both None for BASELINE_SCENARIO
# (keeps build_systems()'s own single-row defaults, 100.0/125.0/125.0 and
# 20.0/25.0/25.0); all three double scenarios reuse the SAME
# DR_ROLLER_FLOATING_POSITIONS / DR_LOCATING_POSITIONS -- the recentering
# only depends on the mounting envelope (DR_ROLLER_CATALOG's / DR_CATALOG's
# b/d/D), not on whether the individual rows are homogeneous or
# heterogeneous.
SCENARIOS = [
    (BASELINE_SCENARIO, BB6204, CRNU204, None, None),
    (DOUBLE_HOMO_SCENARIO,
     partial(DR6204,   row_geom=DR_ROW_GEOM_HOMO[0]),
     partial(DR_NU204, row_geom=DR_ROLLER_ROW_GEOM_HOMO[0]),
     DR_ROLLER_FLOATING_POSITIONS, DR_LOCATING_POSITIONS),
    (DOUBLE_HETERO_SCENARIO,
     partial(DR6204,   row_geom=DR_ROW_GEOM_HETERO[0]),
     partial(DR_NU204, row_geom=DR_ROLLER_ROW_GEOM_HETERO[0]),
     DR_ROLLER_FLOATING_POSITIONS, DR_LOCATING_POSITIONS),
]


# ===========================================================================
# BUILD HELPERS
# ===========================================================================

def make_stepped_shaft(name, total_length, d_seat, d_body,
                       l_seat_a, l_seat_b, fillet_r, material_id="AISI_1045"):
    l_body = total_length - l_seat_a - l_seat_b
    sh = Shaft(label=name)
    sh.add_section(ShaftSection(length=l_seat_a, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatA"))
    sh.add_section(ShaftSection(length=l_body, diameter=d_body,
                                material_id=material_id, label=f"{name}-body"))
    sh.add_section(ShaftSection(length=l_seat_b, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatB"))

    shoulder = Shoulder(fillet_radius=fillet_r, diameter_large=d_body, diameter_small=d_seat)
    sh.set_transition(0, shoulder)   # seatA / body
    sh.set_transition(1, shoulder)   # body / seatB

    return sh

def build_systems(b: float, locating_builder, floating_builder=CRNU204,
                  floating_positions: tuple[float, float, float] | None = None,
                  locating_positions: tuple[float, float, float] | None = None) -> dict[str, ShaftSystem]:
    """
    floating_positions -- (pos_shaft1, pos_shaft2, pos_shaft3) for the
    FLOATING bearing, defaulting to the original (100.0, 125.0, 125.0) --
    tuned for the single-row NU204's own width. A wider floating_builder
    (e.g. the double-row DR_NU204) needs its own, recentered positions --
    see DR_ROLLER_FLOATING_POSITIONS and the FIX changelog entry explaining
    why (a shoulder-overlap crash on shaft1 the first time this was run
    with an unchanged position).

    locating_positions -- same idea, LOCATING side: (pos_shaft1, pos_shaft2,
    pos_shaft3), defaulting to the original (20.0, 25.0, 25.0) -- tuned for
    the single-row 6204's own width. A wider locating_builder (e.g. the
    double-row DR6204) needs its own, recentered positions -- see
    DR_LOCATING_POSITIONS. Only needed once Shaft.shoulders() (core) was
    fixed to report every transition instead of only the one that used to
    live on shoulder_right -- before that fix, ShaftSystem.validate_or_raise()
    never saw the seatA/body shoulder at all, so this crash was silently
    missed. See this module's changelog, newest entry.
    """
    gear_kw = dict(**GEAR_KW_BASE, b=b)
    pos1, pos2, pos3    = floating_positions  if floating_positions  is not None else (100.0, 125.0, 125.0)
    pos1a, pos2a, pos3a = locating_positions if locating_positions is not None else (20.0, 25.0, 25.0)

    z1 = SpurHelicalGear(z=20, position=55.0, label="z1", **gear_kw)
    z2 = SpurHelicalGear(z=60, position=95.0, label="z2", **gear_kw)
    z3 = SpurHelicalGear(z=20, position=55.0, label="z3", **gear_kw)
    z4 = SpurHelicalGear(z=60, position=55.0, label="z4", **gear_kw)

    stage1 = SpurHelicalGearMeshing(z1, z2, label="stage1")
    stage2 = SpurHelicalGearMeshing(z3, z4, label="stage2")

    s1 = make_stepped_shaft("shaft1", 120.0, 20.0, 25.0, 30.0, 30.0, 1.5)
    s2 = make_stepped_shaft("shaft2", 150.0, 20.0, 28.0, 35.0, 35.0, 2.0)
    s3 = make_stepped_shaft("shaft3", 150.0, 20.0, 25.0, 35.0, 35.0, 1.5)

    sys1 = ShaftSystem(s1, name="shaft1(motor)", speed_rpm=450.0)
    sys2 = ShaftSystem(s2, name="shaft2",        speed_rpm=150.0)
    sys3 = ShaftSystem(s3, name="shaft3(pulley)", speed_rpm=50.0)

    sys1.shaft_origin_x = 0.0
    sys2.shaft_origin_x = sys1.shaft_origin_x + (z1.position - z2.position)
    sys3.shaft_origin_x = sys2.shaft_origin_x + (z3.position - z4.position)

    sys1.add_bearing(locating_builder(pos1a, "brg1a")); sys1.add_bearing(floating_builder(pos1, "brg1b"))
    sys2.add_bearing(locating_builder(pos2a, "brg2a")); sys2.add_bearing(floating_builder(pos2, "brg2b"))
    sys3.add_bearing(locating_builder(pos3a, "brg3a")); sys3.add_bearing(floating_builder(pos3, "brg3b"))

    ge_z1 = GearElement(z1, role="driver", rotation_dir=1, label="z1")
    ge_z2 = GearElement(z2, role="driven",                 label="z2")
    ge_z3 = GearElement(z3, role="driver",                 label="z3")
    ge_z4 = GearElement(z4, role="driven",                 label="z4")

    sys1.add_gear(ge_z1)
    sys2.add_gear(ge_z2); sys2.add_gear(ge_z3)
    sys3.add_gear(ge_z4)

    link1 = SpurHelicalMeshLink(sys1, ge_z1, sys2, ge_z2, stage1,
                                phi_deg=270.0, distribute_loads=True, label="stage1")
    link2 = SpurHelicalMeshLink(sys2, ge_z3, sys3, ge_z4, stage2,
                                phi_deg=270.0, distribute_loads=True, label="stage2")

    gearbox = SpurHelicalGearSystem([sys1, sys2, sys3], [link1, link2],
                                    label="load-lifting-bearing-comparison")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    T3      = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope  = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))
    # Fa=0 on every shaft here, same as design_load_distribution_mixed_bearings.py.

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    return {sh.name: sh for sh in (sys1, sys2, sys3)}


def gear_grade_nodes(shaft_sys: ShaftSystem, grade: str = GEAR_GRADE) -> list[float]:
    base_nodes = Mesh1D(shaft_sys).x_nodes
    extra: list[float] = []
    for ge in shaft_sys.gears:
        lo, hi = shaft_sys.gear_extent(ge)
        extra += Grader(lo, hi, base_nodes).get_grade(grade)
    return extra


def _contact_distribution(b_obj, res):
    """
    res is bundle.load_distribution[0] -- a bare per-row result. Both
    ball's and roller's contact_distribution() now take the
    BallBearingResult/RollerBearingResult wrapper and return a list (one
    per row) -- wrap res in a length-1 container and unwrap [0] either way.
    """
    if b_obj.bearing_type == BearingType.CYLINDRICAL_ROLLER:
        return roller_contact_distribution(b_obj, RollerBearingResult(rows=[res]))[0]
    return ball_contact_distribution(b_obj, BallBearingResult(rows=[res]))[0]


def _capacity_Cr(b_obj: Bearing) -> float:
    """Cr forwarded to per_element_dynamic_capacity() -- CR_6204 or CR_NU204 by bearing type."""
    return CR_NU204 if b_obj.bearing_type == BearingType.CYLINDRICAL_ROLLER else CR_6204


def _per_element_capacity(b_obj: Bearing) -> tuple[float, float]:
    """(Q_ci, Q_ce) -- per-element dynamic capacity, ISO/TS 16281 S4.3.1.2 (ball) / S5.3.1.2 (roller)."""
    return b_obj.family.per_element_dynamic_capacity(b_obj, Cr=_capacity_Cr(b_obj))


def _bearing_life(label: str,
                  Q_ci_rows: list[float], Q_ei_rows: list[float] | None,
                  Q_ce_rows: list[float], Q_ee_rows: list[float] | None,
                  Cr: float) -> dict | None:
    """L10r/Pref_r via ball_bearing_postprocessing.py (eq.29-31). None for roller (Q_ei_rows is None)."""
    if Q_ei_rows is None or Q_ee_rows is None:
        return None
    per_row, L10r_bearing = basic_reference_rating_life(
        label, Q_ci_rows, Q_ei_rows, Q_ce_rows, Q_ee_rows,
    )
    pref = DynamicEquivalentReferenceLoad.from_L10r(label, L10r_bearing, Cr=Cr)
    return dict(
        L10r_rows=[p.L10r for p in per_row],
        L10r=L10r_bearing,
        Pref_r=pref.Pref_r,
    )


def _bearing_life_roller(label: str, row_bearings: list[Bearing],
                         row_results: list, Cr: float,
                         inner_rotating: bool = INNER_ROTATING,
                         outer_rotating: bool = OUTER_ROTATING) -> dict:
    """
    L10r/Pref_r for roller rows -- eq.(65)-(67). q_kci/q_kce come from
    RollingElementCapacity.per_lamina() (uniform across k -- eq.56-57 gives
    ONE value, broadcast to n_s here), starting from the same whole-roller
    Q_ci/Q_ce this script already computes via _per_element_capacity().
    q_kei/q_kee come from LaminaDynamicEquivalentLoad.from_distribution(),
    called once per row directly (not from_bearing_result()'s multi-row
    branch, which roller_bearing_postprocessing.py's own docstring flags as
    unexercised/untested).
    """
    q_kci_rows, q_kei_rows, q_kce_rows, q_kee_rows = [], [], [], []
    for row_bearing, row_res in zip(row_bearings, row_results):
        n_s = row_bearing.n_s
        Q_ci, Q_ce = _per_element_capacity(row_bearing)
        q_ci, q_ce = RollingElementCapacity.per_lamina(Q_ci, Q_ce, n_s)
        q_kci_rows.append(np.full(n_s, q_ci))
        q_kce_rows.append(np.full(n_s, q_ce))
        eq = LaminaDynamicEquivalentLoad.from_distribution(
            row_bearing, row_res,
            inner_rotating=inner_rotating, outer_rotating=outer_rotating,
        )
        q_kei_rows.append(eq.q_kei)
        q_kee_rows.append(eq.q_kee)

    per_row, L10r_bearing = roller_basic_reference_rating_life(
        label, q_kci_rows, q_kei_rows, q_kce_rows, q_kee_rows,
    )
    pref = RollerDynamicEquivalentReferenceLoad.from_L10r(label, L10r_bearing, Cr=Cr)
    return dict(
        L10r_rows=[p.L10r for p in per_row],
        L10r=L10r_bearing,
        Pref_r=pref.Pref_r,
    )


def _make_dr_multirow_view(label: str, dr_rows: list[dict],
                           bearing_type: BearingType, arrangement: str) -> SimpleNamespace:
    """
    Duck-typed multi-row stand-in for a double-row scenario's locating or
    floating slot. Used both for RollingBearingSolver (.rows/.i/
    .bearing_type/.arrangement/.label are all it reads) AND, for the ball
    case, directly as the `bearing` argument to
    ISO16281MultiRowBallSolverSharedDisplacement.solve_bearing() (which
    only reads .rows off it) -- same object, same shape, both solvers are
    handed an identical bearing description. The FEM still sees the real
    single-row DR6204()/DR_NU204() Bearing.

    `dr_rows`/`bearing_type`/`arrangement` are all explicit parameters now
    (bearing_type/arrangement used to be hardcoded to DEEP_GROOVE_BALL/
    "locating") -- there is now more than one row-geometry set AND more
    than one bearing kind in play (MULTIROW_SLOTS_BY_SCENARIO), and either
    slot ("a"/locating or "b"/floating) can be the multi-row one, so the
    caller must say which view this is for.
    """
    return SimpleNamespace(
        rows=dr_rows, i=len(dr_rows), label=label,
        bearing_type=bearing_type, arrangement=arrangement,
    )


def _format_double_row_result(node, bundle, mr_result,
                              dr_rows: list[dict], dr_row_bearings: list[Bearing]) -> dict:
    """
    Formats an already-solved, already-postprocessed double-row bundle into
    the same dict shape the print loop / plots expect. Pure report
    formatting: Q_ei/Q_ee/Ka come from `bundle` (populated by
    RollingBearingSolver.postprocess_and_record()); Q_j/phi_global/Q_max/
    n_loaded/dist_traces (this script's own table/plot fields) and, this
    turn, Q_ci/Q_ce are computed here. `mr_result`
    (bundle.extra["multirow_result"]) is now a BallBearingResult (via
    .multirow()) -- its outer-solve diagnostics are named outer_ok/
    outer_n_iter/outer_residual, not ok/n_iter/residual.

    `dr_rows`/`dr_row_bearings` are explicit parameters now (were module-
    level DR_ROWS/DR_ROW_BEARINGS globals) -- see _make_dr_multirow_view()'s
    docstring for why.
    """
    row_results = bundle.load_distribution
    row_eq      = bundle.dynamic_equivalent_load

    # Q_ci/Q_ce, genuinely PER ROW -- computed directly from each row's own
    # Bearing object, not from bundle.capacity (which stays a single
    # row0-representative value, patched separately for whatever else
    # reads it off the bundle -- see this module's changelog). This used to
    # be "a single whole-bearing pair, wrapped in a length-1 list, numerically
    # identical across rows" -- that was only ever true because the two rows
    # WERE identical (DOUBLE_ROW_SCENARIO). It stops being correct for
    # DOUBLE_ROW_HETERO_SCENARIO, so it's computed per row here instead.
    Q_ci = []
    Q_ce = []
    for row_bearing in dr_row_bearings:
        qci, qce = _per_element_capacity(row_bearing)
        Q_ci.append(qci)
        Q_ce.append(qce)

    phi_Fr = row_results[0].phi_Fr

    Q_max       = 0.0
    n_loaded    = 0
    Z_total     = 0
    dist_traces = []
    for j, (row_geom, row_res) in enumerate(zip(dr_rows, row_results)):
        Q_j         = row_geom["cp"] * row_res.delta_j ** 1.5
        phi_global  = (row_geom["phi_j"] + phi_Fr) % (2.0 * np.pi)
        Q_max       = max(Q_max, float(Q_j.max()))
        n_loaded   += int((Q_j > 0).sum())
        Z_total    += int(row_geom["Z"])
        dist_traces.append((phi_global, Q_j, f"row{j}"))

    # L10r/Pref_r -- eq.(29)-(31), per row + combined.
    Q_ei_rows = [e.Q_ei for e in row_eq]
    Q_ee_rows = [e.Q_ee for e in row_eq]
    life = _bearing_life(node.label, Q_ci, Q_ei_rows, Q_ce, Q_ee_rows,
                         Cr=_capacity_Cr(dr_row_bearings[0]))

    return dict(
        kind="deep_groove_ball_double_row [REAL 2-row solve, via RollingBearingSolver]",
        Fr=node.Fr, Fa=node.Fa,
        Q_max=Q_max, n_loaded=n_loaded, Z=Z_total,
        Ka=bundle.stiffness.Ka, ok=mr_result.outer_ok,
        delta_r=mr_result.delta_r, delta_a=mr_result.delta_a, psi=mr_result.psi,
        dist_traces=dist_traces,
        # outer (multi-row) load-split convergence diagnostics -- None for
        # an ordinary single-row bearing (no outer solve to report).
        mr_n_iter=mr_result.outer_n_iter, mr_residual=mr_result.outer_residual,
        mr_f_r=mr_result.f_r, mr_f_a=mr_result.f_a,
        Q_ei=Q_ei_rows, Q_ee=Q_ee_rows,
        Q_ci=Q_ci, Q_ce=Q_ce,
        L10r_rows=life["L10r_rows"], L10r=life["L10r"], Pref_r=life["Pref_r"],
    )


def _format_double_row_result_roller(node, bundle, mr_result,
                                     dr_rows: list[dict], dr_row_bearings: list[Bearing]) -> dict:
    """
    Roller counterpart of _format_double_row_result() -- same report-
    formatting role, adapted for line contact. `mr_result` is a
    RollerBearingResult (via .multirow()); its rows are
    RollerLoadDistributionResult, which has no scalar-per-element field
    analogous to ball's delta_j/cp (ball: Q_j = cp * delta_j**1.5, ONE
    Hertzian point contact per ball). A roller instead carries n_s lamina
    forces (q_jk, shape (Z, n_s)) -- the per-roller total load is their sum
    over laminae, Q_j = sum_k q_jk (shape (Z,)), read straight off q_jk
    (already computed by elements()/solve_contact(), eq.(36) -- no extra
    cp/delta_j combination needed the way the ball side needs one).

    mr_result.f_a is always None here (radial rollers carry no axial load,
    see RollerBearingResult.multirow()'s own docstring) and delta_a is
    always 0.0 on every row (roller_bearing_solver.py never solves for it).
    Q_ei/Q_ee stay None, same as every other roller row in this script --
    see this module's "Known gaps" docstring entry.
    """
    row_results = bundle.load_distribution

    Q_ci = []
    Q_ce = []
    for row_bearing in dr_row_bearings:
        qci, qce = _per_element_capacity(row_bearing)
        Q_ci.append(qci)
        Q_ce.append(qce)

    phi_Fr = row_results[0].phi_Fr

    Q_max       = 0.0
    n_loaded    = 0
    Z_total     = 0
    dist_traces = []
    for j, (row_geom, row_res) in enumerate(zip(dr_rows, row_results)):
        Q_j         = np.sum(row_res.q_jk, axis=1)   # per-roller total, eq.(36) summed over laminae
        phi_global  = (row_geom["phi_j"] + phi_Fr) % (2.0 * np.pi)
        Q_max       = max(Q_max, float(Q_j.max()))
        n_loaded   += int((Q_j > 0).sum())
        Z_total    += int(row_geom["Z"])
        dist_traces.append((phi_global, Q_j, f"row{j}"))

    life = _bearing_life_roller(node.label, dr_row_bearings, row_results,
                                Cr=_capacity_Cr(dr_row_bearings[0]))

    return dict(
        kind="cylindrical_roller_double_row [REAL 2-row solve, via RollingBearingSolver -- ILLUSTRATIVE, see module docstring]",
        Fr=node.Fr, Fa=node.Fa,
        Q_max=Q_max, n_loaded=n_loaded, Z=Z_total,
        Ka=bundle.stiffness.Ka, ok=mr_result.outer_ok,
        delta_r=mr_result.delta_r, delta_a=mr_result.delta_a, psi=mr_result.psi,
        dist_traces=dist_traces,
        mr_n_iter=mr_result.outer_n_iter, mr_residual=mr_result.outer_residual,
        mr_f_r=mr_result.f_r, mr_f_a=mr_result.f_a,
        Q_ei=None, Q_ee=None,
        Q_ci=Q_ci, Q_ce=Q_ce,
        L10r_rows=life["L10r_rows"], L10r=life["L10r"], Pref_r=life["Pref_r"],
    )


# ===========================================================================
# SOLVER COMPARISON -- fraction-based (ISO16281MultiRowBallSolver, already
# solved via RollingBearingSolver) vs. shared-displacement
# (ISO16281MultiRowBallSolverSharedDisplacement, called directly here).
# Both solve the SAME physical problem under the SAME co-located-rows
# idealization -- if both are implemented correctly they should agree to
# solver tolerance; a mismatch beyond that is a signal to go check one of
# the two, not an expected/acceptable discrepancy.
# ===========================================================================

def _reconstruct_solver_inputs(node, bundle, mr_result) -> tuple[float, float, float, float]:
    """
    (Fr_xz, Fr_xy, Fa, psi) -- reconstructed from the ALREADY-SOLVED
    fraction-based result so the shared-displacement solver is handed an
    IDENTICAL problem, not a re-derived (and possibly inconsistent) one.
    See this module's docstring changelog entry for the exact-round-trip
    argument and its one load-bearing assumption.
    """
    phi_Fr = bundle.load_distribution[0].phi_Fr
    Fr_xz  = node.Fr * float(np.cos(phi_Fr))
    Fr_xy  = node.Fr * float(np.sin(phi_Fr))
    return Fr_xz, Fr_xy, node.Fa, mr_result.psi


def _solver_comparison_metrics(mr, dr_rows: list[dict]) -> dict:
    """
    Q_max per row (+ delta_r/delta_a/f_r/f_a/outer diagnostics), read off a
    BallBearingResult -- same shape whichever of the two solvers produced
    it (both return via .multirow()). Mirrors the per-row Q_j computation
    _format_double_row_result() already does for the fraction-based result,
    applied here to either solver's output so the two are read identically.

    `dr_rows` is now an explicit parameter (was the module-level DR_ROWS
    global) -- must be the SAME row-geometry set (dr_rows[j]["cp"]) that
    was actually handed to whichever solver produced `mr`, or the reported
    Q_max would be computed with the wrong row's cp.
    """
    Q_max_rows = []
    for row_geom, row_res in zip(dr_rows, mr.rows):
        Q_j = row_geom["cp"] * row_res.delta_j ** 1.5
        Q_max_rows.append(float(Q_j.max()))
    return dict(
        delta_r=mr.delta_r, delta_a=mr.delta_a, psi=mr.psi,
        Q_max_rows=Q_max_rows,
        f_r=list(mr.f_r), f_a=list(mr.f_a),
        ok=mr.outer_ok, n_iter=mr.outer_n_iter, residual=mr.outer_residual,
    )


def print_solver_comparison(solver_comparison: dict[tuple[str, str], dict]) -> None:
    """
    solver_comparison is keyed by (scenario_name, shaft_name) -- both
    double-row scenarios (homogeneous and heterogeneous) populate it now,
    so a bare shaft_name key would collide between the two.
    """
    print(f"\n{'='*W}")
    print("  SOLVER COMPARISON -- double-row locating bearing")
    print("  fraction-based (ISO16281MultiRowBallSolver)  vs.")
    print("  shared-displacement (ISO16281MultiRowBallSolverSharedDisplacement)")
    print(f"{'='*W}")

    for (scenario_name, shaft_name), entry in sorted(solver_comparison.items()):
        dr_rows         = entry["dr_rows"]
        dr_row_bearings = entry["dr_row_bearings"]
        m_frac   = _solver_comparison_metrics(entry["fraction"], dr_rows)
        m_shared = _solver_comparison_metrics(entry["shared"], dr_rows)

        print(f"\n-- {shaft_name}  |  {scenario_name} "
              f"{'-'*max(W-8-len(shaft_name)-len(scenario_name), 4)}")
        print(f"      Fr={entry['Fr']:>9.1f} N   Fa={entry['Fa']:>9.1f} N   "
              f"phi_Fr={np.degrees(entry['phi_Fr']):>8.3f} deg")
        # Dw is NOT in REQUIRED_ATTRS (dr_rows only carries what the solvers
        # themselves need -- A/alpha_0/phi_j/Ri/cp/Dpw/Z), so it's read off
        # the real per-row Bearing objects instead, not off dr_rows itself.
        for j, (row_geom, row_bearing) in enumerate(zip(dr_rows, dr_row_bearings)):
            print(f"      row{j}: Dw={row_bearing.Dw:>6.2f} mm   cp={row_geom['cp']:>12.3f}")
        if not m_frac["ok"]:
            print(f"      WARNING: fraction-based outer solve did NOT converge "
                  f"(ok=False, residual={m_frac['residual']:.3e}, n_iter={m_frac['n_iter']}) "
                  f"-- its numbers below are NOT trustworthy, see caveat in this "
                  f"module's docstring / conversation notes about an Fa~=0 + "
                  f"heterogeneous-rows degeneracy in the fraction parametrization.")
        if not m_shared["ok"]:
            print(f"      WARNING: shared-displacement outer solve did NOT converge "
                  f"(ok=False, residual={m_shared['residual']:.3e}, n_iter={m_shared['n_iter']}) "
                  f"-- its numbers below are NOT trustworthy.")
        print(f"      {'':22s}  {'fraction-based':>18s}  {'shared-disp.':>18s}  {'diff':>12s}")
        print(f"      {'delta_r [mm]':22s}  {m_frac['delta_r']:>18.6f}  {m_shared['delta_r']:>18.6f}  "
              f"{m_shared['delta_r']-m_frac['delta_r']:>12.3e}")
        print(f"      {'delta_a [mm]':22s}  {m_frac['delta_a']:>18.6f}  {m_shared['delta_a']:>18.6f}  "
              f"{m_shared['delta_a']-m_frac['delta_a']:>12.3e}")

        for j in range(len(dr_rows)):
            qf, qs = m_frac["Q_max_rows"][j], m_shared["Q_max_rows"][j]
            rel = (qs - qf) / qf * 100.0 if qf else float("nan")
            print(f"      {'Q_max row'+str(j)+' [N]':22s}  {qf:>18.3f}  {qs:>18.3f}  "
                  f"{rel:>11.4f}%")
            fr_f, fr_s = m_frac["f_r"][j], m_shared["f_r"][j]
            fa_f, fa_s = m_frac["f_a"][j], m_shared["f_a"][j]
            print(f"      {'  f_r row'+str(j):22s}  {fr_f:>18.5f}  {fr_s:>18.5f}  "
                  f"{fr_s-fr_f:>12.3e}")
            print(f"      {'  f_a row'+str(j):22s}  {fa_f:>18.5f}  {fa_s:>18.5f}  "
                  f"{fa_s-fa_f:>12.3e}")

        print(f"      {'outer n_iter':22s}  {m_frac['n_iter']:>18}  {m_shared['n_iter']:>18}")
        print(f"      {'outer residual':22s}  {m_frac['residual']:>18.3e}  {m_shared['residual']:>18.3e}")
        print(f"      {'ok':22s}  {str(m_frac['ok']):>18}  {str(m_shared['ok']):>18}")


# ===========================================================================
# PLOTS -- read the SAME `rows` dict the printed table already built, no
# re-solving.
#   plot_polar_comparison() -- one figure per shaft, (slot x scenario) grid
#     of polar Q_j-vs-phi_global plots; double-row's locating subplot shows
#     both rows' traces overlaid.
#   plot_qmax_comparison() -- grouped bar chart, Q_max per scenario, one
#     subplot per shaft's locating bearing.
# ===========================================================================

def plot_polar_comparison(rows: dict[tuple[str, str], dict[str, dict]]) -> None:
    shaft_names = sorted({shaft_name for shaft_name, _ in rows})
    slots       = ["locating (a)", "floating (b)"]
    scenario_names = [s for s, _, _, _, _ in SCENARIOS]

    for shaft_name in shaft_names:
        fig, axes = plt.subplots(len(slots), len(scenario_names),
                                 figsize=(5 * len(scenario_names), 5 * len(slots)),
                                 subplot_kw={"projection": "polar"})
        fig.suptitle(f"Load distribution comparison -- {shaft_name}  (ISO/TS 16281)",
                    fontsize=11, fontweight="bold")

        for i, slot in enumerate(slots):
            by_scenario = rows.get((shaft_name, slot), {})
            for j, scenario_name in enumerate(scenario_names):
                ax = axes[i, j] if len(slots) > 1 else axes[j]
                r  = by_scenario.get(scenario_name)
                if r is None:
                    ax.set_visible(False)
                    continue

                colors = ["#2166ac", "#d62728"]   # row0, row1 (if 2 rows)
                for k, (phi, Q, row_label) in enumerate(r["dist_traces"]):
                    phi_c = np.append(phi, phi[0])
                    Q_c   = np.append(Q,   Q[0])
                    color = colors[k % len(colors)]
                    lbl   = row_label if row_label is not None else None
                    ax.plot(phi_c, Q_c, color=color, lw=1.4, marker="o", ms=3, label=lbl)
                    ax.fill(phi_c, Q_c, color=color, alpha=0.10)

                if any(lbl for _, _, lbl in r["dist_traces"]):
                    ax.legend(fontsize=6, frameon=False, loc="upper right")

                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
                ax.tick_params(labelsize=6)
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                short = scenario_name.split("(")[0].strip()
                ax.set_title(f"{short}\n{slot}\nQ_max={r['Q_max']:.0f} N  "
                            f"n_loaded={r['n_loaded']}/{r['Z']}",
                            fontsize=7)

        plt.tight_layout()
        plt.show()


def plot_qmax_comparison(rows: dict[tuple[str, str], dict[str, dict]]) -> None:
    shaft_names = sorted({shaft_name for shaft_name, slot in rows if slot == "locating (a)"})
    scenario_names = [s for s, _, _, _, _ in SCENARIOS]
    short_names = [s.split("(")[0].strip() for s in scenario_names]

    fig, axes = plt.subplots(1, len(shaft_names), figsize=(5 * len(shaft_names), 4.5))
    if len(shaft_names) == 1:
        axes = [axes]
    fig.suptitle("Q_max comparison -- locating bearing, by scenario", fontsize=11, fontweight="bold")

    for ax, shaft_name in zip(axes, shaft_names):
        by_scenario = rows.get((shaft_name, "locating (a)"), {})
        vals  = [by_scenario[s]["Q_max"] if s in by_scenario else 0.0 for s in scenario_names]
        bars  = ax.bar(short_names, vals, color=["#4c72b0", "#c44e52", "#55a868"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}",
                   ha="center", va="bottom", fontsize=8)
        ax.set_title(shaft_name, fontsize=9)
        ax.set_ylabel("Q_max [N]", fontsize=8)
        ax.tick_params(labelsize=7, axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2, lw=0.4, axis="y")

    plt.tight_layout()
    plt.show()


# ===========================================================================
# MAIN -- one comparison table per shaft, scenarios as columns
# ===========================================================================

W = 100

if __name__ == "__main__":
    library = SimpleFEMResultsLibrary()

    # rows[(shaft_name, bearing_slot)] = {scenario_name: row_dict}
    rows: dict[tuple[str, str], dict[str, dict]] = {}

    # solver_comparison[(scenario_name, shaft_name)] = {
    #     "fraction": BallBearingResult, "shared": BallBearingResult,
    #     "Fr": ..., "Fa": ..., "phi_Fr": ..., "dr_rows": [...],
    # } -- populated for every scenario's double-row-BALL locating bearing
    # (ball-only -- see "Known gaps" for why the roller side has no
    # analogous entry here). Both "fraction" and "shared" are independent,
    # directly-called solve_bearing() results (see the FIX in this module's
    # changelog for why neither reuses bundle.extra["multirow_result"]).
    solver_comparison: dict[tuple[str, str], dict] = {}

    # Two genuinely different solver instances -- fraction_solver is the
    # TRUE fraction-based class (ISO16281MultiRowBallSolver), no longer
    # registered as MULTIROW_SOLVER but still fully usable when called
    # directly, exactly like shared_solver already was.
    fraction_solver = ISO16281MultiRowBallSolver(tol=SOLVER_TOL)
    shared_solver    = ISO16281MultiRowBallSolverSharedDisplacement(tol=SOLVER_TOL)

    for scenario_name, locating_builder, floating_builder, floating_positions, locating_positions in SCENARIOS:
        print(f"\n=== scenario: {scenario_name} ===")

        shaft_systems = build_systems(B_STUDY, locating_builder, floating_builder,
                                      floating_positions, locating_positions)

        for name, shaft_sys in shaft_systems.items():
            bearings    = {b.label: b for b in shaft_sys.bearings}
            extra_nodes = gear_grade_nodes(shaft_sys)

            fem = SimpleFEMSolver()
            fem.solve(shaft_sys, extra_mandatory=extra_nodes)
            ShaftResultsReader(fem, shaft_sys).read(library)

            # Solver-facing bearings dict -- same as `bearings` except any
            # multi-row slot's label is swapped for a duck-typed multi-row
            # view, so RollingBearingSolver's own multi-row dispatch picks
            # it up automatically. multirow_by_label maps label -> (row
            # bearings, dr_rows, bearing_type) for whichever slot(s) this
            # scenario makes multi-row (0, 1, or -- new this turn -- 2).
            bearings_for_solver = dict(bearings)
            multirow_by_label: dict[str, tuple[list[Bearing], list[dict], BearingType]] = {}
            for slot_suffix, (row_bearings, dr_rows_i, btype) in MULTIROW_SLOTS_BY_SCENARIO.get(scenario_name, {}).items():
                lbl = next(l for l in bearings if l.endswith(slot_suffix))
                multirow_by_label[lbl] = (row_bearings, dr_rows_i, btype)
                arrangement = "locating" if slot_suffix == "a" else "floating"
                bearings_for_solver[lbl] = _make_dr_multirow_view(lbl, dr_rows_i, btype, arrangement)

            # catalog -- per-bearing kwargs for postprocess_and_record().
            # "capacity" is omitted for any multirow label: the duck-typed
            # multi-row view has no .family, so it's patched in manually
            # afterward using the real DR6204()/DR_NU204() Bearing instead.
            # "dynamic_equivalent_load" is omitted for the roller (see
            # "Known gaps" above).
            catalog: dict[str, dict] = {}
            for lbl, b_real in bearings.items():
                entry: dict = {}
                if lbl not in multirow_by_label:
                    entry["capacity"] = {"Cr": _capacity_Cr(b_real)}
                if b_real.bearing_type != BearingType.CYLINDRICAL_ROLLER:
                    entry["dynamic_equivalent_load"] = dict(
                        inner_rotating=INNER_ROTATING, outer_rotating=OUTER_ROTATING,
                    )
                catalog[lbl] = entry

            # ONE call per shaft: solves (internally, since load_distribution
            # is not pre-supplied) AND records capacity / dynamic_equivalent_
            # load / stiffness for every bearing on the shaft.
            results_lib = BearingResultsLibrary()
            solver      = RollingBearingSolver(tol=SOLVER_TOL)
            results_lib = solver.postprocess_and_record(
                shaft_sys, bearings_for_solver, library, catalog,
                results=results_lib,
            )

            for lbl in multirow_by_label:
                # Manual capacity patch, via the real DR6204()/DR_NU204()
                # Bearing (bearings[lbl], not the duck-typed solver view).
                results_lib.set_capacity(
                    lbl, _per_element_capacity(bearings[lbl]),
                )

            shaft_res     = library.get(name)
            node_by_label = {n.label: n for n in shaft_res.bearing_nodes}

            for lbl in bearings:
                node   = node_by_label[lbl]
                b_obj  = bearings[lbl]
                slot   = "locating (a)" if lbl.endswith("a") else "floating (b)"
                key    = (name, slot)
                bundle = results_lib.get(lbl)

                if lbl in multirow_by_label:
                    row_bearings_l, dr_rows_l, btype_l = multirow_by_label[lbl]
                    mr_result = bundle.extra["multirow_result"]

                    if btype_l == BearingType.DEEP_GROOVE_BALL:
                        rows.setdefault(key, {})[scenario_name] = _format_double_row_result(
                            node, bundle, mr_result, dr_rows_l, row_bearings_l,
                        )

                        # --- SOLVER COMPARISON: two INDEPENDENT direct calls,
                        # neither reusing bundle.extra["multirow_result"]
                        # (which is whatever RollingBearingSolver's dispatch
                        # actually used -- currently shared-displacement, see
                        # this module's changelog FIX entry for why using it
                        # as "fraction" would have compared that solver
                        # against itself). Ball-only -- see "Known gaps" for
                        # why there is no roller-side analogue of this block.
                        Fr_xz, Fr_xy, Fa, psi = _reconstruct_solver_inputs(node, bundle, mr_result)
                        mr_view = _make_dr_multirow_view(lbl, dr_rows_l, btype_l, "locating")
                        mr_result_fraction = fraction_solver.solve_bearing(
                            mr_view, Fr_xz, Fr_xy, Fa, psi, label=f"{lbl}-fraction",
                        )
                        mr_result_shared = shared_solver.solve_bearing(
                            mr_view, Fr_xz, Fr_xy, Fa, psi, label=f"{lbl}-shared",
                        )
                        solver_comparison[(scenario_name, name)] = dict(
                            fraction=mr_result_fraction, shared=mr_result_shared,
                            Fr=node.Fr, Fa=node.Fa,
                            phi_Fr=bundle.load_distribution[0].phi_Fr,
                            dr_rows=dr_rows_l, dr_row_bearings=row_bearings_l,
                        )
                    else:
                        rows.setdefault(key, {})[scenario_name] = _format_double_row_result_roller(
                            node, bundle, mr_result, dr_rows_l, row_bearings_l,
                        )
                    continue

                res      = bundle.load_distribution[0]
                dist     = _contact_distribution(b_obj, res)
                Q_arr    = dist[:, 1]
                n_loaded = int((Q_arr > 0).sum())
                Q_max    = float(Q_arr.max())
                stiff    = bundle.stiffness

                Q_ci, Q_ce = bundle.capacity
                Q_ci, Q_ce = [Q_ci], [Q_ce]

                if bundle.dynamic_equivalent_load is not None:
                    eq = bundle.dynamic_equivalent_load[0]
                    Q_ei, Q_ee = [eq.Q_ei], [eq.Q_ee]
                else:
                    Q_ei, Q_ee = None, None

                if b_obj.bearing_type == BearingType.CYLINDRICAL_ROLLER:
                    life = _bearing_life_roller(lbl, [b_obj], [res], Cr=_capacity_Cr(b_obj))
                else:
                    life = _bearing_life(lbl, Q_ci, Q_ei, Q_ce, Q_ee, Cr=_capacity_Cr(b_obj))

                rows.setdefault(key, {})[scenario_name] = dict(
                    kind=b_obj.family.name,
                    Fr=node.Fr, Fa=node.Fa,
                    Q_max=Q_max, n_loaded=n_loaded, Z=b_obj.Z,
                    Ka=stiff.Ka, ok=res.ok,
                    delta_r=res.delta_r, delta_a=res.delta_a, psi=res.psi,
                    dist_traces=[(dist[:, 0], dist[:, 1], None)],
                    mr_n_iter=None, mr_residual=None, mr_f_r=None, mr_f_a=None,
                    Q_ei=Q_ei, Q_ee=Q_ee,
                    Q_ci=Q_ci, Q_ce=Q_ce,
                    L10r_rows=(life["L10r_rows"] if life else None),
                    L10r=(life["L10r"] if life else None),
                    Pref_r=(life["Pref_r"] if life else None),
                )

    # comparison table -- grouped by SCENARIO (outer), shaft/slot (inner)
    print(f"\n{'='*W}")
    print("  BEARING COMBINATION COMPARISON")
    print(f"{'='*W}")

    shaft_slot_keys = sorted(rows.keys())

    for scenario_name, _, _, _, _ in SCENARIOS:
        print(f"\n{'='*W}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"{'='*W}")

        for shaft_name, slot in shaft_slot_keys:
            r = rows[(shaft_name, slot)].get(scenario_name)
            if r is None:
                continue
            if r["Ka"] is None:
                ka_txt = "n/a"
            elif r["Ka"] == float("inf"):
                ka_txt = "inf"
            else:
                ka_txt = f"{r['Ka']:.3e}"
            print(f"\n-- {shaft_name} / {slot} {'-'*max(W-14-len(shaft_name)-len(slot), 4)}")
            print(f"      kind={r['kind']:<28}  ok={r['ok']}")
            print(f"      Fr={r['Fr']:>9.1f} N   Fa={r['Fa']:>9.1f} N   "
                  f"Q_max={r['Q_max']:>9.1f} N   n_loaded={r['n_loaded']}/{r['Z']}")
            print(f"      delta_r={r['delta_r']:>10.5f} mm   delta_a={r['delta_a']:>10.5f} mm   "
                  f"psi={np.degrees(r['psi']):>9.5f} deg")
            print(f"      Ka={ka_txt}")

            if r["mr_n_iter"] is not None:
                f_r_txt = ", ".join(f"{v:.4f}" for v in r["mr_f_r"])
                # mr_f_a is always None for a roller multi-row row (radial
                # rollers carry no axial load, so there is nothing to
                # split -- RollerBearingResult.multirow() never sets it) --
                # guard instead of iterating over None.
                f_a_txt = (", ".join(f"{v:.4f}" for v in r["mr_f_a"])
                          if r["mr_f_a"] is not None
                          else "n/a (no axial load -- radial roller)")
                print(f"      [multirow outer solve]  ok={r['ok']}   "
                      f"n_iter={r['mr_n_iter']}   residual={r['mr_residual']:.3e}")
                print(f"          f_r (per row) = [{f_r_txt}]")
                print(f"          f_a (per row) = [{f_a_txt}]")

            qci_txt = ", ".join(f"{v:.1f}" for v in r["Q_ci"])
            qce_txt = ", ".join(f"{v:.1f}" for v in r["Q_ce"])
            print(f"      Q_ci (per row) = [{qci_txt}] N   "
                  f"Q_ce (per row) = [{qce_txt}] N")

            if r["Q_ei"] is not None:
                qei_txt = ", ".join(f"{v:.1f}" for v in r["Q_ei"])
                qee_txt = ", ".join(f"{v:.1f}" for v in r["Q_ee"])
                print(f"      Q_ei (per row) = [{qei_txt}] N   "
                      f"Q_ee (per row) = [{qee_txt}] N")

            if r["L10r"] is not None:
                l10r_rows_txt = ", ".join(f"{v:.2f}" for v in r["L10r_rows"])
                print(f"      L10r (per row) = [{l10r_rows_txt}] Mrev   "
                      f"L10r (bearing) = {r['L10r']:.2f} Mrev   "
                      f"Pref_r = {r['Pref_r']:.1f} N")
            else:
                print(f"      L10r/Pref_r: n/a (unexpected -- see Known gaps)")

    print_solver_comparison(solver_comparison)

    plot_polar_comparison(rows)
    plot_qmax_comparison(rows)

    print(f"\n{'='*W}")
    print("  done -- see module docstring before trusting these numbers.")
    print(f"{'='*W}\n")