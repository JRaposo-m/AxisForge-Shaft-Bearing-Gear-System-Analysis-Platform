"""
design_double_row_life_verification.py

Life-only script: DOUBLE-row locating (2x 6204-row, homogeneous) + DOUBLE-row
floating roller (2x NU204-row, homogeneous) -- the DOUBLE_HOMO_SCENARIO slice
of design_bearing_combination_comparison.py, stripped down to just L10r/Pref_r
plus an internal numeric verification of the row-combination law (Zaretsky
eq.49a: L^-e = sum(Li^-e)) against the actual solved per-row values.

Duplicated (not imported) from design_bearing_combination_comparison.py per
fixtures/README.md ("do not modify for a specific case -- duplicate and
rename") -- that script already covers this scenario inside a much bigger
comparison; this one exists so the life numbers + verification aren't buried
under distribution tables, solver comparison and plots.

Verification performed, per shaft, per bearing type:
  1. rows equal    -- per-row L10r match (rows are geometrically identical,
                       so load should split ~50/50 -> equal row life).
  2. combine check -- L10r_bearing (from combine_row_L10r()) reproduces
                       row_L10r * n_rows**(-1/e) computed directly here,
                       independently of that function.
No plots, no solver comparison, no baseline/heterogeneous scenarios --
see design_bearing_combination_comparison.py for those.

UPDATED, this turn -- nonzero axial load + more laminae
-----------------------------------------------------------------------
Every shaft now also carries an explicit AXIAL_LOAD_N via AxialLoad(),
so the ball locating bearing's Q_ce/Q_ee (axial-side eq.29 term) is driven
by a real external Fa instead of only the small internal delta_a that
misalignment (psi) alone produced before. AxialLoad's exact import path/
signature was NOT independently confirmed against this repo (mirrored on
RadialLoad's own (position, magnitude, label=...) shape, dropping
theta_deg since axial has one fixed direction) -- if this raises an
ImportError/TypeError, tell me the real signature and I'll fix it, same
as the RollingElementCapacity import earlier in this conversation.
The roller (NU204, radial cylindrical roller) still structurally carries
NO axial load regardless (Ka_regime stays "no_load") -- see roller_
bearing_postprocessing.py's bearing_stiffness() docstring -- so this only
changes the ball side's numbers, by design (locating/floating split).

CR_GEOM["n_s"] raised 40 -> 60 (more laminae along the roller length).
"""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.deep_groove_ball import DeepGrooveBallFamily
from axisforge.core.machine_elements.Bearings.families.roller_bearing.radial.subtypes.cylindrical_roller import CylindricalRollerFamily
from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.loads import RadialLoad, TorqueLoad, AxialLoad
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    ShaftResultsReader,
    SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_solver import (
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_solver import (
    REQUIRED_ATTRS as ROLLER_REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    BearingResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_postprocessing import (
    basic_reference_rating_life,
    DynamicEquivalentReferenceLoad,
    _E_BALL,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_postprocessing import (
    LaminaDynamicEquivalentLoad,
    basic_reference_rating_life as roller_basic_reference_rating_life,
    DynamicEquivalentReferenceLoad as RollerDynamicEquivalentReferenceLoad,
    _E_ROLLER,
)
from axisforge.core.machine_elements.Bearings.families.roller_bearing.radial.functions.capacity import (
    RollingElementCapacity,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.rolling_bearing_solver import (
    RollingBearingSolver,
)
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_grade import Grader


# ===========================================================================
# PARAMETERS -- same values as design_bearing_combination_comparison.py's
# DOUBLE_HOMO_SCENARIO (locating=ball, floating=roller, both homogeneous).
# ===========================================================================

B_STUDY  = 20.0
P_W      = 1000.0
RPM_IN   = 450.0
r_pulley = 0.050

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

SOLVER_TOL = 1.0e-6
GEAR_GRADE = "grade_3"

INNER_ROTATING = True
OUTER_ROTATING = False

# Nonzero external axial load, one per shaft -- comfortably above the
# near-zero range where the Fa~=0 outer-solve degeneracy (see design_
# bearing_combination_comparison.py's changelog) could bite; not derived
# from any gear-mesh thrust or a manufacturer Fa_min figure, purely
# illustrative. Only the ball locating bearing reacts it (see module
# docstring) -- adjust here if you want a different magnitude.
AXIAL_LOAD_N = 500.0

BB_GEOM = dict(Dw=7.94, Dpw=33.5, Z=8, s=0.010, E=206_000.0, nu=0.3)
CR_6204 = 12_700.0

CR_GEOM  = dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=60)
CR_NU204 = 28_500.0

DR_ROW_GEOM_HOMO = [
    dict(Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
         E=BB_GEOM["E"], nu=BB_GEOM["nu"], s=BB_GEOM["s"])
    for _ in range(2)
]
DR_CATALOG = dict(d=20.0, D=47.0, b=28.0, C=CR_6204, C0=6_550.0,
                  designation="DR6204-illustrative")

DR_ROLLER_ROW_GEOM_HOMO = [
    dict(Dwe=CR_GEOM["Dwe"], Lwe=CR_GEOM["Lwe"], Dpw=CR_GEOM["Dpw"],
         Z=CR_GEOM["Z"], s=CR_GEOM["s"], n_s=CR_GEOM["n_s"])
    for _ in range(2)
]
DR_ROLLER_CATALOG = dict(d=20.0, D=47.0, b=24.0, C=CR_NU204, C0=22_000.0,
                         designation="DR-NU204-illustrative")

# Recentered floating position per shaft -- see design_bearing_combination_
# comparison.py's DR_ROLLER_FLOATING_POSITIONS comment for the margin math.
DR_ROLLER_FLOATING_POSITIONS = (105.0, 132.5, 132.5)


# ===========================================================================
# BEARING BUILDERS
# ===========================================================================

def _row_attrs(b: Bearing) -> dict:
    return {a: getattr(b, a) for a in REQUIRED_ATTRS}


def _build_dr_rows(row_geoms: list[dict], label_prefix: str) -> tuple[list[Bearing], list[dict]]:
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


DR_ROW_BEARINGS_HOMO, DR_ROWS_HOMO = _build_dr_rows(DR_ROW_GEOM_HOMO, "dr-homo")


def _row_attrs_roller(b: Bearing) -> dict:
    return {a: getattr(b, a) for a in ROLLER_REQUIRED_ATTRS}


def _build_dr_rows_roller(row_geoms: list[dict], label_prefix: str) -> tuple[list[Bearing], list[dict]]:
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


DR_ROLLER_ROW_BEARINGS_HOMO, DR_ROLLER_ROWS_HOMO = _build_dr_rows_roller(
    DR_ROLLER_ROW_GEOM_HOMO, "dr-roller-homo")


def DR6204(position, label, row_geom: dict) -> Bearing:
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(arrangement="locating", label=label,
                               position=position, **DR_CATALOG),
        geometry=row_geom,
        analyses={"point_contact": True},
    )


def DR_NU204(position, label, row_geom: dict) -> Bearing:
    return Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(arrangement="floating", label=label,
                               position=position, **DR_ROLLER_CATALOG),
        geometry=row_geom,
        analyses={"line_contact": True},
    )


# ===========================================================================
# SHAFT SYSTEM -- identical gearbox topology to design_bearing_combination_
# comparison.py, hardcoded to the double-homo builders (one scenario only).
# ===========================================================================

def make_stepped_shaft(name, total_length, d_seat, d_body,
                       l_seat_a, l_seat_b, fillet_r, material_id="AISI_1045"):
    l_body = total_length - l_seat_a - l_seat_b
    sh = Shaft(label=name)
    sh.add_section(ShaftSection(length=l_seat_a, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatA"))
    sh.add_section(ShaftSection(
        length=l_body, diameter=d_body, material_id=material_id, label=f"{name}-body",
        shoulder_left=Shoulder(fillet_radius=fillet_r, diameter_large=d_body, diameter_small=d_seat),
        shoulder_right=Shoulder(fillet_radius=fillet_r, diameter_large=d_body, diameter_small=d_seat),
    ))
    sh.add_section(ShaftSection(length=l_seat_b, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatB"))
    return sh


def build_systems() -> dict[str, ShaftSystem]:
    """Double-homo scenario only -- locating=DR6204 (row0 geom), floating=DR_NU204 (row0 geom)."""
    locating_builder = partial(DR6204,   row_geom=DR_ROW_GEOM_HOMO[0])
    floating_builder = partial(DR_NU204, row_geom=DR_ROLLER_ROW_GEOM_HOMO[0])
    pos1, pos2, pos3 = DR_ROLLER_FLOATING_POSITIONS

    gear_kw = dict(**GEAR_KW_BASE, b=B_STUDY)

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

    sys1.add_bearing(locating_builder(20.0,  "brg1a")); sys1.add_bearing(floating_builder(pos1, "brg1b"))
    sys2.add_bearing(locating_builder(25.0,  "brg2a")); sys2.add_bearing(floating_builder(pos2, "brg2b"))
    sys3.add_bearing(locating_builder(25.0,  "brg3a")); sys3.add_bearing(floating_builder(pos3, "brg3b"))

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
                                    label="double-row-homo-life-verification")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    T3     = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    # Axial load per shaft -- see AXIAL_LOAD_N's own comment. Positioned at
    # each shaft's own locating-bearing seat (20.0/25.0/25.0) -- illustrative
    # only, not modeling a real gear-thrust origin.
    sys1.add_load(AxialLoad(20.0, AXIAL_LOAD_N, label="axial-shaft1"))
    sys2.add_load(AxialLoad(25.0, AXIAL_LOAD_N, label="axial-shaft2"))
    sys3.add_load(AxialLoad(25.0, AXIAL_LOAD_N, label="axial-shaft3"))

    return {sh.name: sh for sh in (sys1, sys2, sys3)}


def gear_grade_nodes(shaft_sys: ShaftSystem, grade: str = GEAR_GRADE) -> list[float]:
    base_nodes = Mesh1D(shaft_sys).x_nodes
    extra: list[float] = []
    for ge in shaft_sys.gears:
        lo, hi = shaft_sys.gear_extent(ge)
        extra += Grader(lo, hi, base_nodes).get_grade(grade)
    return extra


def _make_dr_multirow_view(label, dr_rows, bearing_type, arrangement):
    return SimpleNamespace(rows=dr_rows, i=len(dr_rows), label=label,
                           bearing_type=bearing_type, arrangement=arrangement)


# ===========================================================================
# LIFE -- eq.(29)-(31) ball, eq.(65)-(67) roller. Copied unchanged from
# design_bearing_combination_comparison.py's _bearing_life()/_bearing_life_
# roller()/_per_element_capacity()/_capacity_Cr().
# ===========================================================================

def _capacity_Cr(b_obj: Bearing) -> float:
    return CR_NU204 if b_obj.bearing_type == BearingType.CYLINDRICAL_ROLLER else CR_6204


def _per_element_capacity(b_obj: Bearing) -> tuple[float, float]:
    return b_obj.family.per_element_dynamic_capacity(b_obj, Cr=_capacity_Cr(b_obj))


def _bearing_life(label, Q_ci_rows, Q_ei_rows, Q_ce_rows, Q_ee_rows, Cr):
    per_row, L10r_bearing = basic_reference_rating_life(label, Q_ci_rows, Q_ei_rows, Q_ce_rows, Q_ee_rows)
    pref = DynamicEquivalentReferenceLoad.from_L10r(label, L10r_bearing, Cr=Cr)
    return dict(L10r_rows=[p.L10r for p in per_row], L10r=L10r_bearing, Pref_r=pref.Pref_r)


def _bearing_life_roller(label, row_bearings, row_results, Cr,
                         inner_rotating=INNER_ROTATING, outer_rotating=OUTER_ROTATING):
    q_kci_rows, q_kei_rows, q_kce_rows, q_kee_rows = [], [], [], []
    for row_bearing, row_res in zip(row_bearings, row_results):
        n_s = row_bearing.n_s
        Q_ci, Q_ce = _per_element_capacity(row_bearing)
        q_ci, q_ce = RollingElementCapacity.per_lamina(Q_ci, Q_ce, n_s)
        q_kci_rows.append(np.full(n_s, q_ci))
        q_kce_rows.append(np.full(n_s, q_ce))
        eq = LaminaDynamicEquivalentLoad.from_distribution(
            row_bearing, row_res, inner_rotating=inner_rotating, outer_rotating=outer_rotating,
        )
        q_kei_rows.append(eq.q_kei)
        q_kee_rows.append(eq.q_kee)
    per_row, L10r_bearing = roller_basic_reference_rating_life(label, q_kci_rows, q_kei_rows, q_kce_rows, q_kee_rows)
    pref = RollerDynamicEquivalentReferenceLoad.from_L10r(label, L10r_bearing, Cr=Cr)
    return dict(L10r_rows=[p.L10r for p in per_row], L10r=L10r_bearing, Pref_r=pref.Pref_r,
               q_kci_rows=q_kci_rows, q_kei_rows=q_kei_rows,
               q_kce_rows=q_kce_rows, q_kee_rows=q_kee_rows)


def report_lamina_stats(kind: str, q_kci_rows, q_kei_rows, q_kce_rows, q_kee_rows) -> None:
    """
    Per-row lamina diagnostics -- separates the two competing effects
    discussed in conversation: f_k edge amplification (eq.60, pushes q_kei/
    q_kee UP at the two edge laminae, k=1 and k=n_s) vs. the Z-normalization
    in eq.(61)-(64) (divides by the FULL roller count Z, not just the
    loaded ones -- pushes q_kei/q_kee DOWN when few rollers are loaded).
    Prints min/max/mean of q_kei/q_kee per row (and which lamina carries the
    max -- should be an edge index if f_k dominates there) plus how much of
    eq.(65)'s summed terms comes from just the 2 edge laminae vs the rest.
    """
    for j, (q_kci, q_kei, q_kce, q_kee) in enumerate(zip(q_kci_rows, q_kei_rows, q_kce_rows, q_kee_rows)):
        n_s = len(q_kei)
        term_i = (q_kei / q_kci) ** 4.5
        term_e = (q_kee / q_kce) ** 4.5
        total  = term_i.sum() + term_e.sum()
        edge   = [0, n_s - 1]
        edge_frac = (term_i[edge].sum() + term_e[edge].sum()) / total if total > 0 else float("nan")

        print(f"      [{kind} row{j}] q_kei: min={q_kei.min():.3f}  max={q_kei.max():.3f} "
              f"(k={int(np.argmax(q_kei)) + 1}/{n_s})  mean={q_kei.mean():.3f} N")
        print(f"      [{kind} row{j}] q_kee: min={q_kee.min():.3f}  max={q_kee.max():.3f} "
              f"(k={int(np.argmax(q_kee)) + 1}/{n_s})  mean={q_kee.mean():.3f} N")
        print(f"      [{kind} row{j}] edge laminae (k=1,{n_s}) share of eq.(65) sum: {edge_frac:.1%}")


# ===========================================================================
# VERIFICATION -- row-combination law (Zaretsky eq.49a) checked against the
# ACTUAL solved per-row L10r from this run, independently of combine_row_
# L10r() itself: L10r_bearing must equal row_L10r * n_rows**(-1/e) when rows
# are (numerically) identical.
# ===========================================================================

_TOL_REL = 1.0e-9


def verify_homogeneous_life(kind: str, L10r_rows: list[float], L10r_bearing: float, e: float) -> bool:
    n = len(L10r_rows)
    row0 = L10r_rows[0]
    rows_equal = all(abs(L - row0) <= _TOL_REL * row0 for L in L10r_rows)

    expected = row0 * n ** (-1.0 / e)
    diff_rel = abs(L10r_bearing - expected) / expected
    combine_ok = diff_rel <= _TOL_REL

    ok = rows_equal and combine_ok
    print(f"      [{kind}] rows equal (per-row L10r all match): {rows_equal}")
    print(f"      [{kind}] combine check: row*{n}^(-1/e) = {expected:.6f}   "
          f"actual = {L10r_bearing:.6f}   rel.diff = {diff_rel:.3e}   "
          f"{'PASS' if combine_ok else 'FAIL'}")
    return ok


# Life exponent p, used ONLY in this rough sanity formula below -- NOT the
# same p as the actual Pref_r computation in roller_bearing_postprocessing.py
# (which hardcodes ISO/TS 16281 eq.(66)-(67)'s own ^(3/10), i.e. p=10/3,
# unconditionally, unaffected by this constant).
#
# _P_BALL=3 matches BOTH ISO/TS 16281 (ball's ^(1/3)) AND Zaretsky's own p=3
# for ball -- the two sources agree, so this sanity formula is internally
# consistent for ball (and it shows: predicted matches actual closely).
#
# _P_ROLLER: Zaretsky's NASA/TP-2013-215305/REV1 (the SAME report e=9/8
# comes from, eq.49a/49b -- see its eq.(52a)/(52b) discussion) states the
# roller load-life exponent is p=4 in HIS OWN framework, NOT p=10/3 -- that
# 10/3 is ISO/TS 16281's own (more modern) refinement, used in the real
# Pref_r formula but never stated as belonging to Zaretsky's e=9/8. Mixing
# Zaretsky's e with ISO's p (as this script did before) combines two
# non-matching conventions -- using Zaretsky's own p=4 here instead is the
# self-consistent choice for e=9/8. It still won't match perfectly (this
# formula ignores lamina/stress-riser effects regardless), but it removes
# that specific inconsistency.
_P_BALL   = 3.0
_P_ROLLER = 4.0


def report_pref_fr_ratio(kind: str, Fr: float, Pref_r: float, n_rows: int, p: float, e: float) -> None:
    """
    Pref_r/Fr -- INFORMATIONAL only, not gated into ALL CHECKS. For a
    single-row bearing with Fa=0, Pref_r ~= Fr (P=X*Fr, X=1) -- verified
    separately on the baseline scenario. For a double-row bearing that
    identity does not hold: two rows share Fr and their combined life
    (better than one row alone) needs a LOWER equivalent load against the
    bearing's own Cr to reproduce it. If load splits evenly, first-order:

        Pref_r ~= (Fr/n_rows) * n_rows**(1/(p*e))

    This ignores axial/misalignment/lamina effects, so treat it as a
    plausibility check, not an exact identity. p/e must come from the SAME
    source for this to be self-consistent -- see _P_ROLLER's own comment
    for why roller uses Zaretsky's p=4 here, not ISO's p=10/3.
    """
    actual    = Pref_r / Fr
    predicted = (1.0 / n_rows) * n_rows ** (1.0 / (p * e))
    print(f"      [{kind}] Pref_r/Fr = {actual:.4f}   "
          f"predicted ~= (Fr/n)*n^(1/(p*e)) = {predicted:.4f}   "
          f"diff = {actual - predicted:+.4f}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    library = SimpleFEMResultsLibrary()
    shaft_systems = build_systems()

    all_ok = True

    for name, shaft_sys in shaft_systems.items():
        print(f"\n=== {name} ===")

        bearings    = {b.label: b for b in shaft_sys.bearings}
        loc_label   = next(l for l in bearings if l.endswith("a"))
        flo_label   = next(l for l in bearings if l.endswith("b"))
        extra_nodes = gear_grade_nodes(shaft_sys)

        fem = SimpleFEMSolver()
        fem.solve(shaft_sys, extra_mandatory=extra_nodes)
        ShaftResultsReader(fem, shaft_sys).read(library)

        bearings_for_solver = dict(bearings)
        bearings_for_solver[loc_label] = _make_dr_multirow_view(
            loc_label, DR_ROWS_HOMO, BearingType.DEEP_GROOVE_BALL, "locating")
        bearings_for_solver[flo_label] = _make_dr_multirow_view(
            flo_label, DR_ROLLER_ROWS_HOMO, BearingType.CYLINDRICAL_ROLLER, "floating")

        catalog = {
            loc_label: dict(dynamic_equivalent_load=dict(
                inner_rotating=INNER_ROTATING, outer_rotating=OUTER_ROTATING)),
            flo_label: {},
        }

        results_lib = BearingResultsLibrary()
        solver = RollingBearingSolver(tol=SOLVER_TOL)
        results_lib = solver.postprocess_and_record(
            shaft_sys, bearings_for_solver, library, catalog, results=results_lib,
        )

        shaft_res     = library.get(name)
        node_by_label = {n.label: n for n in shaft_res.bearing_nodes}
        node_loc, node_flo = node_by_label[loc_label], node_by_label[flo_label]
        bundle_loc, bundle_flo = results_lib.get(loc_label), results_lib.get(flo_label)

        # --- ball locating ---
        Q_ci, Q_ce = [], []
        for rb in DR_ROW_BEARINGS_HOMO:
            qci, qce = _per_element_capacity(rb)
            Q_ci.append(qci); Q_ce.append(qce)
        Q_ei_rows = [e.Q_ei for e in bundle_loc.dynamic_equivalent_load]
        Q_ee_rows = [e.Q_ee for e in bundle_loc.dynamic_equivalent_load]
        life_ball = _bearing_life(loc_label, Q_ci, Q_ei_rows, Q_ce, Q_ee_rows,
                                  Cr=_capacity_Cr(DR_ROW_BEARINGS_HOMO[0]))

        print(f"  ball locating ({loc_label})  Fr={node_loc.Fr:.1f} N  Fa={node_loc.Fa:.1f} N")
        print(f"      L10r (per row) = {['%.2f' % v for v in life_ball['L10r_rows']]} Mrev   "
              f"L10r (bearing) = {life_ball['L10r']:.2f} Mrev   Pref_r = {life_ball['Pref_r']:.1f} N")
        ok_ball = verify_homogeneous_life("ball", life_ball["L10r_rows"], life_ball["L10r"], _E_BALL)
        report_pref_fr_ratio("ball", node_loc.Fr, life_ball["Pref_r"],
                             len(DR_ROW_BEARINGS_HOMO), _P_BALL, _E_BALL)

        # --- roller floating ---
        row_results_roller = bundle_flo.load_distribution
        life_roller = _bearing_life_roller(flo_label, DR_ROLLER_ROW_BEARINGS_HOMO, row_results_roller,
                                           Cr=_capacity_Cr(DR_ROLLER_ROW_BEARINGS_HOMO[0]))

        print(f"  roller floating ({flo_label})  Fr={node_flo.Fr:.1f} N  Fa={node_flo.Fa:.1f} N")
        print(f"      L10r (per row) = {['%.2f' % v for v in life_roller['L10r_rows']]} Mrev   "
              f"L10r (bearing) = {life_roller['L10r']:.2f} Mrev   Pref_r = {life_roller['Pref_r']:.1f} N")
        ok_roller = verify_homogeneous_life("roller", life_roller["L10r_rows"], life_roller["L10r"], _E_ROLLER)
        report_pref_fr_ratio("roller", node_flo.Fr, life_roller["Pref_r"],
                             len(DR_ROLLER_ROW_BEARINGS_HOMO), _P_ROLLER, _E_ROLLER)
        report_lamina_stats("roller", life_roller["q_kci_rows"], life_roller["q_kei_rows"],
                            life_roller["q_kce_rows"], life_roller["q_kee_rows"])

        all_ok = all_ok and ok_ball and ok_roller

    print(f"\n{'='*60}")
    print(f"  ALL CHECKS {'PASSED' if all_ok else 'FAILED'}")
    print(f"{'='*60}")