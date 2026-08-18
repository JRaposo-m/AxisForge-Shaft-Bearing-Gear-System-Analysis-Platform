"""
design_load_distribution_mixed_bearings.py

Coupled shaft-bearing analysis for the load-lifting gearbox at fixed b,
now with TWO bearing contact types per shaft: a locating deep-groove ball
bearing (6204) plus a floating cylindrical roller bearing (NU204) — the
"locating ball + floating roller" arrangement rolling_bearing_solver.py's
module docstring calls out as a common real gearbox pairing.

Duplicated from design_load_distribution_convergence.py rather than
edited in place, per the fixtures contract (fixtures/README.md: "do not
modify for a specific case — duplicate and rename"). Only the bearing
construction (brgXb switches from BB6204 to CRNU204) and the per-type
post-processing dispatch (BEARING_KITS below) are new; the FEM/gear
pipeline itself (steps 1, 3, and the plotting) is unchanged.

Pipeline per shaft:
  1. build ShaftSystem — locating 6204 + floating NU204 per shaft
  2. setup_internal_geometry (+ compute_hertz_point_contact /
     compute_line_contact_spring_constant, per bearing type)
  3. grade_3 nodes injected at gear intervals (mesh fixed, no convergence study)
  4. SimpleFEMSolver.solve -> ShaftResultsReader.read(library) -> SimpleFEMResultsLibrary
  5. RollingBearingSolver.solve(library) -> load distribution, dispatched by
     BearingType to ISO16281BallSolver (DEEP_GROOVE_BALL) or
     ISO16281RollerSolver (CYLINDRICAL_ROLLER) — both registered now
  6. per-type capacity + dynamic equivalent load, selected via BEARING_KITS
     (RollingElementCapacity/DynamicEquivalentRollingElementLoad for ball —
     both per-bearing scalars; RollerElementCapacity/LaminaDynamicEquivalentLoad
     for roller — capacity is whole-roller (Q_ci/Q_ce) plus per-lamina
     (q_ci/q_ce, eq.56-57), and the dynamic equivalent load is now genuinely
     per-lamina (q_kei/q_kee, arrays of length n_s, eq.61-64) rather than a
     single per-bearing figure, per ISO/TS 16281 S5.3.4 — see BEARING_KITS'
     derel_summary for how this script reduces that array to a single
     worst-lamina number for the console/plot output. See BEARING_KITS below
     for why this dispatch lives here and not in RollingBearingSolver: the
     orchestrator only unifies solve())
  7. plots:
       - global deflection diagrams  (v_xz, v_xy, v_res)
       - polar load distribution per bearing (Q_j vs phi_j_global)

NOT executed end-to-end in the session that wrote this file — the FEM/gear
machinery (SpurHelicalGear, ShaftSystem, SimpleFEMSolver, Mesh1D, ...) is
this repo's own code, not something that session could responsibly stub.
What WAS verified there, against real ISO16281BallSolver + ISO16281RollerSolver
instances: CylindricalRollerBearing's construction/geometry pipeline,
RollingBearingSolver's mixed-type dispatch (grouping, per-type solve,
order-preserving merge), and both capacity classes evaluating without
error. Run this once and sanity-check the printed output (Q_max fractions,
n_loaded/Z, delta_r/delta_a signs) before trusting it for a real design.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.types.ball_bearing.subtypes.deep_groove_ball import DeepGrooveBallBearing
from axisforge.core.machine_elements.Bearings.types.roller_bearing.subtypes.cylindrical_roller import CylindricalRollerBearing
from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.loads import RadialLoad, TorqueLoad
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
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing import (
    ISO16281BallSolver,
    RollingElementCapacity,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_postprocessing import (
    contact_distribution as ball_contact_distribution,
    bearing_stiffness as ball_bearing_stiffness,
    DynamicEquivalentRollingElementLoad as BallDynamicEquivalentRollingElementLoad,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing import (
    RollerElementCapacity,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_postprocessing import (
    contact_distribution as roller_contact_distribution,
    bearing_stiffness as roller_bearing_stiffness,
    LaminaDynamicEquivalentLoad,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.rolling_bearing_solver import (
    RollingBearingSolver,
)
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_grade import Grader

from dataclasses import dataclass
from typing import Callable


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

# --- locating bearing: deep-groove ball, 6204 (d=20, D=47, B=14) ---
BB_GEOM = dict(
    Dw  = 7.94,
    Dpw = 33.5,
    Z   = 8,
    ri  = 0.52 * 7.94,
    re  = 0.53 * 7.94,
    s   = 0.010,
    E   = 206_000.0,
    nu  = 0.3,
)
CR_6204 = 12_700.0   # dynamic radial load rating [N]  (ISO 281, 6204)

# --- floating bearing: cylindrical roller, NU204 (same d/D/B envelope as 6204) ---
# d=20, D=47, B=14 and C=28.5 kN, C0=22 kN are real SKF NU 204 ECP catalogue
# values (Explorer class, C-type cage). Internal geometry (Dwe, Lwe, Z, s)
# is NOT published by the manufacturer — these are engineering estimates
# for illustration, the same kind of approximation BB_GEOM's groove radii
# (ri=0.52*Dw, re=0.53*Dw) already make above. Replace with manufacturer
# internal-geometry data before using this for a real design.
CR_GEOM = dict(
    Dwe = 6.5,     # roller diameter [mm] — estimate
    Lwe = 6.0,     # effective roller length [mm] — estimate
    Dpw = 33.5,    # pitch circle diameter [mm] — same envelope as 6204
    Z   = 12,      # roller count — estimate, typical for this envelope
    s   = 0.015,   # diametral clearance [mm] — estimate, CN/normal-class order
    n_s = 40,      # laminae per roller (>= 30 per ISO/TS 16281 S5.2.2)
)
CR_NU204 = 28_500.0   # dynamic radial load rating [N] (SKF NU 204 ECP, Explorer)


# ===========================================================================
# PER-BEARING-TYPE POST-PROCESSING DISPATCH
#
# RollingBearingSolver only unifies solve() (see its module docstring) —
# everything downstream of a converged LoadDistributionResult (capacity,
# dynamic equivalent load, secant stiffness, contact distribution) is
# per-type, because the ball and roller postprocessing modules deliberately
# don't import each other (see their own module docstrings). This script
# is a caller that already knows each bearing's concrete BearingType, so it
# owns the dispatch itself — a small registry keyed by BearingType, mirroring
# the _SOLVER_MAP pattern inside rolling_bearing_solver.py, kept local to
# this script rather than inside the library (a caller-specific convenience,
# not a shared contract).
# ===========================================================================

def _ball_stiffness(b_obj, res, node):
    return ball_bearing_stiffness(b_obj, res, Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=node.Fa)


def _roller_stiffness(b_obj, res, node):
    # Roller bearings (NU/N-type) carry no axial load — bearing_stiffness()
    # takes no Fa argument here; Ka comes back inf/"no_load" unconditionally.
    return roller_bearing_stiffness(b_obj, res, Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy)


def _ball_derel_summary(b_obj, derel):
    """
    Ball's DynamicEquivalentRollingElementLoad.Q_ei/Q_ee are already single
    per-bearing scalars — nothing to reduce.
    """
    return ("Q_ei", float(derel.Q_ei), "Q_ee", float(derel.Q_ee))


def _roller_derel_summary(b_obj, derel):
    """
    Roller's LaminaDynamicEquivalentLoad.q_kei/q_kee are (n_s,) arrays — one
    value PER LAMINA (ISO/TS 16281 eq.61-64), not one per bearing (see the
    module docstring in roller_bearing_postprocessing.py for why this
    replaced the old per-roller DynamicEquivalentRollingElementLoad). For a
    single-line console/plot summary this script reports the WORST lamina
    (max q_kei / max q_kee) — the one that would govern a life check against
    RollerElementCapacity's q_ci/q_ce (also per lamina, eq.56-57) — while the
    full (n_s,) arrays remain available on `derel` itself for anyone who
    wants the per-lamina detail (e.g. via lamina_distribution()-style plots).
    """
    n_s = b_obj.n_s
    return (f"max q_kei (n_s={n_s})", float(derel.q_kei.max()),
            f"max q_kee (n_s={n_s})", float(derel.q_kee.max()))


def _roller_capacity_extra(cap) -> str | None:
    """Per-lamina capacity (eq.56-57) — has no ball-bearing analogue."""
    return f"q_ci (lamina) = {cap.q_ci:>9.2f} N    q_ce (lamina) = {cap.q_ce:>9.2f} N"


def _ball_capacity_extra(cap) -> str | None:
    return None


@dataclass(frozen=True)
class BearingKit:
    """Per-BearingType post-processing bundle — see module note above."""
    contact_distribution : Callable
    stiffness             : Callable   # (b_obj, res, node) -> BearingStiffness
    capacity_cls          : type       # .radial(bearing, Cr=...) -> capacity dataclass
    derel_cls              : type       # .from_distribution(bearing, res, ...) -> derel dataclass
    derel_summary          : Callable   # (b_obj, derel) -> (lbl_i, val_i, lbl_e, val_e), single-number view
    capacity_extra          : Callable   # (cap) -> str | None, extra capacity line (roller-only: lamina q_ci/q_ce)
    Cr                     : float
    supports_axial         : bool       # False => skip minimum_axial_load entirely


BEARING_KITS: dict[BearingType, BearingKit] = {
    BearingType.DEEP_GROOVE_BALL: BearingKit(
        contact_distribution = ball_contact_distribution,
        stiffness             = _ball_stiffness,
        capacity_cls          = RollingElementCapacity,
        derel_cls              = BallDynamicEquivalentRollingElementLoad,
        derel_summary          = _ball_derel_summary,
        capacity_extra          = _ball_capacity_extra,
        Cr                     = CR_6204,
        supports_axial         = True,
    ),
    BearingType.CYLINDRICAL_ROLLER: BearingKit(
        contact_distribution = roller_contact_distribution,
        stiffness             = _roller_stiffness,
        capacity_cls          = RollerElementCapacity,
        derel_cls              = LaminaDynamicEquivalentLoad,
        derel_summary          = _roller_derel_summary,
        capacity_extra          = _roller_capacity_extra,
        Cr                     = CR_NU204,
        supports_axial         = False,
    ),
}


# ===========================================================================
# PRINT HELPERS
# ===========================================================================

W = 72

def _hdr(text: str) -> None:
    print(f"\n  +{'─'*(W-4)}+")
    print(f"  |  {text:<{W-6}}|")
    print(f"  +{'─'*(W-4)}+")

def _ok(flag: bool) -> str:
    return "OK  converged" if flag else "!!  NOT converged"

def _fmt_k(k: float) -> str:
    return f"{k:.4e}" if k != float("inf") else "    inf (rigid)"


def print_bearing_result(lbl, b_obj, node, res, kit: BearingKit,
                         cap, derel, stiff, Fa_min, res_min):

    dist     = kit.contact_distribution(b_obj, res)
    phi_arr  = dist[:, 0]
    Q_arr    = dist[:, 1]
    n_loaded = int((Q_arr > 0).sum())
    Q_max    = float(Q_arr.max())

    psi_xz   = node.psi_xz
    psi_xy   = node.psi_xy
    psi_proj = psi_xz * np.cos(res.phi_Fr) + psi_xy * np.sin(res.phi_Fr)

    kind = "point contact (ball)" if kit.supports_axial else "line contact (roller)"

    # ── header ────────────────────────────────────────────────────────
    print(f"\n  +── Bearing  {lbl}  [{kind}]  {'─'*max(W-30-len(lbl)-len(kind), 4)}+")

    # ── reactions ─────────────────────────────────────────────────────
    print(f"  |")
    print(f"  |  REACTIONS")
    print(f"  |    Fr   = {node.Fr:>10.2f} N    "
          f"Fr_xz = {node.Fr_xz:>10.2f} N    Fr_xy = {node.Fr_xy:>10.2f} N")
    print(f"  |    Fa   = {node.Fa:>10.2f} N")

    # ── solver status ─────────────────────────────────────────────────
    print(f"  |")
    print(f"  |  SOLVER  [{_ok(res.ok)}]")
    print(f"  |    residual = {res.residual:.2e} N    nfev = {res.n_iter}")
    print(f"  |    dr = {res.delta_r:.4e} mm    "
          f"da = {res.delta_a:.4e} mm    "
          f"phi(Fr) = {np.degrees(res.phi_Fr):>7.2f} deg")

    # ── misalignment ──────────────────────────────────────────────────
    print(f"  |")
    print(f"  |  MISALIGNMENT  (shaft slope across seat)")
    print(f"  |    psi_xz = {np.degrees(psi_xz):>9.5f} deg    "
          f"psi_xy = {np.degrees(psi_xy):>9.5f} deg    "
          f"psi_proj(Fr) = {np.degrees(psi_proj):>9.5f} deg")

    # ── stiffness ─────────────────────────────────────────────────────
    print(f"  |")
    print(f"  |  SECANT STIFFNESS")
    print(f"  |    Kr_xz = {_fmt_k(stiff.Kr_xz):>16} N/mm    "
          f"Kr_xy = {_fmt_k(stiff.Kr_xy):>16} N/mm    "
          f"Ka = {_fmt_k(stiff.Ka):>16} N/mm    "
          f"[{stiff.Ka_regime}]")

    # ── axial preload ─────────────────────────────────────────────────
    print(f"  |")
    print(f"  |  MINIMUM AXIAL PRELOAD")
    if not kit.supports_axial:
        print(f"  |    N/A -- line-contact roller bearing (NU/N-type): no flange, "
              f"no axial capacity")
    elif Fa_min == 0.0:
        print(f"  |    Not required  (da = {res.delta_a:.4e} mm >= 0)")
    else:
        print(f"  |    Fa_min = {Fa_min:.2f} N  ->  da = {res_min.delta_a:.4e} mm at Fa_min")

    # ── ISO/TS 16281 capacity + dynamic equivalent load ─────────────────
    lbl_i, val_i, lbl_e, val_e = kit.derel_summary(b_obj, derel)
    extra_cap_line = kit.capacity_extra(cap)

    print(f"  |")
    print(f"  |  ISO/TS 16281  CAPACITY / EQUIVALENT LOAD")
    print(f"  |    Q_ci (inner) = {cap.Q_ci:>9.2f} N    "
          f"Q_ce (outer) = {cap.Q_ce:>9.2f} N")
    if extra_cap_line is not None:
        print(f"  |    {extra_cap_line}")
    print(f"  |    {lbl_i} (inner) = {val_i:>9.2f} N    "
          f"{lbl_e} (outer) = {val_e:>9.2f} N    "
          f"(inner_rotating = {derel.inner_rotating})")

    # ── load distribution ─────────────────────────────────────────────
    print(f"  |")
    print(f"  |  LOAD DISTRIBUTION  "
          f"({n_loaded}/{b_obj.Z} elements loaded    Q_max = {Q_max:.1f} N)")
    print(f"  |    {'j':>3}   {'phi_global':>10}   {'Q_j':>10}")
    print(f"  |    {'─'*3}   {'─'*10}   {'─'*10}")
    for j, (phi_i, Q_i) in enumerate(zip(phi_arr, Q_arr)):
        bar  = "#" * int(Q_i / max(Q_max, 1) * 20) if Q_i > 0 else "."
        flag = "  <- max" if Q_i == Q_max and Q_max > 0 else ""
        print(f"  |    {j:>3}   {np.degrees(phi_i):>8.2f} deg   {Q_i:>10.2f} N   {bar}{flag}")

    print(f"  +{'─'*(W-4)}+")


# ===========================================================================
# BUILD HELPERS
# ===========================================================================

def make_stepped_shaft(name, total_length, d_seat, d_body,
                       l_seat_a, l_seat_b, fillet_r,
                       material_id="AISI_1045"):
    l_body = total_length - l_seat_a - l_seat_b
    sh = Shaft(label=name)
    sh.add_section(ShaftSection(length=l_seat_a, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatA"))
    sh.add_section(ShaftSection(
        length=l_body, diameter=d_body, material_id=material_id,
        label=f"{name}-body",
        shoulder_left=Shoulder(fillet_radius=fillet_r,
                               diameter_large=d_body, diameter_small=d_seat),
        shoulder_right=Shoulder(fillet_radius=fillet_r,
                                diameter_large=d_body, diameter_small=d_seat),
    ))
    sh.add_section(ShaftSection(length=l_seat_b, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatB"))
    return sh


def BB6204(position, label):
    """Locating deep-groove ball bearing — always arrangement='locating'."""
    b = DeepGrooveBallBearing(
        d=20.0, D=47.0,
        designation="6204",
        b=14.0, C=CR_6204, C0=6_550.0,
        arrangement="locating",
        contact_angle_deg=0.0,
        label=label,
        position=position,
    )
    b.setup_internal_geometry(
        ri=BB_GEOM["ri"], re=BB_GEOM["re"],
        Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"],
        Z=BB_GEOM["Z"],
        E=BB_GEOM["E"], nu=BB_GEOM["nu"],
        s=BB_GEOM["s"],
    )
    b.compute_hertz_point_contact()
    return b


def CRNU204(position, label):
    """
    Floating cylindrical roller bearing — always arrangement='floating'
    (CylindricalRollerBearing raises if given anything else: an NU/N-type
    bearing has no flange, so it can never be the locating bearing).
    """
    b = CylindricalRollerBearing(
        d=20.0, D=47.0,
        designation="NU204",
        b=14.0, C=CR_NU204, C0=22_000.0,
        label=label,
        position=position,
    )
    b.setup_internal_geometry(
        Dwe=CR_GEOM["Dwe"], Lwe=CR_GEOM["Lwe"], Dpw=CR_GEOM["Dpw"],
        Z=CR_GEOM["Z"], s=CR_GEOM["s"], n_s=CR_GEOM["n_s"],
    )
    b.compute_line_contact_spring_constant()
    return b


# ===========================================================================
# ASSEMBLE GEAR SYSTEM
# ===========================================================================

def build_systems(b: float) -> dict[str, ShaftSystem]:
    gear_kw = dict(**GEAR_KW_BASE, b=b)

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

    # brgXa: locating ball (6204).  brgXb: floating roller (NU204) — was
    # BB6204(..., locating=False) in the single-type version of this script.
    sys1.add_bearing(BB6204(20.0,  "brg1a"))
    sys1.add_bearing(CRNU204(100.0, "brg1b"))
    sys2.add_bearing(BB6204(25.0,  "brg2a"))
    sys2.add_bearing(CRNU204(125.0, "brg2b"))
    sys3.add_bearing(BB6204(25.0,  "brg3a"))
    sys3.add_bearing(CRNU204(125.0, "brg3b"))

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
                                    label="load-lifting-mixed-bearings")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    T3      = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope  = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    return {sh.name: sh for sh in (sys1, sys2, sys3)}


# ===========================================================================
# MESH HELPER
# ===========================================================================

def gear_grade_nodes(shaft_sys: ShaftSystem, grade: str = GEAR_GRADE) -> list[float]:
    base_nodes = Mesh1D(shaft_sys).x_nodes
    extra: list[float] = []
    for ge in shaft_sys.gears:
        lo, hi = shaft_sys.gear_extent(ge)
        extra += Grader(lo, hi, base_nodes).get_grade(grade)
    return extra


# ===========================================================================
# MAIN SOLVE LOOP
# ===========================================================================

print(f"\n  +{'─'*(W-4)}+")
print(f"  |  AXISFORGE -- Load-Lifting Gearbox Analysis (mixed bearings){'':<{W-64}}|")
print(f"  |  Face width b = {B_STUDY} mm   P = {P_W} W   n_in = {RPM_IN} rpm{'':<{W-56}}|")
print(f"  |  brgXa = 6204 (locating, ball)   brgXb = NU204 (floating, roller){'':<{W-70}}|")
print(f"  +{'─'*(W-4)}+")

shaft_systems = build_systems(B_STUDY)
library       = SimpleFEMResultsLibrary()
load_results  : dict[str, dict] = {}

for name, shaft_sys in shaft_systems.items():

    print(f"\n\n  +{'─'*(W-4)}+")
    print(f"  |  SHAFT: {name:<{W-12}}|")
    print(f"  +{'─'*(W-4)}+")

    bearings    = {b.label: b for b in shaft_sys.bearings}
    extra_nodes = gear_grade_nodes(shaft_sys)

    # 1. FEM
    print(f"\n  [1/3]  FEM solve  (mesh: {GEAR_GRADE}) ...")
    fem = SimpleFEMSolver()
    fem.solve(shaft_sys, extra_mandatory=extra_nodes)
    ShaftResultsReader(fem, shaft_sys).read(library)
    print(f"         done  --  {len(fem.x_nodes)} nodes")

    # 2. ISO 16281 load distribution — orchestrated across bearing types.
    #    Both DEEP_GROOVE_BALL and CYLINDRICAL_ROLLER are registered in
    #    rolling_bearing_solver._SOLVER_MAP now, so brgXa/brgXb dispatch to
    #    ISO16281BallSolver/ISO16281RollerSolver respectively and merge back
    #    in the order `bearings` was built (brgXa before brgXb).
    print(f"\n  [2/3]  ISO/TS 16281 bearing solve (mixed types) ...")
    solver    = RollingBearingSolver(tol=SOLVER_TOL)
    load_dist = solver.solve(shaft_sys, bearings, library)
    load_results[name] = load_dist
    print(f"         done  --  {len(load_dist)} bearings solved")

    # 3. Per-bearing output
    print(f"\n  [3/3]  Results")

    shaft_res     = library.get(name)
    node_by_label = {n.label: n for n in shaft_res.bearing_nodes}

    # minimum_axial_load is ball-only (BEARING_KITS.supports_axial gates it
    # per bearing below) and lives on ISO16281BallSolver itself, not as a
    # free function — one instance, reused for every ball bearing on this shaft.
    ball_solver = ISO16281BallSolver(tol=SOLVER_TOL)

    for lbl, res in load_dist.items():
        node  = node_by_label[lbl]
        b_obj = bearings[lbl]
        kit   = BEARING_KITS[b_obj.bearing_type]

        stiff = kit.stiffness(b_obj, res, node)

        if kit.supports_axial:
            psi_proj = node.psi_xz * np.cos(res.phi_Fr) + node.psi_xy * np.sin(res.phi_Fr)
            Fa_min, res_min = ball_solver.minimum_axial_load(
                b_obj,
                Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy,
                psi=psi_proj,
                delta_r_init=res.delta_r, delta_a_init=res.delta_a,
            )
        else:
            Fa_min, res_min = None, None

        cap   = kit.capacity_cls.radial(b_obj, Cr=kit.Cr)
        derel = kit.derel_cls.from_distribution(
            b_obj, res, inner_rotating=True, outer_rotating=False)

        print_bearing_result(
            lbl=lbl, b_obj=b_obj, node=node, res=res, kit=kit,
            cap=cap, derel=derel,
            stiff=stiff,
            Fa_min=Fa_min, res_min=res_min,
        )


# ===========================================================================
# PLOT 1 -- deflection diagrams per shaft
# ===========================================================================

for name, shaft_sys in shaft_systems.items():
    results = library.get(name)

    x    = results.x
    v_xz = results.v_xz
    v_xy = results.v_xy
    v    = results.v

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Deflection -- {name}  (b={B_STUDY} mm, mesh={GEAR_GRADE}, mixed bearings)",
        fontsize=11, fontweight="bold",
    )

    for ax, data, label, color in [
        (axes[0], v_xz, "v_xz [mm]",       "#2166ac"),
        (axes[1], v_xy, "v_xy [mm]",       "#d62728"),
        (axes[2], v,    "v resultant [mm]", "#1a1a1a"),
    ]:
        ax.plot(x, data, color=color, lw=1.5)
        ax.axhline(0, color="0.7", lw=0.5)
        for brg in shaft_sys.bearings:
            ax.axvline(brg.position, color="0.5", lw=0.8, ls=":",
                       label="bearing" if brg == shaft_sys.bearings[0] else "")
        for ge in shaft_sys.gears:
            lo, hi = shaft_sys.gear_extent(ge)
            ax.axvspan(lo, hi, alpha=0.08, color="#33a02c",
                       label="gear" if ge == shaft_sys.gears[0] else "")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    axes[-1].set_xlabel("x [mm]", fontsize=8)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# PLOT 2 -- polar load distribution per bearing
# ===========================================================================

def plot_bearing_polar(name: str, shaft_sys: ShaftSystem, load_dist: dict):
    bearings_list = shaft_sys.bearings
    n   = len(bearings_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                             subplot_kw={"projection": "polar"})
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Load distribution -- {name}  (ISO/TS 16281, mixed bearings)",
                 fontsize=11, fontweight="bold")

    for ax, b in zip(axes, bearings_list):
        res  = load_dist[b.label]
        kit  = BEARING_KITS[b.bearing_type]
        dist = kit.contact_distribution(b, res)
        phi, Q = dist[:, 0], dist[:, 1]

        phi_c = np.append(phi, phi[0])
        Q_c   = np.append(Q,   Q[0])

        color = "#2166ac" if kit.supports_axial else "#d62728"
        ax.plot(phi_c, Q_c, color=color, lw=1.5, marker="o", ms=4)
        ax.fill(phi_c, Q_c, color=color, alpha=0.12)

        cap   = kit.capacity_cls.radial(b, Cr=kit.Cr)
        derel = kit.derel_cls.from_distribution(b, res)
        kind  = "ball" if kit.supports_axial else "roller"
        lbl_i, val_i, _, _ = kit.derel_summary(b, derel)

        ax.set_title(
            f"{b.label} ({kind})\n"
            f"Q_max = {Q.max():.0f} N   n_loaded = {int((Q>0).sum())}/{b.Z}\n"
            f"phi(Fr) = {np.degrees(res.phi_Fr):.1f} deg\n"
            f"Q_ci = {cap.Q_ci:.0f} N   {lbl_i} = {val_i:.0f} N",
            fontsize=7,
        )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    plt.tight_layout()
    plt.show()


for name, shaft_sys in shaft_systems.items():
    plot_bearing_polar(name, shaft_sys, load_results[name])

print(f"\n  +{'─'*(W-4)}+")
print(f"  |  Analysis complete.{'':<{W-22}}|")
print(f"  +{'─'*(W-4)}+\n")