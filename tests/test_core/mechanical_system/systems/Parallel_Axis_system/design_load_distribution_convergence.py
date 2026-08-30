"""
design_load_distribution_convergence.py

Coupled shaft-bearing analysis for the load-lifting gearbox at fixed b.

Pipeline per shaft:
  1. build ShaftSystem with deep-groove ball bearings (6204), assembled via
     Bearing.assemble(family=DeepGrooveBallFamily(), catalog=BearingCatalog(...),
     geometry=..., analyses={"point_contact": True})
  2. grade_3 nodes injected at gear intervals (mesh fixed, no convergence study)
  3. SimpleFEMSolver.solve -> ShaftResultsReader.read(library) -> SimpleFEMResultsLibrary
  4. RollingBearingSolver.solve(library) -> dict[label -> list[row results]]
     (dispatches internally to ISO16281BallSolver via dispatch.py)
  5. bearing.family.per_element_dynamic_capacity() (S4.3.1.2) +
     DynamicEquivalentRollingElementLoad.from_bearing_result() (S4.3.2)
  6. plots:
       - global deflection diagrams  (v_xz, v_xy, v_res)
       - polar load distribution per bearing (Q_j vs phi_j_global)

UPDATED, this turn -- migrated off the retired flat DeepGrooveBallBearing
class-hierarchy API onto Bearing.assemble()/BearingFamily. See inline
notes at BB6204() and in the main solve loop for what changed and why.
Two things below are BEST-EFFORT, not yet confirmed against
Ball_Bearing/results.py and Ball_Bearing/postprocessing.py directly (only
against what rolling_bearing_solver.py's own postprocess_and_record()
shows) -- flagged inline with "ASSUMED":
  - BallBearingResult(rows=...) is the correct wrapper for
    bearing_stiffness()/DynamicEquivalentRollingElementLoad, matching
    rolling_bearing_solver.py's own adapter usage.
  - DynamicEquivalentRollingElementLoad.from_bearing_result()'s kwargs
    (inner_rotating/outer_rotating) and returned attributes (.Q_ei/.Q_ee)
    are assumed unchanged from the old .from_distribution() -- only the
    entry-point method name is confirmed to have changed.
If either assumption is wrong you'll get a TypeError/AttributeError at
that call site -- paste it back and we fix it same as the others.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.deep_groove import DeepGrooveBallFamily
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.bearings.bearing import Bearing
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
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.results import (
    BallBearingResult,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.postprocessing import (
    contact_distribution,
    bearing_stiffness,
    DynamicEquivalentRollingElementLoad,
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

# ri/re removed -- DeepGrooveBallFamily.assemble_geometry() computes them
# internally from Dw alone (RI_OVER_DW = RE_OVER_DW = 0.52, ISO 281:2007
# Table 1); it does not accept ri/re as arguments at all.
BB_GEOM = dict(
    Dw  = 7.94,
    Dpw = 33.5,
    Z   = 8,
    s   = 0.010,
    E   = 206_000.0,
    nu  = 0.3,
)

CR_6204 = 12_700.0   # dynamic radial load rating [N]  (ISO 281, 6204)


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


def print_bearing_result(lbl, b_obj, node, res, wrapped, cap, derel,
                         stiff, Fa_min, res_min):
    """res is the single-row raw result (row_results[0]) -- row-level
    scalar quantities like .phi_Fr/.delta_r/.ok live there. wrapped is the
    BallBearingResult(rows=row_results) container -- CONFIRMED needed by
    contact_distribution()/phi_j_global(), which check .is_multirow (a
    plain row result has no such attribute). cap is the (Q_ci, Q_ce)
    tuple returned by bearing.family.per_element_dynamic_capacity()."""

    # contact_distribution() returns a list with one (Z, 2) [phi_j, Q_j]
    # array per row -- single-row DGBB -> dist[0] is that one (Z, 2) array.
    dist     = contact_distribution(b_obj, wrapped)
    row_dist = np.asarray(dist[0])
    phi_arr  = row_dist[:, 0]
    Q_arr    = row_dist[:, 1]
    n_loaded = int((Q_arr > 0).sum())
    Q_max    = float(Q_arr.max())

    psi_xz   = node.psi_xz
    psi_xy   = node.psi_xy
    psi_proj = psi_xz * np.cos(res.phi_Fr) + psi_xy * np.sin(res.phi_Fr)

    # ── header ────────────────────────────────────────────────────────
    print(f"\n  +── Bearing  {lbl}  {'─'*(W-18)}+")

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
    if Fa_min == 0.0:
        print(f"  |    Not required  (da = {res.delta_a:.4e} mm >= 0)")
    else:
        print(f"  |    Fa_min = {Fa_min:.2f} N  ->  da = {res_min.delta_a:.4e} mm at Fa_min")

    # ── ISO/TS 16281 §4.3 ─────────────────────────────────────────────
    Q_ci, Q_ce = cap   # per_element_dynamic_capacity() returns a plain (Q_ci, Q_ce) tuple
    print(f"  |")
    print(f"  |  ISO/TS 16281  S4.3")
    print(f"  |    S4.3.1.2  Q_ci (inner) = {Q_ci:>9.2f} N    "
          f"Q_ce (outer) = {Q_ce:>9.2f} N")
    d0 = derel[0]   # from_bearing_result() returns list[DERL], one per row -- single-row DGBB -> len 1
    print(f"  |    S4.3.2    Q_ei (inner) = {d0.Q_ei:>9.2f} N    "
          f"Q_ee (outer) = {d0.Q_ee:>9.2f} N    "
          f"(inner_rotating = {d0.inner_rotating})")

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


def BB6204(position, label, locating=False):
    """
    UPDATED -- migrated off the retired DeepGrooveBallBearing class
    hierarchy. contact_angle_deg=0.0 has no equivalent in BearingCatalog
    or DeepGrooveBallFamily.assemble_geometry() -- this family only takes
    `s` (diametral clearance), alpha_0 is derived internally from A and s.
    setup_internal_geometry()/compute_hertz_point_contact() are gone too
    -- Bearing.assemble() runs family.assemble_geometry() internally and
    the bearing comes back fully assembled.
    """
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(
            d=20.0, D=47.0, b=14.0,
            C=CR_6204, C0=6_550.0,
            designation="6204",
            label=label,
            position=position,
            arrangement="locating" if locating else "floating",
        ),
        geometry=dict(
            Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
            E=BB_GEOM["E"], s=BB_GEOM["s"], nu=BB_GEOM["nu"], i=1,
        ),
        analyses={"point_contact": True},
    )


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

    sys1.add_bearing(BB6204(20.0,  "brg1a", locating=True))
    sys1.add_bearing(BB6204(100.0, "brg1b"))
    sys2.add_bearing(BB6204(25.0,  "brg2a", locating=True))
    sys2.add_bearing(BB6204(125.0, "brg2b"))
    sys3.add_bearing(BB6204(25.0,  "brg3a", locating=True))
    sys3.add_bearing(BB6204(125.0, "brg3b"))

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
                                    label="load-lifting")
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
print(f"  |  AXISFORGE -- Load-Lifting Gearbox Analysis{'':<{W-48}}|")
print(f"  |  Face width b = {B_STUDY} mm   P = {P_W} W   n_in = {RPM_IN} rpm{'':<{W-56}}|")
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

    # 2. ISO 16281 load distribution — orchestrated across bearing types
    #    (only DEEP_GROOVE_BALL is registered so far, but the dispatch runs
    #    regardless of how many types end up on this shaft)
    #    RollingBearingSolver.solve() returns dict[label -> list[row results]]
    #    (confirmed against rolling_bearing_solver.py -- merged[label] =
    #    local_lib.get(label).rows), NOT a single result object per label.
    print(f"\n  [2/3]  ISO/TS 16281 bearing solve ...")
    solver    = RollingBearingSolver(tol=SOLVER_TOL)
    load_dist = solver.solve(shaft_sys, bearings, library)
    load_results[name] = load_dist
    print(f"         done  --  {len(load_dist)} bearings solved")

    # 3. Per-bearing output
    print(f"\n  [3/3]  Results")

    shaft_res     = library.get(name)
    node_by_label = {n.label: n for n in shaft_res.bearing_nodes}

    # Post-processing utilities and minimum_axial_load are per-type — every
    # bearing here is known to be a DGBB, so call them on ISO16281BallSolver
    # directly rather than through the orchestrator (which only unifies solve()).
    ball_solver = ISO16281BallSolver(tol=SOLVER_TOL)

    for lbl, row_results in load_dist.items():
        node  = node_by_label[lbl]
        b_obj = bearings[lbl]

        # Single-row DGBB -> exactly one element in .rows. row-level
        # quantities (phi_Fr, ok, delta_r, delta_a, residual, n_iter) live
        # on that element; bearing_stiffness()/DynamicEquivalentRolling-
        # ElementLoad take the wrapped container instead (ASSUMED, see
        # module docstring -- matches rolling_bearing_solver.py's own
        # postprocess_and_record(), not yet confirmed against
        # results.py/postprocessing.py directly).
        row     = row_results[0]
        wrapped = BallBearingResult(rows=row_results)

        stiff = bearing_stiffness(
            b_obj, wrapped, Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=node.Fa)

        psi_proj = node.psi_xz * np.cos(row.phi_Fr) + node.psi_xy * np.sin(row.phi_Fr)
        Fa_min, res_min = ball_solver.minimum_axial_load(
            b_obj,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy,
            psi=psi_proj,
            delta_r_init=row.delta_r, delta_a_init=row.delta_a,
        )

        cap   = b_obj.family.per_element_dynamic_capacity(b_obj, Cr=CR_6204)
        derel = DynamicEquivalentRollingElementLoad.from_bearing_result(
            b_obj, wrapped, inner_rotating=True, outer_rotating=False)

        print_bearing_result(
            lbl=lbl, b_obj=b_obj, node=node, res=row, wrapped=wrapped,
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
        f"Deflection -- {name}  (b={B_STUDY} mm, mesh={GEAR_GRADE})",
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
    fig.suptitle(f"Load distribution -- {name}  (ISO/TS 16281)",
                 fontsize=11, fontweight="bold")

    for ax, b in zip(axes, bearings_list):
        row_results = load_dist[b.label]
        row         = row_results[0]
        wrapped     = BallBearingResult(rows=row_results)

        dist       = contact_distribution(b, wrapped)
        row_dist   = np.asarray(dist[0])
        phi, Q     = row_dist[:, 0], row_dist[:, 1]

        phi_c = np.append(phi, phi[0])
        Q_c   = np.append(Q,   Q[0])

        ax.plot(phi_c, Q_c, color="#2166ac", lw=1.5, marker="o", ms=4)
        ax.fill(phi_c, Q_c, color="#2166ac", alpha=0.12)

        Q_ci, Q_ce = b.family.per_element_dynamic_capacity(b, Cr=CR_6204)
        derel = DynamicEquivalentRollingElementLoad.from_bearing_result(b, wrapped)

        ax.set_title(
            f"{b.label}\n"
            f"Q_max = {Q.max():.0f} N   n_loaded = {int((Q>0).sum())}/{b.Z}\n"
            f"phi(Fr) = {np.degrees(row.phi_Fr):.1f} deg\n"
            f"Q_ci = {Q_ci:.0f} N   Q_ei = {derel[0].Q_ei:.0f} N",
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