"""
design_load_distribution_convergence.py

Coupled shaft-bearing analysis for the load-lifting gearbox at fixed b.

Pipeline per shaft:
  1. build ShaftSystem with deep-groove ball bearings (6204)
  2. setup_internal_geometry + compute_hertz_point_contact per bearing
  3. grade_3 nodes injected at gear intervals (mesh fixed, no convergence study)
  4. SimpleFEMSolver.solve -> ShaftResultsReader.read(library) -> SimpleFEMResultsLibrary
  5. IterativeBearingFEMSolver.solve(library) -> load distribution
  6. RollingElementCapacity (§4.3.1.2) + DynamicEquivalentRollingElementLoad (§4.3.2)
  7. plots:
       - global deflection diagrams  (v_xz, v_xy, v_res)
       - polar load distribution per bearing (Q_j vs phi_j_global)

Changes vs. original:
  - loop variable renamed shaft_sys (avoids shadowing stdlib `sys`)
  - RollingElementCapacity / DynamicEquivalentRollingElementLoad called per bearing
  - matplotlib backend: interactive (plt.show())
  - b_obj bug fixed in plot_bearing_polar (was referencing outer-scope variable)
  - DEBUG block added in main loop for brg1a / shaft1(motor) — remove after validation
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
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
from axisforge.solvers.machine_elements.bearings.ISO_16281_ball_bearing import (
    IterativeBearingFEMSolver,
    RollingElementCapacity,
    DynamicEquivalentRollingElementLoad,
)
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_grade import Grader


# ===========================================================================
# PARAMETERS
# ===========================================================================

B_STUDY  = 20.0
P_W      = 1000.0
RPM_IN   = 450.0
r_pulley = 0.050          # [m]

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

COUPLING_TOL      = 1.0e-6
COUPLING_MAX_ITER = 100

GEAR_GRADE = "grade_3"

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


# ===========================================================================
# PRINT HELPERS
# ===========================================================================

W = 72   # total console width

def _hdr(text: str, char: str = "═") -> str:
    return f"\n  ┌{'─'*(W-4)}┐\n  │  {text:<{W-6}}│\n  └{'─'*(W-4)}┘"

def _section(text: str) -> str:
    pad = W - 6 - len(text)
    return f"\n  ┌── {text} {'─'*pad}┐"

def _row(label: str, value: str, width: int = 28) -> str:
    return f"  │  {label:<{width}}  {value}"

def _sep() -> str:
    return f"  │  {'─'*(W-6)}"

def _end() -> str:
    return f"  └{'─'*(W-4)}┘"

def _ok(flag: bool) -> str:
    return "✓ converged" if flag else "✗ NOT converged"

def _tick(val: float, threshold: float = 0.0) -> str:
    return "▲" if val > threshold else "·"


def print_bearing_result(
    lbl:      str,
    b_obj,
    node,
    res,
    cap,
    derel,
    Kr_xz:   float,
    Kr_xy:   float,
    Ka:       float,
    Fa_min:   float,
    res_min,
) -> None:
    """Structured per-bearing console output."""

    dist     = IterativeBearingFEMSolver.contact_distribution(b_obj, res)
    phi_arr  = dist[:, 0]
    Q_arr    = dist[:, 1]
    n_loaded = int((Q_arr > 0).sum())
    Q_max    = float(Q_arr.max())

    psi_xz   = node.psi_xz
    psi_xy   = node.psi_xy
    psi_proj = psi_xz * np.cos(res.phi_Fr) + psi_xy * np.sin(res.phi_Fr)

    # ── header ────────────────────────────────────────────────────────
    print(f"\n  ┌── Bearing  {lbl}  {'─'*(W-18)}┐")

    # ── reactions ─────────────────────────────────────────────────────
    print(f"  │")
    print(f"  │  {'REACTIONS':}")
    print(f"  │    Fr  = {node.Fr:>10.2f} N       "
          f"Fr_xz = {node.Fr_xz:>10.2f} N    Fr_xy = {node.Fr_xy:>10.2f} N")
    print(f"  │    Fa  = {node.Fa:>10.2f} N")

    # ── solver status ─────────────────────────────────────────────────
    print(f"  │")
    print(f"  │  {'SOLVER':}")
    print(f"  │    Status   {_ok(res.ok):<20}  "
          f"residual = {res.residual:.2e} N    nfev = {res.n_iter}")
    print(f"  │    δr = {res.delta_r:.4e} mm    "
          f"δa = {res.delta_a:.4e} mm    "
          f"φ(Fr) = {np.degrees(res.phi_Fr):>7.2f}°")

    # ── misalignment ──────────────────────────────────────────────────
    print(f"  │")
    print(f"  │  {'MISALIGNMENT  (shaft slope across seat)':}")
    print(f"  │    ψ_xz = {np.degrees(psi_xz):>9.5f}°    "
          f"ψ_xy = {np.degrees(psi_xy):>9.5f}°    "
          f"ψ_proj(Fr) = {np.degrees(psi_proj):>9.5f}°")

    # ── stiffness ─────────────────────────────────────────────────────
    def _fmt_k(k: float) -> str:
        return f"{k:.4e}" if k != float("inf") else "    ∞ (rigid)"

    print(f"  │")
    print(f"  │  {'SECANT STIFFNESS':}")
    print(f"  │    Kr_xz = {_fmt_k(Kr_xz):>14} N/mm    "
          f"Kr_xy = {_fmt_k(Kr_xy):>14} N/mm    "
          f"Ka = {_fmt_k(Ka):>14} N/mm")

    # ── axial preload ─────────────────────────────────────────────────
    print(f"  │")
    print(f"  │  {'MINIMUM AXIAL PRELOAD':}")
    if Fa_min == 0.0:
        print(f"  │    Not required  (δa = {res.delta_a:.4e} mm ≥ 0)")
    else:
        print(f"  │    Fa_min = {Fa_min:.2f} N   →   δa = {res_min.delta_a:.4e} mm at Fa_min")

    # ── ISO/TS 16281 §4.3 ─────────────────────────────────────────────
    print(f"  │")
    print(f"  │  {'ISO/TS 16281  §4.3':}")
    print(f"  │    §4.3.1.2  Q_ci (inner) = {cap.Q_ci:>9.2f} N    "
          f"Q_ce (outer) = {cap.Q_ce:>9.2f} N")
    print(f"  │    §4.3.2    Q_ei (inner) = {derel.Q_ei:>9.2f} N    "
          f"Q_ee (outer) = {derel.Q_ee:>9.2f} N    "
          f"(inner rotating = {derel.inner_rotating})")

    # ── load distribution ─────────────────────────────────────────────
    print(f"  │")
    print(f"  │  {'LOAD DISTRIBUTION   '}"
          f"({n_loaded}/{b_obj.Z} elements loaded    Q_max = {Q_max:.1f} N)")
    print(f"  │    {'j':>3}   {'φ_global':>9}   {'Q_j':>10}   {'':}")
    print(f"  │    {'─'*3}   {'─'*9}   {'─'*10}")
    for j, (phi_i, Q_i) in enumerate(zip(phi_arr, Q_arr)):
        bar   = "█" * int(Q_i / max(Q_max, 1) * 20) if Q_i > 0 else "·"
        flag  = "  ← max" if Q_i == Q_max and Q_max > 0 else ""
        print(f"  │    {j:>3}   {np.degrees(phi_i):>7.2f}°   {Q_i:>10.2f} N   {bar}{flag}")

    print(f"  └{'─'*(W-4)}┘")


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
    b = Bearing(
        d=20.0, D=47.0,
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        designation="6204",
        b=14.0, C=CR_6204, C0=6_550.0,
        arrangement="locating" if locating else "floating",
        contact_angle_deg=0.0,
        label=label,
        position=position,
    )
    b.setup_internal_geometry(
        ri=BB_GEOM["ri"], re=BB_GEOM["re"],
        Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"],
        Z=BB_GEOM["Z"],  s=BB_GEOM["s"],
        E=BB_GEOM["E"],  nu=BB_GEOM["nu"],
    )
    b.compute_hertz_point_contact()
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
# DEBUG HELPER — Q_ci / Q_ce intermediate values (§4.3.1.2)
# Remove after validation.
# ===========================================================================

def _debug_Qci_Qce(b_obj, Cr: float, lbl: str) -> None:
    """
    Recomputes Q_ci / Q_ce step-by-step and compares with RollingElementCapacity.
    Prints all intermediates for manual cross-check against ISO/TS 16281 §4.3.1.2.
    """
    Z            = b_obj.Z
    alpha_i      = b_obj.alpha_0
    ri           = b_obj.ri
    re           = b_obj.re
    Dw           = b_obj.Dw
    Dpw          = b_obj.Dpw
    gamma        = Dw * np.cos(alpha_i) / Dpw
    radii_ratio  = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))
    bracket      = 1.044 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72 * radii_ratio ** 0.41
    cos07        = np.cos(alpha_i) ** 0.7
    inner        = bracket ** (10.0 / 3.0)
    outer        = bracket ** (-10.0 / 3.0)
    denom_ci     = 0.407 * Z * cos07
    denom_ce     = 0.389 * Z * cos07
    factor_ci    = (1.0 + inner) ** 0.3
    factor_ce    = (1.0 + outer) ** 0.3
    Q_ci_check   = (Cr / denom_ci) * factor_ci
    Q_ce_check   = (Cr / denom_ce) * factor_ce

    cap = RollingElementCapacity.radial(b_obj, Cr=Cr)

    print(f"\n  ┌── DEBUG Q_ci/Q_ce — {lbl} {'─'*36}┐")
    print(f"  │  INPUT")
    print(f"  │    Cr           = {Cr:.2f} N")
    print(f"  │    Z            = {Z}")
    print(f"  │    alpha_0      = {np.degrees(alpha_i):.6f}°")
    print(f"  │    ri           = {ri:.6f} mm")
    print(f"  │    re           = {re:.6f} mm")
    print(f"  │    Dw           = {Dw:.6f} mm")
    print(f"  │    Dpw          = {Dpw:.6f} mm")
    print(f"  │  INTERMEDIATE")
    print(f"  │    gamma          = {gamma:.8f}   [Dw·cos(α₀)/Dpw, eq.(4)]")
    print(f"  │    radii_ratio    = {radii_ratio:.8f}   [ri/re·(2re−Dw)/(2ri−Dw)]")
    print(f"  │    bracket        = {bracket:.8f}   [1.044·(γ-term)^1.72·ratio^0.41]")
    print(f"  │    cos(α₀)^0.7   = {cos07:.8f}")
    print(f"  │    bracket^(+10/3) = {inner:.8f}   → Q_ci eq.(19)")
    print(f"  │    bracket^(−10/3) = {outer:.8f}   → Q_ce eq.(20)")
    print(f"  │  CAPACITY")
    print(f"  │    denom_ci  = 0.407·{Z}·cos^0.7 = {denom_ci:.6f}")
    print(f"  │    denom_ce  = 0.389·{Z}·cos^0.7 = {denom_ce:.6f}")
    print(f"  │    factor_ci = (1 + {inner:.6f})^0.3 = {factor_ci:.8f}")
    print(f"  │    factor_ce = (1 + {outer:.6f})^0.3 = {factor_ce:.8f}")
    print(f"  │  RESULT vs CODE")
    print(f"  │    Q_ci  check = {Q_ci_check:.4f} N   code = {cap.Q_ci:.4f} N   "
          f"Δ = {abs(Q_ci_check - cap.Q_ci):.4f} N")
    print(f"  │    Q_ce  check = {Q_ce_check:.4f} N   code = {cap.Q_ce:.4f} N   "
          f"Δ = {abs(Q_ce_check - cap.Q_ce):.4f} N")
    print(f"  └{'─'*62}┘")


# ===========================================================================
# MAIN SOLVE LOOP
# ===========================================================================

print(f"\n{'═'*W}")
print(f"  AXISFORGE — Load-Lifting Gearbox Analysis")
print(f"  Face width b = {B_STUDY} mm   P = {P_W} W   n_in = {RPM_IN} rpm")
print(f"{'═'*W}")

shaft_systems = build_systems(B_STUDY)
library       = SimpleFEMResultsLibrary()
load_results  : dict[str, dict] = {}

for name, shaft_sys in shaft_systems.items():

    print(f"\n\n{'═'*W}")
    print(f"  SHAFT: {name}")
    print(f"{'═'*W}")

    bearings    = {b.label: b for b in shaft_sys.bearings}
    extra_nodes = gear_grade_nodes(shaft_sys)

    # 1. FEM
    print(f"\n  [ 1/3 ]  FEM solve  (mesh grade: {GEAR_GRADE}) ...")
    fem = SimpleFEMSolver()
    fem.solve(shaft_sys, extra_mandatory=extra_nodes)
    ShaftResultsReader(fem, shaft_sys).read(library)
    print(f"          done  —  {len(fem.x_nodes)} nodes")

    # 2. ISO 16281 load distribution
    print(f"\n  [ 2/3 ]  ISO/TS 16281 coupled bearing-FEM solve ...")
    coupled   = IterativeBearingFEMSolver(tol=COUPLING_TOL, max_iter=COUPLING_MAX_ITER)
    load_dist = coupled.solve(shaft_sys, bearings, library)
    load_results[name] = load_dist
    print(f"          done  —  {len(load_dist)} bearings solved")

    # 3. Per-bearing output
    print(f"\n  [ 3/3 ]  Results")

    shaft_res     = library.get(name)
    node_by_label = {n.label: n for n in shaft_res.bearing_nodes}

    for lbl, res in load_dist.items():
        node  = node_by_label[lbl]
        b_obj = bearings[lbl]

        Kr_xz, Kr_xy, Ka = IterativeBearingFEMSolver.bearing_stiffness(
            b_obj, res, Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=node.Fa)

        Fa_min, res_min = coupled.minimum_axial_load(
            b_obj,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy,
            psi_xz=node.psi_xz, psi_xy=node.psi_xy,
            delta_r_init=res.delta_r, delta_a_init=res.delta_a,
        )

        cap   = RollingElementCapacity.radial(b_obj, Cr=CR_6204)
        derel = DynamicEquivalentRollingElementLoad.from_distribution(
            b_obj, res, inner_rotating=True, outer_rotating=False)

        # ------------------------------------------------------------------
        # DEBUG — Q_ci / Q_ce intermediate check for brg1a only.
        # Remove this block after validation is complete.
        # ------------------------------------------------------------------
        if lbl == "brg1a" and name == "shaft1(motor)":
            _debug_Qci_Qce(b_obj, Cr=CR_6204, lbl=lbl)
        # ------------------------------------------------------------------

        print_bearing_result(
            lbl=lbl, b_obj=b_obj, node=node, res=res,
            cap=cap, derel=derel,
            Kr_xz=Kr_xz, Kr_xy=Kr_xy, Ka=Ka,
            Fa_min=Fa_min, res_min=res_min,
        )


# ===========================================================================
# PLOT 1 — deflection diagrams per shaft
# ===========================================================================

for name, shaft_sys in shaft_systems.items():
    results = library.get(name)

    x    = results.x
    v_xz = results.v_xz
    v_xy = results.v_xy
    v    = results.v

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"Deflection — {name}  (coupled bearing-FEM, b={B_STUDY} mm, "
        f"gear mesh={GEAR_GRADE})",
        fontsize=11, fontweight="bold",
    )

    for ax, data, label, color in [
        (axes[0], v_xz, "v_xz [mm]",       "#2166ac"),
        (axes[1], v_xy, "v_xy [mm]",       "#d62728"),
        (axes[2], v,    "v resultant [mm]", "#1a1a1a"),
    ]:
        ax.plot(x, data, color=color, lw=1.5, label=label)
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
# PLOT 2 — polar load distribution per bearing
# ===========================================================================

def plot_bearing_polar(name: str, shaft_sys: ShaftSystem, load_dist: dict):
    bearings = shaft_sys.bearings
    n = len(bearings)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                             subplot_kw={"projection": "polar"})
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Load distribution — {name}  (ISO/TS 16281)", fontsize=11,
                 fontweight="bold")

    for ax, b in zip(axes, bearings):
        res  = load_dist[b.label]
        dist = IterativeBearingFEMSolver.contact_distribution(b, res)
        phi, Q = dist[:, 0], dist[:, 1]

        phi_c = np.append(phi, phi[0])
        Q_c   = np.append(Q,   Q[0])

        ax.plot(phi_c, Q_c, color="#2166ac", lw=1.5, marker="o", ms=4)
        ax.fill(phi_c, Q_c, color="#2166ac", alpha=0.12)

        cap   = RollingElementCapacity.radial(b, Cr=CR_6204)
        derel = DynamicEquivalentRollingElementLoad.from_distribution(b, res)

        ax.set_title(
            f"{b.label}\n"
            f"Q_max = {Q.max():.0f} N   n_loaded = {int((Q>0).sum())}/{b.Z}\n"
            f"φ(Fr) = {np.degrees(res.phi_Fr):.1f}°\n"
            f"Q_ci = {cap.Q_ci:.0f} N   Q_ei = {derel.Q_ei:.0f} N",
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

print(f"\n{'═'*W}")
print(f"  Analysis complete.")
print(f"{'═'*W}\n")