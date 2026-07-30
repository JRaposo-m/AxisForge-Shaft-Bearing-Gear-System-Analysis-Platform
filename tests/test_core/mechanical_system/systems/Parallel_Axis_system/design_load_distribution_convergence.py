"""
design_load_distribution_convergence.py

Coupled shaft-bearing analysis for the load-lifting gearbox at fixed b.

Pipeline per shaft:
  1. build ShaftSystem with deep-groove ball bearings (6204)
  2. setup_internal_geometry + compute_hertz_point_contact per bearing
  3. grade_3 nodes injected at gear intervals (mesh fixed, no convergence study)
  4. IterativeBearingFEMSolver.solve  -> load distribution + coupled FEM
  5. plots:
       - global deflection diagrams  (v_xz, v_xy, v_res)
       - polar load distribution per bearing (Q_j vs phi_j, XZ and XY)
       - bearing results summary table
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
from axisforge.solvers.machine_elements.bearings.ISO_16281_ball_bearing import IterativeBearingFEMSolver
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_grade import Grader
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader

# ===========================================================================
# PARAMETERS
# ===========================================================================

B_STUDY  = 20.0
P_W      = 1000.0
RPM_IN   = 450.0
r_pulley = 0.050

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

COUPLING_TOL      = 1.0e-6
COUPLING_MAX_ITER = 100

GEAR_GRADE = "grade_3"   # mesh fixo nas engrenagens — sem estudo de convergência

# --- 6204 deep-groove ball bearing internal geometry ---
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

# ===========================================================================
# BUILD
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
    """Deep-groove ball bearing 6204 with internal geometry attached."""
    b = Bearing(
        d=20.0, D=47.0,
        bearing_type=BearingType.DEEP_GROOVE_BALL,
        designation="6204",
        b=14.0, C=12_700.0, C0=6_550.0,
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
    sys3 = ShaftSystem(s3, name="shaft3(pulley)",speed_rpm=50.0)

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

    T3     = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    return {sh.name: sh for sh in (sys1, sys2, sys3)}


# ===========================================================================
# HELPERS
# ===========================================================================

def gear_grade_nodes(sys: ShaftSystem, grade: str = GEAR_GRADE) -> list[float]:
    """
    Gera nós de refinamento fixo (grade_N) para todos os intervalos de
    engrenagem do ShaftSystem. Usa a mesh base do sistema como input ao Grader.
    """
    base_nodes = Mesh1D(sys).x_nodes
    extra: list[float] = []
    for ge in sys.gears:
        lo, hi = sys.gear_extent(ge)
        extra += Grader(lo, hi, base_nodes).get_grade(grade)
    return extra


# ===========================================================================
# SOLVE — coupled bearing-FEM (grade_3 fixo nas engrenagens)
# ===========================================================================

print(f"Building system (b = {B_STUDY} mm) ...")
shaft_systems = build_systems(B_STUDY)

coupled_solvers: dict[str, IterativeBearingFEMSolver] = {}
load_results:    dict[str, dict] = {}

for name, sys in shaft_systems.items():
    print(f"\n=== {name} ===")

    bearings = {b.label: b for b in sys.bearings}

    # nós grade_3 nos intervalos de engrenagem
    extra_nodes = gear_grade_nodes(sys)

    # FEM base com mesh enriquecida nas engrenagens
    fem_base = SimpleFEMSolver()
    fem_base.solve(sys, extra_mandatory=extra_nodes)

    # coupled bearing-FEM solve (reutiliza fem_base, não reconstrói mesh)
    print("  Coupled bearing-FEM solve ...")
    coupled = IterativeBearingFEMSolver(tol=COUPLING_TOL, max_iter=COUPLING_MAX_ITER)
    load_dist = coupled.solve(sys, bearings, fem=fem_base)
    coupled_solvers[name] = coupled
    load_results[name]    = load_dist

    # --- print resultados ISO 16281 ---
    print(f"\n  {'Bearing':<8}  {'Fr_xz':>8}  {'Fr_xy':>8}  {'Fa':>7}  "
          f"{'Fr_res':>8}  {'dr_xz':>10}  {'dr_xy':>10}  "
          f"{'dr_res':>10}  {'da':>10}")
    print(f"  {'':8}  {'[N]':>8}  {'[N]':>8}  {'[N]':>7}  "
          f"{'[N]':>8}  {'[mm]':>10}  {'[mm]':>10}  "
          f"{'[mm]':>10}  {'[mm]':>10}")
    print("  " + "-" * 95)

    for lbl, res in load_dist.items():
        data = coupled.bearing_data[lbl]
        Fr_res = np.sqrt(data['Fr_xz']**2 + data['Fr_xy']**2)
        print(f"  {lbl:<8}  {data['Fr_xz']:>8.1f}  {data['Fr_xy']:>8.1f}  "
              f"{data['Fa']:>7.1f}  {Fr_res:>8.1f}  "
              f"{res.delta_r_xz:>10.3e}  {res.delta_r_xy:>10.3e}  "
              f"{res.delta_r_res:>10.3e}  {res.delta_a:>10.3e}")

    print()
    for lbl, res in load_dist.items():
        Q_xz = coupled_solvers[name].bearing_data  # acesso ao bearing
        b_obj = next(b for b in sys.bearings if b.label == lbl)
        Q_xz_arr = b_obj.cp * np.maximum(res.delta_j_xz, 0.0) ** 1.5
        Q_xy_arr = b_obj.cp * np.maximum(res.delta_j_xy, 0.0) ** 1.5
        print(f"  {lbl}: ok_xz={res.ok_xz}  res_xz={res.residual_xz:.2e}  "
              f"nfev_xz={res.n_iter_xz}  |  "
              f"ok_xy={res.ok_xy}  res_xy={res.residual_xy:.2e}  "
              f"nfev_xy={res.n_iter_xy}")
        print(f"         max Q_xz={Q_xz_arr.max():.1f} N  "
              f"max Q_xy={Q_xy_arr.max():.1f} N  "
              f"n_loaded_xz={int((res.delta_j_xz > 0).sum())}/{b_obj.Z}  "
              f"n_loaded_xy={int((res.delta_j_xy > 0).sum())}/{b_obj.Z}")


# ===========================================================================
# PLOT 1 — deflection diagrams per shaft
# ===========================================================================

for name, sys in shaft_systems.items():
    slv     = coupled_solvers[name].fem_converged
    results = ShaftResultsReader(slv, sys).read()

    x    = results.x
    v_xz = results.v_xz
    v_xy = results.v_xy
    v    = results.v

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Deflection — {name}  (coupled bearing-FEM, b={B_STUDY} mm, "
                 f"gear mesh={GEAR_GRADE})",
                 fontsize=11, fontweight="bold")

    for ax, data, label, color in [
        (axes[0], v_xz * 1e3, "v_xz [µm]", "#2166ac"),
        (axes[1], v_xy * 1e3, "v_xy [µm]", "#d62728"),
        (axes[2], v    * 1e3, "v resultant [µm]", "#1a1a1a"),
    ]:
        ax.plot(x, data, color=color, lw=1.5, label=label)
        ax.axhline(0, color="0.7", lw=0.5)
        for b in sys.bearings:
            ax.axvline(b.position, color="0.5", lw=0.8, ls=":",
                       label="bearing" if b == sys.bearings[0] else "")
        for ge in sys.gears:
            lo, hi = sys.gear_extent(ge)
            ax.axvspan(lo, hi, alpha=0.08, color="#33a02c",
                       label="gear" if ge == sys.gears[0] else "")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.2, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    axes[-1].set_xlabel("x [mm]", fontsize=8)
    plt.tight_layout()


# ===========================================================================
# PLOT 2 — polar load distribution per bearing (XZ e XY)
# ===========================================================================

def plot_bearing_polar(name: str, sys: ShaftSystem, load_dist: dict):
    bearings = sys.bearings
    n = len(bearings)
    fig, axes = plt.subplots(1, n * 2, figsize=(5 * n * 2, 5),
                             subplot_kw={"projection": "polar"})
    if n == 1:
        axes = list(axes)
    fig.suptitle(f"Load distribution — {name}  (ISO/TS 16281)", fontsize=11,
                 fontweight="bold")

    for i, b in enumerate(bearings):
        res = load_dist[b.label]
        data = coupled_solvers[name].bearing_data[b.label]
        phi = b.phi_j

        Q_xz = b.cp * np.maximum(res.delta_j_xz, 0.0) ** 1.5
        Q_xy = b.cp * np.maximum(res.delta_j_xy, 0.0) ** 1.5

        # ângulo global da carga em cada plano
        phi_load_xz = np.arctan2(data['Fr_xz'], 0)
        phi_load_xy  = np.arctan2(data['Fr_xy'], 0) + np.pi / 2  # +90° plano Y

        # phi_j no referencial global
        phi_xz = phi + phi_load_xz
        phi_xy  = phi + phi_load_xy

        # --- col 0: XZ + XY sobrepostos ---
        ax = axes[i * 2]
        for Q, phi_plot, color, label in [
            (Q_xz, phi_xz, "#2166ac", "XZ"),
            (Q_xy, phi_xy, "#d62728", "XY"),
        ]:
            Q_c   = np.append(Q, Q[0])
            phi_c = np.append(phi_plot, phi_plot[0])
            ax.plot(phi_c, Q_c, color=color, lw=1.5, marker="o", ms=4, label=label)
            ax.fill(phi_c, Q_c, color=color, alpha=0.10)

        ax.set_title(f"{b.label}", fontsize=9, fontweight="bold")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # menos numeros radiais
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                bbox_to_anchor=(1.25, 1.1))

        # --- col 1: resultante bola a bola ---
        # interpola Q_xy no phi_j global do XZ para somar nas mesmas bolas físicas
        Q_xy_interp = np.interp(phi_xz % (2*np.pi),
                                phi_xy  % (2*np.pi),
                                Q_xy,
                                period=2*np.pi)
        Q_res = Q_xz + Q_xy_interp

        # resultante
        phi_orig_c = np.append(phi_xz, phi_xz[0])
        Q_res_c    = np.append(Q_res,  Q_res[0])
        ax2 = axes[i * 2 + 1]
        ax2.plot(phi_orig_c, Q_res_c, color="#1a1a1a", lw=1.5, marker="o", ms=4,
                label="resultante")
        ax2.fill(phi_orig_c, Q_res_c, color="#1a1a1a", alpha=0.10)
        ax2.set_title(f"{b.label} — max Q={Q_res.max():.0f} N", fontsize=9,
                    fontweight="bold")
        ax2.set_theta_zero_location("E")
        ax2.set_theta_direction(1)
        ax2.tick_params(labelsize=6)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.legend(fontsize=8, frameon=False, loc="upper right",
                bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()


for name, sys in shaft_systems.items():
    plot_bearing_polar(name, sys, load_results[name])

plt.show()