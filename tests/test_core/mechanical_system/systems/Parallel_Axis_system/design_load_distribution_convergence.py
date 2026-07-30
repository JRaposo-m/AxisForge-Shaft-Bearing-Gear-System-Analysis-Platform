"""
design_load_distribution_convergence.py

Coupled shaft-bearing analysis for the load-lifting gearbox at fixed b.

Pipeline per shaft:
  1. build ShaftSystem with deep-groove ball bearings (6204)
  2. setup_internal_geometry + compute_hertz_point_contact per bearing
  3. IterativeBearingFEMSolver.solve  -> load distribution + coupled FEM
  4. MeshConvergenceStudy on the converged FEM (bearings EXCLUDED --
     no axial gradient inside the seat, GCI is undefined there)
  5. plots:
       - global deflection diagrams  (v_xz, v_xy, v_res)
       - mesh convergence per interval
       - polar load distribution per bearing (Q_j vs phi_j)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
from axisforge.solvers.machine_elements.bearings.load_distribution import IterativeBearingFEMSolver
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_convergence_study import MeshConvergenceStudy
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader

# ===========================================================================
# PARAMETERS
# ===========================================================================

B_STUDY   = 20.0
P_W       = 1000.0
RPM_IN    = 450.0
r_pulley  = 0.050

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

GCI_THRESHOLD = 0.01
SAFETY_FACTOR = 1.25
MAX_LEVELS    = 8

COUPLING_TOL      = 1.0e-3
COUPLING_MAX_ITER = 30

SHAFT_NAMES = ["shaft1(motor)", "shaft2", "shaft3(pulley)"]

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
# SOLVE — coupled bearing-FEM + convergence study
# ===========================================================================

print(f"Building system (b = {B_STUDY} mm) ...")
shaft_systems = build_systems(B_STUDY)

coupled_solvers: dict[str, IterativeBearingFEMSolver] = {}
load_results:    dict[str, dict] = {}
conv_results:    dict[str, object] = {}

for name, sys in shaft_systems.items():
    print(f"\n=== {name} ===")

    bearings = {b.label: b for b in sys.bearings}

    # 1. coupled bearing-FEM solve
    print("  Coupled bearing-FEM solve ...")
    coupled = IterativeBearingFEMSolver(tol=COUPLING_TOL, max_iter=COUPLING_MAX_ITER)
    load_dist = coupled.solve(sys, bearings)
    coupled_solvers[name] = coupled
    load_results[name]    = load_dist

    for lbl, res in load_dist.items():
        data = coupled.bearing_data[lbl]
        print(f"    {lbl}: Fr_xz={data['Fr_xz']:8.1f}  Fr_xy={data['Fr_xy']:8.1f}  "
            f"Fa={data['Fa']:6.1f}  |  dr_xz={res.delta_r_xz:.3e}  "
            f"dr_xy={res.delta_r_xy:.3e}  da={res.delta_a:.3e}")

    # 2. convergence study on converged FEM — bearings excluded
    print("  Convergence study ...")
    intervals, skipped = MeshConvergenceStudy.intervals_from_shaft_system(sys)
    for msg in skipped:
        print(f"    [SKIP] {msg}")

    # exclude bearing seats: no axial gradient inside the seat, GCI undefined
    bearing_extents = {
        (round(lo, 6), round(hi, 6))
        for b in sys.bearings
        for lo, hi in [sys.bearing_extent(b)]
    }
    intervals = [
        (x_lo, x_hi, label) for x_lo, x_hi, label in intervals
        if (round(x_lo, 6), round(x_hi, 6)) not in bearing_extents
    ]

    if not intervals:
        print("    No intervals — nothing to study.")
        conv_results[name] = None
        continue

    study = MeshConvergenceStudy(
        global_solver = coupled.fem_converged,
        gci_threshold = GCI_THRESHOLD,
        safety_factor = SAFETY_FACTOR,
        max_levels    = MAX_LEVELS,
    )
    conv_results[name] = study.run(sys, intervals)
    conv_results[name].print_report()


# ===========================================================================
# PLOT 1 — global deflection diagrams per shaft
# ===========================================================================

for name, sys in shaft_systems.items():
    slv     = coupled_solvers[name].fem_converged
    results = ShaftResultsReader(slv, sys).read()

    x    = results.x
    v_xz = results.v_xz
    v_xy = results.v_xy
    v    = results.v

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Deflection — {name}  (coupled bearing-FEM, b={B_STUDY} mm)",
                 fontsize=11, fontweight="bold")

    for ax, data, label, color in [
        (axes[0], v_xz, "v_xz [µm]", "#2166ac"),
        (axes[1], v_xy, "v_xy [µm]", "#d62728"),
        (axes[2], v,    "v resultant [µm]", "#1a1a1a"),
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
# PLOT 2 — mesh convergence per interval
# ===========================================================================

for name, result in conv_results.items():
    if result is None:
        continue
    records = list(result.per_load.values())
    if not records:
        continue

    n = len(records)
    fig = plt.figure(figsize=(14, 4 * n))
    fig.suptitle(f"Mesh convergence — {name}  (b={B_STUDY} mm)",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.6, wspace=0.35)

    for row, rec in enumerate(records):
        levels = list(range(len(rec.levels)))
        m_xz  = [m[0] for m in rec.point_metrics_history]
        m_xy  = [m[1] for m in rec.point_metrics_history]
        m_res = [m[2] for m in rec.point_metrics_history]

        for col, (metrics, plane, color) in enumerate([
            (m_xz,  "XZ", "#2166ac"),
            (m_xy,  "XY", "#d62728"),
            (m_res, "resultant", "#1a1a1a"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(levels, metrics, marker="o", color=color, lw=1.5, ms=5)

            for lvl_idx, gci_dict in enumerate(rec.gci_history):
                lvl_fine = lvl_idx + 2
                key = {"XZ": "xz", "XY": "xy", "resultant": "res"}[plane]
                g = gci_dict[key]
                if hasattr(g, "GCI_f_m") and not np.isnan(g.GCI_f_m):
                    ax.annotate(f"GCI={g.GCI_f_m*100:.3f}%",
                                xy=(lvl_fine, metrics[lvl_fine]),
                                xytext=(lvl_fine + 0.05, metrics[lvl_fine]),
                                fontsize=6,
                                color="#d62728" if g.GCI_f_m > GCI_THRESHOLD else "#33a02c",
                                va="center")

            ax.axhline(metrics[-1], color="0.6", lw=0.7, ls="--")
            sc = "#33a02c" if rec.converged else "#d62728"
            st = "converged" if rec.converged else "NOT converged"
            ax.set_title(f"[{rec.label}] {plane}  {st}", fontsize=8, color=sc)
            ax.set_xlabel("level", fontsize=7)
            ax.set_ylabel("v [mm]", fontsize=7)
            ax.set_xticks(levels)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2, lw=0.4)
            ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()


# ===========================================================================
# PLOT 3 — polar load distribution per bearing (XZ plane)
# ===========================================================================

def plot_bearing_polar(name: str, sys: ShaftSystem, load_dist: dict):
    bearings = sys.bearings
    n = len(bearings)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2),
                             subplot_kw={"projection": "polar"})
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Load distribution (XZ) — {name}", fontsize=11, fontweight="bold")

    for ax, b in zip(axes, bearings):
        res = load_dist[b.label]
        phi = b.phi_j
        Q   = b.cp * np.maximum(res.delta_j_xz, 0.0) ** 1.5   # contact force [N]

        phi_closed = np.append(phi, phi[0])
        Q_closed   = np.append(Q,   Q[0])

        ax.plot(phi_closed, Q_closed, color="#2166ac", lw=1.5, marker="o", ms=4)
        ax.fill(phi_closed, Q_closed, color="#2166ac", alpha=0.12)
        ax.set_title(f"{b.label}\nmax Q = {Q.max():.1f} N", fontsize=8)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.tick_params(labelsize=6)

    plt.tight_layout()


for name, sys in shaft_systems.items():
    plot_bearing_polar(name, sys, load_results[name])


plt.show()