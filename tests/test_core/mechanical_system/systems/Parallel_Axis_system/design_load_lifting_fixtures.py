"""
design_load_lifting_fixtures.py

Rewrite of design_load_lifting_shafts.py using AxisForge fixtures.

The script declares all geometry explicitly using fixtures, then passes
pre-built objects to build_systems() for assembly only.

Pipeline:
  1. Build GearPairFixture per stage (gear geometry + meshing)
  2. Build ShaftFixture per shaft (shaft geometry)
  3. Build Bearings per shaft (positioned, solver-ready)
  4. Declare StageSpec per stage (phi_deg, gear axial positions)
  5. Call build_systems() -- assembly + resolve only
  6. Add external loads on shaft_systems
  7. Close torque equilibrium
  8. FEM solve
  9. Post-process + output + plots

Engineering content (unchanged from original):
  - 2-stage spur gearbox (z1/z2, z3/z4), phi_deg=270 for both stages
  - 3 stepped shafts with real Shoulder fillets
  - DGBB 6304 bearings (d=20 mm seats)
  - Motor torque equilibrium on shaft1 (assumption A2)
  - Pulley load resistance on shaft3 (assumption A3)
  - SimpleFEMSolver (Timoshenko) on all three shafts

ASSUMPTIONS:
  A1. Bearing seats at smaller step diameter; body section stepped UP.
  A2. Motor coupling at x=0: TorqueLoad(-T_in) closes T(x) on shaft1.
  A3. Pulley resistance on shaft3: TorqueLoad(+T3) at pulley position.
  A4. DGBB 6304 (d=20 mm, b=15 mm) used on all three shafts.

CONSOLE OUTPUT: ASCII only (Windows PowerShell / cp1252 safe).
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

# --- core ---
from axisforge.core.loads import RadialLoad, TorqueLoad
from axisforge.core.mechanical_system.Parallel_Axis_systems.schematic import (
    draw_gear_system, draw_shaft_detail, INK,
)

# --- solvers ---
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import (
    SimpleFEMSolver,
)

# --- fixtures ---
from axisforge.fixtures.gears.spur_helical import make_spur_helical, GearPairFixture
from axisforge.fixtures.shafts.shaft_fixture import make_stepped_3section
from axisforge.fixtures.bearings.dgbb_generic import DGBB_6304
from axisforge.fixtures.systems.linear_gear_chain import (
    build_systems, StageSpec,
)


# ===========================================================================
# PARAMETERS
# ===========================================================================

P_W      = 1000.0   # input power [W]
RPM_IN   = 450.0    # input shaft speed [rpm]
G        = 9.81     # gravitational acceleration [m/s^2]
R_PULLEY = 0.050    # pulley radius [m]
PHI_DEG  = 270.0    # line-of-centres angle, both stages [deg]


# ===========================================================================
# 1. GEAR PAIRS
# ===========================================================================

z1 = make_spur_helical(z=20, mn=2.0, b=20.0, position=55.0,
                        label="z1", material_id="42CrMo4")
z2 = make_spur_helical(z=60, mn=2.0, b=20.0, position=95.0,
                        label="z2", material_id="42CrMo4")
z3 = make_spur_helical(z=20, mn=2.0, b=20.0, position=55.0,
                        label="z3", material_id="42CrMo4")
z4 = make_spur_helical(z=60, mn=2.0, b=20.0, position=55.0,
                        label="z4", material_id="42CrMo4")

stage1_pair = GearPairFixture.from_fixtures(z1, z2, label="stage1")
stage2_pair = GearPairFixture.from_fixtures(z3, z4, label="stage2")


# ===========================================================================
# 2. SHAFT FIXTURES
# ===========================================================================

shaft1_fix = make_stepped_3section(
    total_length=120.0, d_seat=20.0, d_body=25.0,
    l_seat_a=30.0, l_seat_b=30.0, fillet_r=1.5,
    material_id="AISI_1045", name="shaft1(motor)",
)
shaft2_fix = make_stepped_3section(
    total_length=150.0, d_seat=20.0, d_body=28.0,
    l_seat_a=35.0, l_seat_b=35.0, fillet_r=2.0,
    material_id="AISI_1045", name="shaft2",
)
shaft3_fix = make_stepped_3section(
    total_length=150.0, d_seat=20.0, d_body=25.0,
    l_seat_a=35.0, l_seat_b=35.0, fillet_r=1.5,
    material_id="AISI_1045", name="shaft3(pulley)",
)


# ===========================================================================
# 3. BEARINGS
# ===========================================================================
# DGBB 6304: d=20 mm bore matches d_seat on all three shafts.
# make_ready() calls setup_internal_geometry() + compute_hertz_point_contact().

def dgbb(position: float, label: str, locating: bool = False):
    arrangement = "locating" if locating else "floating"
    return DGBB_6304.with_position(position).with_arrangement(arrangement).with_label(label).make_ready()

bearings_shaft1 = [dgbb(20.0,  "brg1a", locating=True), dgbb(100.0, "brg1b")]
bearings_shaft2 = [dgbb(25.0,  "brg2a", locating=True), dgbb(125.0, "brg2b")]
bearings_shaft3 = [dgbb(25.0,  "brg3a", locating=True), dgbb(125.0, "brg3b")]


# ===========================================================================
# 4. STAGE SPECS
# ===========================================================================
# StageSpec carries only what build_systems() needs for assembly:
# phi_deg and axial positions of the gears on their shafts.
# All gear geometry is already in stage1_pair / stage2_pair.

stage_specs = [
    StageSpec(pos_driver=55.0, pos_driven=95.0, phi_deg=PHI_DEG, label="stage1"),
    StageSpec(pos_driver=55.0, pos_driven=55.0, phi_deg=PHI_DEG, label="stage2"),
]


# ===========================================================================
# 5. BUILD AND RESOLVE
# ===========================================================================

print("=" * 74)
print("  AxisForge -- load-lifting gearbox (fixtures)")
print("=" * 74)

result = build_systems(
    shaft_fixtures=[shaft1_fix, shaft2_fix, shaft3_fix],
    pairs=[stage1_pair, stage2_pair],
    bearings_per_shaft=[bearings_shaft1, bearings_shaft2, bearings_shaft3],
    stage_specs=stage_specs,
    P_W=P_W,
    rpm_in=RPM_IN,
    rotation_dir=1,
    speed_rpm_per_shaft=[450.0, 150.0, 50.0],
    label="load-lifting",
)

gearbox          = result.gearbox
sys1, sys2, sys3 = result.shaft_systems
stage1           = result.pairs[0].meshing
stage2           = result.pairs[1].meshing


# ===========================================================================
# 6. GEAR AND SYSTEM SUMMARY
# ===========================================================================

result.print_gear_summary()

print("\n" + "-" * 74)
print("  GEAR FIXTURES")
print("-" * 74)
for i in range(len(stage_specs)):
    drv  = result.driver_fixture(i)
    drn  = result.driven_fixture(i)
    pair = result.pairs[i]
    print(f"  stage {i} ({pair.label})")
    print(f"    driver : {drv.label}  z={drv.z}  mn={drv.mn}  "
          f"b={drv.b} mm  x={drv.x:.4f}  d={drv.gear.d:.4f} mm  "
          f"da={drv.gear.da:.4f} mm  mat={drv.material_id}")
    print(f"    driven : {drn.label}  z={drn.z}  mn={drn.mn}  "
          f"b={drn.b} mm  x={drn.x:.4f}  d={drn.gear.d:.4f} mm  "
          f"da={drn.gear.da:.4f} mm  mat={drn.material_id}")

print("\n" + "-" * 74)
print("  GEAR ELEMENTS")
print("-" * 74)
for i in range(len(stage_specs)):
    ge_drv, ge_drn = result.gear_elements[i]
    print(f"  stage {i} : {ge_drv!r}")
    print(f"           : {ge_drn!r}")

print("\n" + "-" * 74)
print("  MESH LINKS")
print("-" * 74)
for lnk in result.links:
    print(f"  {lnk!r}")


# ===========================================================================
# SCHEMATIC
# ===========================================================================

fig_schem, ax_schem = plt.subplots(figsize=(12, 5))
draw_gear_system(ax_schem, gearbox, axis="z")
fig_schem.suptitle("Gearbox schematic -- system (1D)", fontsize=12, fontweight="bold")
fig_schem.tight_layout()

for sh in (sys1, sys2, sys3):
    fig_d, ax_d = plt.subplots(figsize=(9, 2.6))
    draw_shaft_detail(ax_d, sh)
    fig_d.suptitle(f"{sh.name} -- detail (2D)", fontsize=11, fontweight="bold")
    fig_d.tight_layout()


# ===========================================================================
# 7. EXTERNAL LOADS
# ===========================================================================

T3     = gearbox._resolved[id(sys3)]["T_out_Nm"]
F_rope = T3 / R_PULLEY

sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))


# ===========================================================================
# CONSOLE SUMMARY
# ===========================================================================

def dump_loads(shaft) -> None:
    print(f"\n  loads on {shaft.name}:")
    for ld in shaft.loads:
        kind = type(ld).__name__
        if hasattr(ld, "theta_deg"):
            print(f"    {kind:<11} @x={ld.position:6.1f}mm  "
                  f"|F|={ld.magnitude:8.1f}N  "
                  f"theta={ld.theta_deg:6.1f} deg  [{ld.source}] {ld.label}")
        else:
            print(f"    {kind:<11} @x={ld.position:6.1f}mm  "
                  f"val={ld.magnitude:11.4f}  [{ld.source}] {ld.label}")


print("\n" + "-" * 74)
print("  RESOLVED TRANSMISSION")
print("-" * 74)
print(gearbox.summary())

print("\n  shaft              speed[rpm]  T_out[N.m]  rot   axis(y,z)[mm]")
for sh in (sys1, sys2, sys3):
    r    = gearbox._resolved[id(sh)]
    y, z = r["shaft_position"]
    print(f"  {sh.name:<17} {sh.speed_rpm:>7.1f}   {r['T_out_Nm']:>9.3f}   "
          f"{int(r['rotation_dir']):>+3d}   ({y:6.1f},{z:7.1f})")

print("\n" + "-" * 74)
print("  MESH FORCES PER STAGE")
print("-" * 74)
for st, drv_shaft, drv_label in (
    (stage1, sys1, "stage1_driver"),
    (stage2, sys2, "stage2_driver"),
):
    fr = next(l.magnitude for l in drv_shaft.radial_loads
              if l.source == "gear_mesh" and l.label == f"{drv_label}:Fr")
    ft = next(l.magnitude for l in drv_shaft.radial_loads
              if l.source == "gear_mesh" and l.label == f"{drv_label}:Ft")
    Fn = ft / math.cos(st.alphatw)
    print(f"  {st.label:<22} Ft={ft:8.1f} N  Fr={fr:8.1f} N  Fn={Fn:8.1f} N")

print("\n" + "-" * 74)
print("  PER-SHAFT LOAD DUMP")
print("-" * 74)
for sh in (sys1, sys2, sys3):
    dump_loads(sh)

print("\n" + "-" * 74)
print("  OUTPUT / LIFTING")
print("-" * 74)
hoist = sys3.speed_rpm / 60.0 * math.pi * (2 * R_PULLEY)
print(f"  output torque (shaft3) : {T3:.2f} N.m  @ {sys3.speed_rpm:.0f} rpm")
print(f"  wire-rope tension      : {F_rope:.1f} N")
print(f"  liftable load (ideal)  : {F_rope / G:.1f} kg   (no losses, static)")
print(f"  hoist speed            : {hoist * 1000:.1f} mm/s")


# ===========================================================================
# 8. TORQUE EQUILIBRIUM  (assumptions A2, A3)
# ===========================================================================

T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))


# ===========================================================================
# 9. FEM SOLVE
# ===========================================================================

print("\n" + "=" * 74)
print("  TORQUE EQUILIBRIUM CHECK (net must be ~0 N.m per shaft)")
print("=" * 74)
check_solver = SimpleFEMSolver(theory="timoshenko")
for sh in (sys1, sys2, sys3):
    errs   = check_solver.validate_torsion_equilibrium(sh)
    status = "OK" if not errs else " / ".join(errs)
    print(f"  {sh.name:<18} {status}")

solvers: dict[str, SimpleFEMSolver] = {}
for sh in (sys1, sys2, sys3):
    s = SimpleFEMSolver(theory="timoshenko")
    s.solve(sh)
    solvers[sh.name] = s

print("\n" + "=" * 74)
print("  GearSystem.validate()")
print("=" * 74)
errs = gearbox.validate()
print("  OK -- no errors" if not errs else "\n".join(f"  - {e}" for e in errs))


# ===========================================================================
# POST-PROCESS
# ===========================================================================

def recover_diagrams(solver: SimpleFEMSolver, shaft_system) -> dict:
    from axisforge.mesh.oneD.shaft.Elements.Timoshenko_Selective_Integration.timoshenko import (
        TimoshenkoBeam,
    )
    beam = TimoshenkoBeam()
    n    = len(solver.x_nodes)
    M_xz = np.zeros(n); M_xy = np.zeros(n)
    V_xz = np.zeros(n); V_xy = np.zeros(n)
    N_ax = np.zeros(n)
    done = set()

    for elem in solver.elements:
        dofs = [
            3 * elem.idx_node_1,     3 * elem.idx_node_1 + 1,
            3 * elem.idx_node_1 + 2, 3 * elem.idx_node_2,
            3 * elem.idx_node_2 + 1, 3 * elem.idx_node_2 + 2,
        ]
        Ke   = beam.stiffness_element(elem)
        f_xz = Ke @ solver.d_total_xz[dofs]
        f_xy = Ke @ solver.d_total_xy[dofs]

        for local, node_id in ((0, elem.idx_node_1), (1, elem.idx_node_2)):
            if node_id in done:
                continue
            done.add(node_id)
            sl = slice(0, 3) if local == 0 else slice(3, 6)
            N_ax[node_id], V_xz[node_id], M_xz[node_id] = f_xz[sl]
            _,              V_xy[node_id], M_xy[node_id] = f_xy[sl]

    v_xz    = np.array([solver.d_total_xz[3 * i + 1] for i in range(n)])
    v_xy    = np.array([solver.d_total_xy[3 * i + 1] for i in range(n)])
    W_local = np.array([shaft_system.shaft.W_at(x)   for x in solver.x_nodes])
    d_local = np.array([shaft_system.shaft.diameter_at(x) for x in solver.x_nodes])
    areas   = np.array([
        shaft_system.shaft.section_at(x)[0].area for x in solver.x_nodes
    ])

    return dict(
        x=np.array(solver.x_nodes),
        M_xz=M_xz, M_xy=M_xy, V_xz=V_xz, V_xy=V_xy,
        v_xz=v_xz, v_xy=v_xy, N_ax=N_ax, d_local=d_local,
        sigma_b=np.sqrt(M_xz**2 + M_xy**2) / W_local,
        sigma_ax=N_ax / areas,
        T=solver.T_total, tau=solver.tau_total,
    )


diagrams = {
    sh.name: recover_diagrams(solvers[sh.name], sh)
    for sh in (sys1, sys2, sys3)
}


# ===========================================================================
# PER-NODE TABLE
# ===========================================================================

def print_node_table(shaft_system, d: dict) -> None:
    n   = len(d["x"])
    sep = "=" * 148
    print(f"\n{sep}")
    print(f"  PER-NODE RESULTS -- {shaft_system.name}  ({n} nodes)")
    print(sep)
    hdr = (
        f"  {'node':>4} {'x[mm]':>9} {'d[mm]':>7} "
        f"{'v_xz[mm]':>11} {'v_xy[mm]':>11} "
        f"{'M_xz[N.mm]':>13} {'M_xy[N.mm]':>13} "
        f"{'V_xz[N]':>10} {'V_xy[N]':>10} "
        f"{'N[N]':>9} {'T[N.m]':>9} "
        f"{'sb[MPa]':>10} {'sax[MPa]':>10} {'tau[MPa]':>10}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i in range(n):
        print(
            f"  {i:>4d} {d['x'][i]:>9.2f} {d['d_local'][i]:>7.2f} "
            f"{d['v_xz'][i]:>11.5f} {d['v_xy'][i]:>11.5f} "
            f"{d['M_xz'][i]:>13.1f} {d['M_xy'][i]:>13.1f} "
            f"{d['V_xz'][i]:>10.1f} {d['V_xy'][i]:>10.1f} "
            f"{d['N_ax'][i]:>9.1f} {d['T'][i]:>9.2f} "
            f"{d['sigma_b'][i]:>10.2f} {d['sigma_ax'][i]:>10.2f} "
            f"{d['tau'][i]:>10.2f}"
        )
    print(sep)


for sh in (sys1, sys2, sys3):
    print_node_table(sh, diagrams[sh.name])


# ===========================================================================
# PLOTS
# ===========================================================================

def plot_shaft(name: str, d: dict, shaft_system) -> None:
    x   = d["x"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    fig.suptitle(f"Shaft analysis -- {name}", fontsize=13, fontweight="bold")

    draw_shaft_detail(axes[0], shaft_system, local=True)
    axes[0].set_xlabel("")

    axes[1].plot(x, d["M_xz"], label="M_xz", color=INK, linestyle="-",  linewidth=1.1)
    axes[1].plot(x, d["M_xy"], label="M_xy", color=INK, linestyle="--", linewidth=1.1)
    axes[1].axhline(0, color="0.6", linewidth=0.6)
    axes[1].set_ylabel("Moment\n[N.mm]")
    axes[1].legend(fontsize=8, loc="upper right", frameon=False)

    axes[2].plot(x, d["V_xz"], label="V_xz", color=INK, linestyle="-",  linewidth=1.1)
    axes[2].plot(x, d["V_xy"], label="V_xy", color=INK, linestyle="--", linewidth=1.1)
    axes[2].axhline(0, color="0.6", linewidth=0.6)
    axes[2].set_ylabel("Shear\n[N]")
    axes[2].legend(fontsize=8, loc="upper right", frameon=False)

    axes[3].plot(x, d["v_xz"] * 1000, label="v_xz (um)", color=INK, linestyle="-",  linewidth=1.1)
    axes[3].plot(x, d["v_xy"] * 1000, label="v_xy (um)", color=INK, linestyle="--", linewidth=1.1)
    axes[3].axhline(0, color="0.6", linewidth=0.6)
    axes[3].set_ylabel("Deflection\n[um]")
    axes[3].legend(fontsize=8, loc="upper right", frameon=False)

    ax4b = axes[4].twinx()
    l1,  = axes[4].plot(x, d["T"],   color=INK, linestyle="-",  linewidth=1.1, label="T(x)")
    l2,  = ax4b.plot(   x, d["tau"], color=INK, linestyle=":",  linewidth=1.1, label="tau(x)")
    axes[4].axhline(0, color="0.6", linewidth=0.6)
    axes[4].set_ylabel("Torque\n[N.m]")
    ax4b.set_ylabel("Shear stress\n[MPa]")
    axes[4].set_xlabel("x [mm]")
    axes[4].legend(handles=[l1, l2], fontsize=8, loc="upper right", frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5, color="0.7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])


for sh in (sys1, sys2, sys3):
    plot_shaft(sh.name, diagrams[sh.name], sh)

fig, ax = plt.subplots(figsize=(9, 4.5))
for sh, ls in zip((sys1, sys2, sys3), ("-", "--", ":")):
    d = diagrams[sh.name]
    ax.plot(d["x"], d["sigma_b"], label=f"{sh.name} sb",
            color=INK, linestyle=ls, linewidth=1.2)
ax.axhline(0, color="0.6", linewidth=0.6)
ax.set_xlabel("x [mm] (local, per-shaft)")
ax.set_ylabel("Bending stress at outer fibre [MPa]")
ax.set_title("Bending stress comparison -- all shafts")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, linewidth=0.5, color="0.7")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

plt.show()


# ===========================================================================
# PEAK VALUES SUMMARY
# ===========================================================================

print("\n" + "=" * 96)
print("  PEAK VALUES PER SHAFT")
print("=" * 96)
print(
    f"  {'shaft':<16}{'max|M|[N.mm]':>14}{'max|V|[N]':>12}"
    f"{'max defl[um]':>14}{'max T[N.m]':>12}"
    f"{'max tau[MPa]':>14}{'max sb[MPa]':>13}"
)
for sh in (sys1, sys2, sys3):
    d     = diagrams[sh.name]
    M_res = np.hypot(d["M_xz"], d["M_xy"])
    V_res = np.hypot(d["V_xz"], d["V_xy"])
    v_res = np.hypot(d["v_xz"], d["v_xy"]) * 1000
    print(
        f"  {sh.name:<16}{M_res.max():14.1f}{V_res.max():12.1f}"
        f"{v_res.max():14.3f}{np.abs(d['T']).max():12.2f}"
        f"{np.abs(d['tau']).max():14.2f}{d['sigma_b'].max():13.2f}"
    )
print("=" * 96)


if __name__ == "__main__":
    pass