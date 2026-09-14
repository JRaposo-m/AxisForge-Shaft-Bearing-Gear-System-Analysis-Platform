"""
design_load_lifting_shafts.py

Extends design_load_lifting.py:
  1. Shafts are now STEPPED with real Shoulder fillets (not uniform d=20 mm).
  2. Torque equilibrium is closed explicitly on shaft1 (motor input) and
     shaft3 (pulley/load resistance) — see "TORQUE EQUILIBRIUM" section
     below for why this is required for a meaningful torsion diagram.
  3. Runs SimpleFEMSolver (bending XZ/XY + torsion) on all three shafts.
  4. Produces verification plots: bending moment, shear, deflection,
     torque and shear-stress diagrams, per shaft.
  5. Draws the assembly SCHEMATIC (system 1D + per-shaft 2D detail) right
     after resolve(), as a quick "is this wired up right" visual check
     before any solver runs — see axisforge/core/mechanical_system/
     Parallel_Axis_systems/schematic.py.

ASSUMPTIONS (flagged explicitly, not silently applied):
  A1. Bearing seats sit at the smaller step diameter (typical practice —
      bore fits N204, d=20 mm); the gear/pulley span is stepped UP for a
      stiffer, larger-diameter body. Fillet radii chosen conservatively
      small (1.5-2.0 mm) vs. step height so Shoulder.__post_init__ passes.
  A2. Motor coupling on shaft1 assumed at x=0 (shaft end nearest the
      source bearing). A TorqueLoad(-T_in, x=0, source="user") is added
      there so T(x) reads T_in from 0 to the gear and 0 beyond — the
      physically expected pattern for "motor -> gear -> nothing further".
      Without this, T(x) on the source shaft would read backwards (0
      before the gear, T_in after) because GearSystem only places a
      TorqueLoad AT the gear mesh point, not at the shaft's power-input
      end.
  A3. Pulley resistance on shaft3: the useful load doesn't just bend the
      shaft (RadialLoad from rope tension) — it also RESISTS rotation
      with a torque = F_rope * r_pulley = T_out(stage2). This was never
      modeled as a TorqueLoad in the original script. Added explicitly
      so shaft3's torque diagram closes to equilibrium (net = 0).
  A4. Bearing width added (N204 catalog face width ~14 mm) so Mesh1D
      places nodes at both bearing faces, not just the centreline.

SOLVER NOTATION:
  SimpleFEMSolver.solve() no longer returns a result object — it
  publishes every intermediate quantity as public attributes on the
  solver instance (x_nodes, elements, d_total_xz, T_total, tau_total,
  ...). Because a second solve() call on the same instance overwrites
  those attributes, one SimpleFEMSolver instance is created PER SHAFT
  below and kept alive in `solvers[shaft.name]` for post-processing.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder

from axisforge.core.loads import RadialLoad, TorqueLoad
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import RigidBearingFEMSolver

# --- schematic (visual sanity check, no solver dependency) -----------------
from axisforge.core.mechanical_system.parallel_axis.schematic import (
    draw_gear_system, draw_shaft_detail, INK,
)


# ===========================================================================
# 1. GEARS — unchanged from design_load_lifting.py
# ===========================================================================

GEAR_KW = dict(mn=2.0, x=0.0, b=20.0, alpha_n_deg=20.0, Ra=0.8,
               material_id="42CrMo4")

z1 = SpurHelicalGear(z=20, position=55.0, label="z1", **GEAR_KW)
z2 = SpurHelicalGear(z=60, position=95.0, label="z2", **GEAR_KW)
z3 = SpurHelicalGear(z=20, position=55.0, label="z3", **GEAR_KW)
z4 = SpurHelicalGear(z=60, position=55.0, label="z4", **GEAR_KW)

stage1 = SpurHelicalGearMeshing(z1, z2, label="stage1 (z1->z2)")
stage2 = SpurHelicalGearMeshing(z3, z4, label="stage2 (z3->z4)")


# ===========================================================================
# 2. STEPPED SHAFTS with real shoulders  (assumption A1)
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


shaft1_geom = make_stepped_shaft("shaft1", total_length=120.0,
                                 d_seat=20.0, d_body=25.0,
                                 l_seat_a=30.0, l_seat_b=30.0, fillet_r=1.5)

shaft2_geom = make_stepped_shaft("shaft2", total_length=150.0,
                                 d_seat=20.0, d_body=28.0,
                                 l_seat_a=35.0, l_seat_b=35.0, fillet_r=2.0)

shaft3_geom = make_stepped_shaft("shaft3", total_length=150.0,
                                 d_seat=20.0, d_body=25.0,
                                 l_seat_a=35.0, l_seat_b=35.0, fillet_r=1.5)

for sh in (shaft1_geom, shaft2_geom, shaft3_geom):
    sh_errors = sh.validate()
    assert not sh_errors, f"{sh.label} geometry invalid: {sh_errors}"


# ===========================================================================
# 3. BEARINGS — N204, width=14 mm added (assumption A4)
# ===========================================================================

def N204(position: float, label: str, locating: bool = False) -> Bearing:
    """
    N204 cylindrical roller bearing — SKF catalog dims (approx):
    d=20mm, D=47mm, b=14mm.

    ASSUMPTION (flagged): N-type cylindrical roller bearings are pure
    radial bearings (no seating shoulder for axial thrust) — 'locating'
    here is a numerical/location placeholder, not a thrust-carrying claim.
    Something must fix the shaft's axial DOF even with Fa=0 (spur gears),
    same rationale as before — just renamed 'fixed'->'locating' to match
    the current Bearing.arrangement convention.
    """
    return Bearing(bearing_type=BearingType.CYLINDRICAL_ROLLER,
                   designation="N204",
                   b=14.0, d=20.0, D=47.0,
                   C=28_500.0, C0=19_000.0,
                   arrangement="locating" if locating else "floating",
                   contact_angle_deg=0.0,
                   X=1.0, Y=0.0,
                   position=position, label=label)


# ===========================================================================
# 4. ASSEMBLE
# ===========================================================================

sys1 = ShaftSystem(shaft1_geom, name="shaft1(motor)", speed_rpm=450.0)
sys2 = ShaftSystem(shaft2_geom, name="shaft2", speed_rpm=150.0)
sys3 = ShaftSystem(shaft3_geom, name="shaft3(pulley)", speed_rpm=50.0)

sys1.shaft_origin_x = 0.0
sys2.shaft_origin_x = sys1.shaft_origin_x + (z1.position - z2.position)
sys3.shaft_origin_x = sys2.shaft_origin_x + (z3.position - z4.position)

sys1.add_bearing(N204(20.0,  "brg1a", locating=True)); sys1.add_bearing(N204(100.0, "brg1b"))
sys2.add_bearing(N204(25.0,  "brg2a", locating=True)); sys2.add_bearing(N204(125.0, "brg2b"))
sys3.add_bearing(N204(25.0,  "brg3a", locating=True)); sys3.add_bearing(N204(125.0, "brg3b"))

ge_z1 = GearElement(z1, role="driver", rotation_dir=1, label="z1")
ge_z2 = GearElement(z2, role="driven", label="z2")
ge_z3 = GearElement(z3, role="driver", label="z3")
ge_z4 = GearElement(z4, role="driven", label="z4")

sys1.add_gear(ge_z1)
sys2.add_gear(ge_z2); sys2.add_gear(ge_z3)
sys3.add_gear(ge_z4)

link1 = SpurHelicalMeshLink(sys1, ge_z1, sys2, ge_z2, stage1, phi_deg=270.0, label="stage1")
link2 = SpurHelicalMeshLink(sys2, ge_z3, sys3, ge_z4, stage2, phi_deg=270.0, label="stage2")

gearbox = SpurHelicalGearSystem([sys1, sys2, sys3], [link1, link2], label="load-lifting")


def dump_loads(shaft: ShaftSystem) -> None:
    print(f"\n  loads on {shaft.name}:")
    for ld in shaft.loads:
        kind = type(ld).__name__
        if hasattr(ld, "theta_deg"):
            print(f"    {kind:<11} @x={ld.position:6.1f}mm  |F|={ld.magnitude:8.1f}N  "
                  f"θ={ld.theta_deg:6.1f}°  [{ld.source}] {ld.label}")
        else:
            print(f"    {kind:<11} @x={ld.position:6.1f}mm  val={ld.magnitude:11.1f}  "
                  f"[{ld.source}] {ld.label}")


# ===========================================================================
# 5. RESOLVE
# ===========================================================================

print("=" * 74)
print("  AxisForge — designing the Figure-1 load-lifting system in pure Python")
print("=" * 74)

for st in (stage1, stage2):
    print(st.summary())

P_W, RPM_IN, g = 1000.0, 450.0, 9.81

topo = gearbox._topology_errors()
print(f"\n  topology check : {'OK' if not topo else topo}")
assert not topo, f"topology errors: {topo}"

gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))


# ===========================================================================
# 5b. SCHEMATIC — quick "is this wired up right" visual check, BEFORE any
# solver runs. Uses only shaft_origin_x / shaft_position / bearings / gears
# / links, already available right after resolve() above.
#
# axis="z": stage1/stage2 both use phi_deg=270, which offsets shaft_position
# along z, not y (see schematic.py module docstring) — with axis="y" every
# shaft would collapse onto the same row.
# ===========================================================================

fig_schem, ax_schem = plt.subplots(figsize=(12, 5))
draw_gear_system(ax_schem, gearbox, axis="z")
fig_schem.suptitle("Gearbox schematic — system (1D)", fontsize=12, fontweight="bold")
fig_schem.tight_layout()

for sh in (sys1, sys2, sys3):
    fig_d, ax_d = plt.subplots(figsize=(9, 2.6))
    draw_shaft_detail(ax_d, sh)
    fig_d.suptitle(f"{sh.name} — detail (2D)", fontsize=11, fontweight="bold")
    fig_d.tight_layout()


T3 = gearbox._resolved[id(sys3)]["T_out_Nm"]
r_pulley = 0.050
F_rope = T3 / r_pulley
sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

print("\n" + "-" * 74)
print("  RESOLVED TRANSMISSION")
print("-" * 74)
print(gearbox.summary())

print("\n  shaft              speed[rpm]  T_out[N·m]  rot   axis(y,z)[mm]   εα")
for sh, mesh_in in ((sys1, None), (sys2, stage1), (sys3, stage2)):
    r = gearbox._resolved[id(sh)]
    y, z = r["shaft_position"]
    eps = f"{mesh_in.epslon_alpha:.3f}" if mesh_in else "  —  "
    print(f"  {sh.name:<17} {sh.speed_rpm:>7.1f}   {r['T_out_Nm']:>9.3f}   "
          f"{int(r['rotation_dir']):>+3d}   ({y:6.1f},{z:7.1f})   {eps}")

print("\n" + "-" * 74)
print("  MESH FORCES PER STAGE (Ft, Fr from injected loads; Fn from mesh)")
print("-" * 74)
for st, drv_shaft, drv_label in ((stage1, sys1, "z1"), (stage2, sys2, "z3")):
    fr = next(l.magnitude for l in drv_shaft.radial_loads
              if l.source == "gear_mesh" and l.label == f"{drv_label}:Fr")
    ft = next(l.magnitude for l in drv_shaft.radial_loads
              if l.source == "gear_mesh" and l.label == f"{drv_label}:Ft")
    Fn = ft / math.cos(st.alphatw)
    print(f"  {st.label:<20} Ft={ft:8.1f} N   Fr={fr:8.1f} N   Fn={Fn:8.1f} N")

print("\n" + "-" * 74)
print("  PER-SHAFT LOAD DUMP")
print("-" * 74)
for sh in (sys1, sys2, sys3):
    dump_loads(sh)

print("\n" + "-" * 74)
print("  BEARING QUICK-LOOK (resultant radial gear force per shaft vs C)")
print("-" * 74)
for sh in (sys2, sys3):
    Fy = sum(l.Fy for l in sh.radial_loads)
    Fz = sum(l.Fz for l in sh.radial_loads)
    R = math.hypot(Fy, Fz)
    per_brg = R / 2.0
    C = sh.bearings[0].C
    print(f"  {sh.name:<16} ΣF_radial={R:8.1f} N  ~{per_brg:7.1f} N/bearing  "
          f"(C={C/1000:.1f} kN -> C/P≈{C/max(per_brg,1e-9):5.1f})")

print("\n" + "-" * 74)
print("  OUTPUT / LIFTING")
print("-" * 74)
hoist = sys3.speed_rpm / 60.0 * math.pi * (2 * r_pulley)
print(f"  output torque (shaft3) : {T3:.2f} N·m  @ {sys3.speed_rpm:.0f} rpm")
print(f"  wire-rope tension      : {F_rope:.1f} N")
print(f"  liftable load (ideal)  : {F_rope/g:.1f} kg   (no losses, static)")
print(f"  hoist speed            : {hoist*1000:.1f} mm/s")


# ===========================================================================
# 6. TORQUE EQUILIBRIUM — close the loop (assumptions A2, A3)
# ===========================================================================

T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]   # torque used at z1 as driver
sys1.add_load(TorqueLoad(0.0, -T_z1_in, source="user", label="motor-input"))

# shaft3: rope tension resists rotation with torque = F_rope * r_pulley = T3
sys3.add_load(TorqueLoad(95.0, T3, source="user", label="pulley-load-resistance"))


# ===========================================================================
# 7. SOLVE — bending/axial FEM + torsion, per shaft
#
# RigidBearingFEMSolver.solve() publishes results as public attributes on the
# instance (no return value) — one solver instance is kept PER SHAFT so
# results from all three don't overwrite each other.
# ===========================================================================

print("=" * 74)
print("  TORQUE EQUILIBRIUM CHECK (per shaft, must be ~0 N*m)")
print("=" * 74)
check_solver = RigidBearingFEMSolver()
for sh in (sys1, sys2, sys3):
    errs = check_solver.validate_torsion_equilibrium(sh)
    status = "OK" if not errs else " / ".join(errs)
    print(f"  {sh.name:<16} {status}")

solvers: dict[str, RigidBearingFEMSolver] = {}
for sh in (sys1, sys2, sys3):
    s = RigidBearingFEMSolver()
    s.solve(sh)
    solvers[sh.name] = s

print("\n" + "=" * 74)
print("  FULL GearSystem.validate()")
print("=" * 74)
errs = gearbox.validate()
print("  OK — no errors" if not errs else "\n".join(f"  - {e}" for e in errs))


# ===========================================================================
# 8. POST-PROCESS: recover M(x), V(x) from nodal displacements per element
# ===========================================================================

def recover_diagrams(solver: RigidBearingFEMSolver, shaft_system: ShaftSystem):
    """
    Per-node bending moment (XZ, XY), shear (XZ, XY), axial force, and
    deflection (v_xz, v_xy) — recovered from element stiffness x nodal
    displacement, same pattern as the legacy StressSolver.

    `solver` is a RigidBearingFEMSolver instance already solve()'d for
    `shaft_system` — reads its public attributes (x_nodes, elements,
    d_total_xz, d_total_xy, T_total, tau_total).
    """
    from axisforge.mesh.shaft.element_type.timoshenko.timoshenko import TimoshenkoBeam
    beam = TimoshenkoBeam()

    n = len(solver.x_nodes)
    M_xz = np.zeros(n); M_xy = np.zeros(n)
    V_xz = np.zeros(n); V_xy = np.zeros(n)
    N_ax = np.zeros(n)
    done = set()

    for elem in solver.elements:
        dofs = [3*elem.idx_node_1, 3*elem.idx_node_1+1, 3*elem.idx_node_1+2,
                 3*elem.idx_node_2, 3*elem.idx_node_2+1, 3*elem.idx_node_2+2]
        Ke = beam.stiffness_element(elem)
        f_xz = Ke @ solver.d_total_xz[dofs]
        f_xy = Ke @ solver.d_total_xy[dofs]

        for local, node_id in ((0, elem.idx_node_1), (1, elem.idx_node_2)):
            if node_id in done:
                continue
            done.add(node_id)
            sl = slice(0, 3) if local == 0 else slice(3, 6)
            Naxz, Vxz, Mxz = f_xz[sl]
            Naxy, Vxy, Mxy = f_xy[sl]
            N_ax[node_id] = Naxz  # axial DOF only carried on XZ solve (see solver docstring)
            V_xz[node_id] = Vxz
            M_xz[node_id] = Mxz
            V_xy[node_id] = Vxy
            M_xy[node_id] = Mxy

    v_xz = np.array([solver.d_total_xz[3*i + 1] for i in range(n)])
    v_xy = np.array([solver.d_total_xy[3*i + 1] for i in range(n)])

    # bending stress at outer fibre, using local section diameter
    d_local = np.array([shaft_system.shaft.diameter_at(x) for x in solver.x_nodes])
    W_local = np.array([shaft_system.shaft.W_at(x) for x in solver.x_nodes])
    # M_xz/M_xy come out of the FEM already in N*mm (K built with E in
    # N/mm^2, lengths in mm) — NOT N*m, unlike TorqueLoad. No *1000 here.
    sigma_b = np.sqrt(M_xz**2 + M_xy**2) / W_local   # N*mm / mm^3 = MPa
    sigma_ax = N_ax / np.array([shaft_system.shaft.section_at(x)[0].area for x in solver.x_nodes])

    return dict(x=np.array(solver.x_nodes), M_xz=M_xz, M_xy=M_xy, V_xz=V_xz, V_xy=V_xy,
                v_xz=v_xz, v_xy=v_xy, N_ax=N_ax, d_local=d_local,
                sigma_b=sigma_b, sigma_ax=sigma_ax,
                T=solver.T_total, tau=solver.tau_total)


diagrams = {sh.name: recover_diagrams(solvers[sh.name], sh)
            for sh in (sys1, sys2, sys3)}


# ===========================================================================
# 8b. PER-NODE TABLE — every mesh node, every quantity
# ===========================================================================

def print_node_table(shaft_system, d: dict) -> None:
    n = len(d["x"])
    print("\n" + "=" * 148)
    print(f"  PER-NODE RESULTS — {shaft_system.name}  ({n} nodes)")
    print("=" * 148)
    header = (f"  {'node':>4} {'x [mm]':>9} {'d [mm]':>7} "
              f"{'v_xz [mm]':>11} {'v_xy [mm]':>11} "
              f"{'M_xz [N·mm]':>13} {'M_xy [N·mm]':>13} "
              f"{'V_xz [N]':>10} {'V_xy [N]':>10} "
              f"{'N [N]':>9} {'T [N·m]':>9} "
              f"{'sigma_b [MPa]':>13} {'sigma_ax [MPa]':>14} {'tau [MPa]':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(n):
        print(f"  {i:>4d} {d['x'][i]:>9.2f} {d['d_local'][i]:>7.2f} "
              f"{d['v_xz'][i]:>11.5f} {d['v_xy'][i]:>11.5f} "
              f"{d['M_xz'][i]:>13.1f} {d['M_xy'][i]:>13.1f} "
              f"{d['V_xz'][i]:>10.1f} {d['V_xy'][i]:>10.1f} "
              f"{d['N_ax'][i]:>9.1f} {d['T'][i]:>9.2f} "
              f"{d['sigma_b'][i]:>13.2f} {d['sigma_ax'][i]:>14.2f} {d['tau'][i]:>10.2f}")
    print("=" * 148)


for sh in (sys1, sys2, sys3):
    print_node_table(sh, diagrams[sh.name])


# ===========================================================================
# 9. PLOTS
# ===========================================================================

def plot_shaft(name, d, shaft_system):
    x = d["x"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    fig.suptitle(f"Shaft analysis — {name}", fontsize=13, fontweight="bold")

    # --- shaft profile: reuse the real schematic (shoulders, bearings, gears,
    # true diameter) instead of a hand-rolled fill_between envelope.
    # local=True aligns it to the same 0..total_length frame as x below
    # (shaft2/shaft3 have shaft_origin_x != 0 — without local=True the
    # profile would sit shifted relative to the moment/shear/deflection
    # panels underneath, since they share this axis).
    ax0 = axes[0]
    draw_shaft_detail(ax0, shaft_system, local=True)
    ax0.set_title("")   # redundant with the figure suptitle above
    ax0.set_xlabel("")  # only the bottom panel (ax4) labels the shared x-axis

    ax1 = axes[1]
    ax1.plot(x, d["M_xz"], label="M_xz", color=INK, linestyle="-", linewidth=1.1)
    ax1.plot(x, d["M_xy"], label="M_xy", color=INK, linestyle="--", linewidth=1.1)
    ax1.axhline(0, color="0.6", linewidth=0.6)
    ax1.set_ylabel("Moment\n[N·mm]")
    ax1.legend(fontsize=8, loc="upper right", frameon=False)

    ax2 = axes[2]
    ax2.plot(x, d["V_xz"], label="V_xz", color=INK, linestyle="-", linewidth=1.1)
    ax2.plot(x, d["V_xy"], label="V_xy", color=INK, linestyle="--", linewidth=1.1)
    ax2.axhline(0, color="0.6", linewidth=0.6)
    ax2.set_ylabel("Shear\n[N]")
    ax2.legend(fontsize=8, loc="upper right", frameon=False)

    ax3 = axes[3]
    ax3.plot(x, d["v_xz"]*1000, label="v_xz (µm)", color=INK, linestyle="-", linewidth=1.1)
    ax3.plot(x, d["v_xy"]*1000, label="v_xy (µm)", color=INK, linestyle="--", linewidth=1.1)
    ax3.axhline(0, color="0.6", linewidth=0.6)
    ax3.set_ylabel("Deflection\n[µm]")
    ax3.legend(fontsize=8, loc="upper right", frameon=False)

    ax4 = axes[4]
    ax4b = ax4.twinx()
    l1, = ax4.plot(x, d["T"], color=INK, linestyle="-", linewidth=1.1, label="T(x)")
    l2, = ax4b.plot(x, d["tau"], color=INK, linestyle=":", linewidth=1.1, label="tau(x)")
    ax4.axhline(0, color="0.6", linewidth=0.6)
    ax4.set_ylabel("Torque\n[N·m]")
    ax4b.set_ylabel("Shear stress\n[MPa]")
    ax4.set_xlabel("x [mm]")
    ax4.legend(handles=[l1, l2], fontsize=8, loc="upper right", frameon=False)

    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5, color="0.7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


for sh in (sys1, sys2, sys3):
    plot_shaft(sh.name, diagrams[sh.name], sh)

# --- combined bending-stress summary across all 3 shafts ---
fig, ax = plt.subplots(figsize=(9, 4.5))
for sh, ls in zip((sys1, sys2, sys3), ("-", "--", ":")):
    d = diagrams[sh.name]
    ax.plot(d["x"], d["sigma_b"], label=f"{sh.name} σ_b", color=INK, linestyle=ls, linewidth=1.2)
ax.axhline(0, color="0.6", linewidth=0.6)
ax.set_xlabel("x [mm] (local, per-shaft)")
ax.set_ylabel("Bending stress at outer fibre [MPa]")
ax.set_title("Bending stress comparison — all shafts")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, linewidth=0.5, color="0.7")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

# show every figure created above, blocking until closed — nothing written to disk
plt.show()


# ===========================================================================
# 10. NUMERIC SUMMARY TABLE
# ===========================================================================

print("\n" + "=" * 96)
print("  PEAK VALUES PER SHAFT")
print("=" * 96)
print(f"  {'shaft':<16}{'max|M| [N·mm]':>16}{'max|V| [N]':>14}"
      f"{'max defl [µm]':>16}{'max T [N·m]':>14}{'max tau [MPa]':>15}{'max σ_b [MPa]':>15}")
for sh in (sys1, sys2, sys3):
    d = diagrams[sh.name]
    M_res = np.hypot(d["M_xz"], d["M_xy"])
    V_res = np.hypot(d["V_xz"], d["V_xy"])
    v_res = np.hypot(d["v_xz"], d["v_xy"]) * 1000
    print(f"  {sh.name:<16}{M_res.max():16.1f}{V_res.max():14.1f}"
          f"{v_res.max():16.3f}{np.abs(d['T']).max():14.2f}"
          f"{np.abs(d['tau']).max():15.2f}{d['sigma_b'].max():15.2f}")
print("=" * 96)