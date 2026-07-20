"""
design_load_lifting_bsweep.py

Runs the load-lifting gearbox for 4 face-width values:
    b = 0 mm  (point load, equivalent to distribute=False)
    b = 7 mm
    b = 14 mm
    b = 20 mm  (full face width)

For each shaft, one figure per quantity (M, V, deflection, sigma_b, T, tau).
Each figure has two subplots side by side: XZ plane (left) and XY plane (right).
All 4 b-values are overlaid on the same subplot.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.loads import RadialLoad, TorqueLoad
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import GearElement, ShaftSystem
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalMeshLink, SpurHelicalGearSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.core.mechanical_system.Parallel_Axis_systems.schematic import draw_shaft_detail, INK

# ===========================================================================
# PARAMETERS
# ===========================================================================

B_VALUES   = [0, 1, 2, 5, 10, 15, 20]   # face widths to sweep [mm]; b=0 -> point load
COLORS     = ["#1a1a1a", "#2166ac", "#d62728", "#33a02c", "#ff7f00", "#984ea3", "#a65628"]
P_W        = 1000.0
RPM_IN     = 450.0
g          = 9.81
r_pulley   = 0.050
SHAFT_NAMES = ["shaft1(motor)", "shaft2", "shaft3(pulley)"]

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")


# ===========================================================================
# HELPERS
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


def N204(position, label, locating=False):
    return Bearing(bearing_type=BearingType.CYLINDRICAL_ROLLER,
                   designation="N204", b=14.0, d=20.0, D=47.0,
                   C=28_500.0, C0=19_000.0,
                   arrangement="locating" if locating else "floating",
                   contact_angle_deg=0.0, X=1.0, Y=0.0,
                   position=position, label=label)


def recover_diagrams(solver, shaft_system):
    from axisforge.mesh.oneD.shaft.Elements.Timoshenko_Selective_Integration.timoshenko import TimoshenkoBeam
    beam = TimoshenkoBeam()
    n    = len(solver.x_nodes)
    M_xz = np.zeros(n); M_xy = np.zeros(n)
    V_xz = np.zeros(n); V_xy = np.zeros(n)
    N_ax = np.zeros(n)
    done = set()
    for elem in solver.elements:
        dofs = [3*elem.idx_node_1,   3*elem.idx_node_1+1, 3*elem.idx_node_1+2,
                3*elem.idx_node_2,   3*elem.idx_node_2+1, 3*elem.idx_node_2+2]
        Ke   = beam.stiffness_element(elem)
        f_xz = Ke @ solver.d_total_xz[dofs]
        f_xy = Ke @ solver.d_total_xy[dofs]
        for local, node_id in ((0, elem.idx_node_1), (1, elem.idx_node_2)):
            if node_id in done:
                continue
            done.add(node_id)
            sl = slice(0, 3) if local == 0 else slice(3, 6)
            Naxz, Vxz, Mxz = f_xz[sl]
            Naxy, Vxy, Mxy = f_xy[sl]
            N_ax[node_id] = Naxz
            V_xz[node_id] = Vxz;  M_xz[node_id] = Mxz
            V_xy[node_id] = Vxy;  M_xy[node_id] = Mxy
    v_xz    = np.array([solver.d_total_xz[3*i+1] for i in range(n)])
    v_xy    = np.array([solver.d_total_xy[3*i+1] for i in range(n)])
    d_local = np.array([shaft_system.shaft.diameter_at(x) for x in solver.x_nodes])
    W_local = np.array([shaft_system.shaft.W_at(x)        for x in solver.x_nodes])
    sigma_b = np.sqrt(M_xz**2 + M_xy**2) / W_local
    sigma_ax = N_ax / np.array([shaft_system.shaft.section_at(x)[0].area
                                 for x in solver.x_nodes])
    return dict(x=np.array(solver.x_nodes),
                M_xz=M_xz, M_xy=M_xy, V_xz=V_xz, V_xy=V_xy,
                v_xz=v_xz, v_xy=v_xy, N_ax=N_ax, d_local=d_local,
                sigma_b=sigma_b, sigma_ax=sigma_ax,
                T=solver.T_total, tau=solver.tau_total)


def build_and_solve(b: float) -> tuple[dict, dict]:
    """Build, resolve and solve for a given face width b [mm]."""
    distribute = b > 0.0
    gear_kw    = dict(**GEAR_KW_BASE, b=b)

    z1 = SpurHelicalGear(z=20, position=55.0, label="z1", **gear_kw)
    z2 = SpurHelicalGear(z=60, position=95.0, label="z2", **gear_kw)
    z3 = SpurHelicalGear(z=20, position=55.0, label="z3", **gear_kw)
    z4 = SpurHelicalGear(z=60, position=55.0, label="z4", **gear_kw)

    stage1 = SpurHelicalGearMeshing(z1, z2, label="stage1")
    stage2 = SpurHelicalGearMeshing(z3, z4, label="stage2")

    s1 = make_stepped_shaft("shaft1", 120.0, 20.0, 25.0, 30.0, 30.0, 1.5)
    s2 = make_stepped_shaft("shaft2", 150.0, 20.0, 28.0, 35.0, 35.0, 2.0)
    s3 = make_stepped_shaft("shaft3", 150.0, 20.0, 25.0, 35.0, 35.0, 1.5)

    sys1 = ShaftSystem(s1, name="shaft1(motor)",  speed_rpm=450.0)
    sys2 = ShaftSystem(s2, name="shaft2",         speed_rpm=150.0)
    sys3 = ShaftSystem(s3, name="shaft3(pulley)", speed_rpm=50.0)

    sys1.shaft_origin_x = 0.0
    sys2.shaft_origin_x = sys1.shaft_origin_x + (z1.position - z2.position)
    sys3.shaft_origin_x = sys2.shaft_origin_x + (z3.position - z4.position)

    sys1.add_bearing(N204(20.0,  "brg1a", locating=True))
    sys1.add_bearing(N204(100.0, "brg1b"))
    sys2.add_bearing(N204(25.0,  "brg2a", locating=True))
    sys2.add_bearing(N204(125.0, "brg2b"))
    sys3.add_bearing(N204(25.0,  "brg3a", locating=True))
    sys3.add_bearing(N204(125.0, "brg3b"))

    ge_z1 = GearElement(z1, role="driver", rotation_dir=1, label="z1")
    ge_z2 = GearElement(z2, role="driven", label="z2")
    ge_z3 = GearElement(z3, role="driver", label="z3")
    ge_z4 = GearElement(z4, role="driven", label="z4")

    sys1.add_gear(ge_z1)
    sys2.add_gear(ge_z2); sys2.add_gear(ge_z3)
    sys3.add_gear(ge_z4)

    link1 = SpurHelicalMeshLink(sys1, ge_z1, sys2, ge_z2, stage1, phi_deg=270.0,
                                 distribute_loads=distribute, label="stage1")
    link2 = SpurHelicalMeshLink(sys2, ge_z3, sys3, ge_z4, stage2, phi_deg=270.0,
                                 distribute_loads=distribute, label="stage2")

    gearbox = SpurHelicalGearSystem([sys1, sys2, sys3], [link1, link2],
                                     label="load-lifting")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    T3     = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    dist_labels = {"*"} if distribute else None
    solvers = {}
    for sh in (sys1, sys2, sys3):
        s = SimpleFEMSolver(theory="timoshenko", distribute_gear_labels=dist_labels)
        s.solve(sh)
        solvers[sh.name] = s

    diagrams = {sh.name: recover_diagrams(solvers[sh.name], sh)
                for sh in (sys1, sys2, sys3)}
    shafts   = {sh.name: sh for sh in (sys1, sys2, sys3)}
    return diagrams, shafts


# ===========================================================================
# RUN SWEEP
# ===========================================================================

results = {}
ref_shafts = {}
for b in B_VALUES:
    print(f"  Solving b={b} mm ...")
    diag, shafts = build_and_solve(b)
    results[b]    = diag
    ref_shafts[b] = shafts

print("All cases complete.\n")


# ===========================================================================
# PLOTS
# ===========================================================================

QUANTITIES = [
    ("M",       "M_xz",   "M_xy",   "Moment [N·mm]",    None),
    ("V",       "V_xz",   "V_xy",   "Shear [N]",        None),
    ("v",       "v_xz",   "v_xy",   "Deflection [µm]",  1000.0),
    ("sigma_b", "sigma_b",None,      "σ_b [MPa]",        None),
    ("T",       "T",      None,      "Torque [N·m]",     None),
    ("tau",     "tau",    None,      "Shear stress τ [MPa]", None),
]
# (label, key_xz, key_xy, ylabel, scale)
# key_xy = None  ->  scalar quantity, only one subplot


# Split quantities into two groups per shaft:
#   Fig A — bending (M, V, v): paired XZ | XY — more space
#   Fig B — stress/torsion (sigma_b, T, tau): scalar, full-width
QUANTITIES_A = [q for q in QUANTITIES if q[2] is not None]   # has XY pair
QUANTITIES_B = [q for q in QUANTITIES if q[2] is None]       # scalar only


def _plot_group(shaft_name: str, qty_list: list, fig_label: str) -> None:
    import matplotlib.gridspec as gridspec

    nrows = len(qty_list)
    fig   = plt.figure(figsize=(16, 3.8 * nrows))
    fig.suptitle(f"{shaft_name} — {fig_label}", fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.45, wspace=0.3)

    for row, (label, key_xz, key_xy, ylabel, scale) in enumerate(qty_list):
        has_two = key_xy is not None
        if has_two:
            ax_xz = fig.add_subplot(gs[row, 0])
            ax_xy = fig.add_subplot(gs[row, 1])
            pair  = [(ax_xz, key_xz, "XZ"), (ax_xy, key_xy, "XY")]
        else:
            ax_sc = fig.add_subplot(gs[row, :])
            pair  = [(ax_sc, key_xz, "")]

        s = scale if scale else 1.0
        for ax, key, plane in pair:
            for b, color in zip(B_VALUES, COLORS):
                d = results[b][shaft_name]
                ax.plot(d["x"], d[key] * s, color=color, lw=1.2,
                        label=f"b={b} mm")
            title = f"{label} — {plane}" if plane else label
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.grid(alpha=0.2, lw=0.4, color="0.7")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, frameon=False, ncol=2)


for shaft_name in SHAFT_NAMES:
    _plot_group(shaft_name, QUANTITIES_A, "M  |  V  |  deflection")
    _plot_group(shaft_name, QUANTITIES_B, "σ_b  |  T  |  τ")


# ===========================================================================
# PEAK VALUES TABLE
# ===========================================================================

print("=" * 90)
print("  PEAK VALUES SWEEP — b = 0 / 7 / 14 / 20 mm")
print("=" * 90)
print(f"  {'shaft':<18}  {'qty':<10}  "
      + "  ".join(f"{'b='+str(b)+' mm':>12}" for b in B_VALUES))
print("  " + "-" * 86)

for shaft_name in SHAFT_NAMES:
    for qty, key in [("max|M|[N.mm]", lambda d: np.hypot(d["M_xz"], d["M_xy"]).max()),
                     ("max sb [MPa]", lambda d: d["sigma_b"].max()),
                     ("max v [um]",   lambda d: np.hypot(d["v_xz"], d["v_xy"]).max()*1000),
                     ("max|V| [N]",   lambda d: np.hypot(d["V_xz"], d["V_xy"]).max())]:
        vals = [key(results[b][shaft_name]) for b in B_VALUES]
        print(f"  {shaft_name:<18}  {qty:<10}  "
              + "  ".join(f"{v:>12.2f}" for v in vals))
    print()

print("=" * 90)

plt.show()