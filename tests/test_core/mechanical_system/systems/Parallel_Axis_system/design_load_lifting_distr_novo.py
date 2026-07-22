"""
design_load_lifting_bsweep.py

Runs the load-lifting gearbox for several face-width values, now using
ShaftResultsReader (static_analysis.py) instead of an inline recover_diagrams.
Includes an exotic sinusoidal distributed load on shaft3 for validation.
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
from axisforge.core.loads import RadialLoad, TorqueLoad, DistributedRadialLoad
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import GearElement, ShaftSystem
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalMeshLink, SpurHelicalGearSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader
from axisforge.core.mechanical_system.Parallel_Axis_systems.schematic import draw_shaft_detail, INK

# ===========================================================================
# PARAMETERS
# ===========================================================================

B_VALUES   = [0, 1, 2, 5, 10, 15, 20]
COLORS     = ["#1a1a1a", "#2166ac", "#d62728", "#33a02c", "#ff7f00", "#984ea3", "#a65628"]
P_W        = 1000.0
RPM_IN     = 450.0
g          = 9.81
r_pulley   = 0.050
SHAFT_NAMES = ["shaft1(motor)", "shaft2", "shaft3(pulley)"]

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

# exotic distributed load parameters (shaft3 only)
X_LO_EXOTIC  = 35.0    # [mm]
X_HI_EXOTIC  = 90.0    # [mm]
Q_AMP_EXOTIC = 50.0    # [N/mm]


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


def recover_diagrams(solver, shaft_system) -> dict:
    results = ShaftResultsReader(solver, shaft_system).read()
    return dict(
        x        = results.x,
        M_xz     = results.M_xz,   M_xy     = results.M_xy,
        V_xz     = results.V_xz,   V_xy     = results.V_xy,
        v_xz     = results.v_xz,   v_xy     = results.v_xy,
        sigma_b  = results.sigma_b,
        T        = results.T,
        tau      = results.tau,
    )


def _exotic_load() -> DistributedRadialLoad:
    """
    Sinusoidal intensity, variable theta — exotic test case on shaft3.
        q(x)     = Q_AMP * sin(pi*(x - x_lo) / (x_hi - x_lo))   [N/mm]
        theta(x) = 90 + 45*sin(2*pi*(x - x_lo) / (x_hi - x_lo)) [deg]
    """
    span = X_HI_EXOTIC - X_LO_EXOTIC
    q_fn     = lambda x: Q_AMP_EXOTIC * np.sin(np.pi * (x - X_LO_EXOTIC) / span)
    theta_fn = lambda x: 90.0 + 45.0 * np.sin(2 * np.pi * (x - X_LO_EXOTIC) / span)
    return DistributedRadialLoad(
        x_lo      = X_LO_EXOTIC,
        x_hi      = X_HI_EXOTIC,
        magnitude = q_fn,
        theta_deg = theta_fn,
        label     = "exotic-sinusoidal",
        source    = "user",
    )


def build_and_solve(b: float) -> tuple[dict, dict]:
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

    T3      = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope  = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    # --- exotic distributed load on shaft3 ---
    sys3.add_load(_exotic_load())

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

results    = {}
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
    ("M",       "M_xz",    "M_xy",  "Moment [N·mm]",       None),
    ("V",       "V_xz",    "V_xy",  "Shear [N]",           None),
    ("v",       "v_xz",    "v_xy",  "Deflection [µm]",     None),
    ("sigma_b", "sigma_b", None,    "σ_b [MPa]",           None),
    ("T",       "T",       None,    "Torque [N·m]",        None),
    ("tau",     "tau",     None,    "Shear stress τ [MPa]",None),
]
QUANTITIES_A = [q for q in QUANTITIES if q[2] is not None]
QUANTITIES_B = [q for q in QUANTITIES if q[2] is None]


def _plot_group(shaft_name: str, qty_list: list, fig_label: str) -> None:
    import matplotlib.gridspec as gridspec

    nrows = len(qty_list)
    fig   = plt.figure(figsize=(16, 3.8 * nrows))
    fig.suptitle(f"{shaft_name} — {fig_label}", fontsize=13, fontweight="bold")
    gs    = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.45, wspace=0.3)

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
                ax.plot(d["x"], d[key] * s, color=color, lw=1.2, label=f"b={b} mm")
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
print("  PEAK VALUES SWEEP  (with exotic sinusoidal load on shaft3)")
print("=" * 90)
print(f"  {'shaft':<18}  {'qty':<14}  "
      + "  ".join(f"{'b='+str(b)+' mm':>12}" for b in B_VALUES))
print("  " + "-" * 90)

for shaft_name in SHAFT_NAMES:
    for qty, fn in [
        ("max|M| [N.mm]", lambda d: np.hypot(d["M_xz"], d["M_xy"]).max()),
        ("max sb [MPa]",  lambda d: d["sigma_b"].max()),
        ("max v [µm]",    lambda d: np.hypot(d["v_xz"], d["v_xy"]).max()),
        ("max|V| [N]",    lambda d: np.hypot(d["V_xz"], d["V_xy"]).max()),
    ]:
        vals = [fn(results[b][shaft_name]) for b in B_VALUES]
        print(f"  {shaft_name:<18}  {qty:<14}  "
              + "  ".join(f"{v:>12.2f}" for v in vals))
    print()

# --- exotic load detail: shaft3 node-by-node ---
print("─" * 90)
print("  EXOTIC LOAD DETAIL — shaft3(pulley), per b value")
print("─" * 90)
print(f"  {'b [mm]':>8}  {'x_M_max [mm]':>14}  {'M_xz_max [N.mm]':>18}"
      f"  {'M_xy_max [N.mm]':>18}  {'sigma_b_max [MPa]':>18}")
print("  " + "-" * 82)

for b in B_VALUES:
    d     = results[b]["shaft3(pulley)"]
    x     = d["x"]
    M_res = np.hypot(d["M_xz"], d["M_xy"])
    idx   = int(np.argmax(M_res))
    print(f"  {b:>8}  {x[idx]:>14.2f}  {d['M_xz'][idx]:>18.2f}"
          f"  {d['M_xy'][idx]:>18.2f}  {d['sigma_b'][idx]:>18.3f}")

print("=" * 90)

# --- node-by-node detail per case ---
print("\n")
for b in B_VALUES:
    d = results[b]["shaft3(pulley)"]
    x = d["x"]
    print(f"  b={b} mm — shaft3(pulley) — node by node")
    print(f"  {'x [mm]':>10}  {'M_xz [N.mm]':>16}  {'M_xy [N.mm]':>16}  {'V_xz [N]':>12}  {'V_xy [N]':>12}  {'sigma_b [MPa]':>15}")
    print("  " + "-" * 90)
    for i in range(len(x)):
        print(f"  {x[i]:>10.2f}  {d['M_xz'][i]:>16.2f}  {d['M_xy'][i]:>16.2f}"
              f"  {d['V_xz'][i]:>12.2f}  {d['V_xy'][i]:>12.2f}  {d['sigma_b'][i]:>15.3f}")
    print()

plt.show()