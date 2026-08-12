"""
test_fanout_branched_system_asymmetric.py

Exploratory analysis — fan-out (branched) parallel-axis transmission.
VARIANT: asymmetric torque_split (0.7 / 0.3) instead of 0.5 / 0.5.

Run directly:
    python test_fanout_branched_system_asymmetric.py

Topology under test  (CASE 1b — single shared pinion, ASYMMETRIC split
                       AND asymmetric torque_split)
---------------------------------------------------------------------
    sys1 (input)  z1(20)  --link_a (split=0.7)-->  z2a(60)  sys2a   u_a=3
                    |
                    +------link_b (split=0.3)-->  z2b(40)  sys2b   u_b=2

Everything from the original CASE 1 script is unchanged EXCEPT:
  - TORQUE_SPLIT_A / TORQUE_SPLIT_B replace the single TORQUE_SPLIT
    (must still sum to 1.0 -- validated by SpurHelicalGearSystem itself).
  - the "expected value" prints are now PASS/FAIL assertions against a
    numeric tolerance, so you get an unambiguous answer instead of having
    to eyeball two numbers.

What this variant additionally verifies beyond CASE 1
-------------------------------------------------------
  - torque_split is actually READ PER-LINK (not silently overwritten or
    averaged): branch A must carry noticeably more torque/force than
    branch B, in a ratio consistent with 0.7:0.3, not 0.5:0.5.
  - the net radial force on z1 changes (in principle) when the split
    changes, since Ft_a and Ft_b no longer come from equal input torques
    even before accounting for the different pitch radii.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver


# ===========================================================================
# PARAMETERS
# ===========================================================================

P_W            = 1_000.0      # input power [W]
RPM_IN         = 450.0        # input speed [rpm]
TORQUE_SPLIT_A = 0.7          # branch A (z1->z2a, u=3) takes 70%
TORQUE_SPLIT_B = 0.3          # branch B (z1->z2b, u=2) takes 30%
TOL_REL        = 1e-6         # relative tolerance for PASS/FAIL checks
GEAR_KW        = dict(mn=2.0, x=0.0, b=20.0, alpha_n_deg=20.0, Ra=0.8,
                      material_id="42CrMo4")


# ===========================================================================
# CONSOLE FORMATTING
# ===========================================================================

W = 78

def hrule(char: str = "=") -> str:
    return char * W

def banner(text: str) -> None:
    print(f"\n{hrule()}")
    print(f"  {text}")
    print(hrule())

def section(text: str) -> None:
    print(f"\n{'-' * W}")
    print(f"  {text}")
    print('-' * W)

def check(label: str, actual: float, expected: float, tol_rel: float = TOL_REL) -> bool:
    """Print a PASS/FAIL line comparing actual vs expected, return bool."""
    denom = abs(expected) if abs(expected) > 1e-12 else 1.0
    rel_err = abs(actual - expected) / denom
    ok = rel_err <= tol_rel
    status = "PASS" if ok else "FAIL"
    print(f"    [{status}] {label:<40} actual={actual:12.4f}  "
          f"expected={expected:12.4f}  rel_err={rel_err:.2e}")
    return ok


# ===========================================================================
# BUILD HELPERS  (unchanged from CASE 1)
# ===========================================================================

def make_stepped_shaft(name, total_length, d_seat, d_body,
                       l_seat_a, l_seat_b, fillet_r,
                       material_id="AISI_1045"):
    l_body = total_length - l_seat_a - l_seat_b
    assert l_body > 0, f"{name}: body length must be > 0, got {l_body}"
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
    return Bearing(
        bearing_type=BearingType.CYLINDRICAL_ROLLER,
        designation="N204",
        b=14.0, d=20.0, D=47.0,
        C=28_500.0, C0=19_000.0,
        arrangement="locating" if locating else "floating",
        contact_angle_deg=0.0,
        X=1.0, Y=0.0,
        position=position, label=label,
    )


# ===========================================================================
# ASSEMBLE BRANCHED SYSTEM  (only the two link_* calls changed)
# ===========================================================================

def build_branched_system():
    z1  = SpurHelicalGear(z=20, position=55.0, label="z1",  **GEAR_KW)
    z2a = SpurHelicalGear(z=60, position=55.0, label="z2a", **GEAR_KW)
    z2b = SpurHelicalGear(z=40, position=55.0, label="z2b", **GEAR_KW)

    stage_a = SpurHelicalGearMeshing(z1, z2a, label="stage_a (z1->z2a)")
    stage_b = SpurHelicalGearMeshing(z1, z2b, label="stage_b (z1->z2b)")

    s1  = make_stepped_shaft("shaft1",  120.0, 20.0, 25.0, 30.0, 30.0, 1.5)
    s2a = make_stepped_shaft("shaft2a", 150.0, 20.0, 28.0, 35.0, 35.0, 2.0)
    s2b = make_stepped_shaft("shaft2b", 150.0, 20.0, 28.0, 35.0, 35.0, 2.0)

    sys1  = ShaftSystem(s1,  name="shaft1(input)", speed_rpm=450.0)
    sys2a = ShaftSystem(s2a, name="shaft2a",       speed_rpm=150.0)   # u_a=3 -> 450/3
    sys2b = ShaftSystem(s2b, name="shaft2b",       speed_rpm=225.0)   # u_b=2 -> 450/2

    sys1.shaft_origin_x  = 0.0
    sys2a.shaft_origin_x = sys1.shaft_origin_x + (z1.position - z2a.position)
    sys2b.shaft_origin_x = sys1.shaft_origin_x + (z1.position - z2b.position)

    sys1.add_bearing(N204(20.0,  "brg1a", locating=True))
    sys1.add_bearing(N204(100.0, "brg1b"))
    sys2a.add_bearing(N204(25.0,  "brg2a_a", locating=True))
    sys2a.add_bearing(N204(125.0, "brg2a_b"))
    sys2b.add_bearing(N204(25.0,  "brg2b_a", locating=True))
    sys2b.add_bearing(N204(125.0, "brg2b_b"))

    ge_z1  = GearElement(z1,  role="driver", rotation_dir=1, label="z1")
    ge_z2a = GearElement(z2a, role="driven", label="z2a")
    ge_z2b = GearElement(z2b, role="driven", label="z2b")

    sys1.add_gear(ge_z1)
    sys2a.add_gear(ge_z2a)
    sys2b.add_gear(ge_z2b)

    # --- mesh links: shared pinion drives both wheels, ASYMMETRIC split ---
    link_a = SpurHelicalMeshLink(sys1, ge_z1, sys2a, ge_z2a, stage_a,
                                 phi_deg=270.0, torque_split=TORQUE_SPLIT_A,
                                 label="stage_a")
    link_b = SpurHelicalMeshLink(sys1, ge_z1, sys2b, ge_z2b, stage_b,
                                 phi_deg=90.0, torque_split=TORQUE_SPLIT_B,
                                 label="stage_b")

    gearbox = SpurHelicalGearSystem([sys1, sys2a, sys2b],
                                    [link_a, link_b],
                                    label="fanout-test-asymmetric")

    return dict(gearbox=gearbox, sys1=sys1, sys2a=sys2a, sys2b=sys2b,
                stage_a=stage_a, stage_b=stage_b)


# ===========================================================================
# LOAD DUMP  (unchanged)
# ===========================================================================

def dump_shaft_loads(shaft: ShaftSystem) -> None:
    print(f"\n  {shaft.name}  ({len(shaft.loads)} loads)")
    print(f"    {'type':<20} {'x[mm]':>8} {'value':>12} {'theta':>8}  source / label")
    print(f"    {'-'*20} {'-'*8} {'-'*12} {'-'*8}  {'-'*24}")
    for ld in shaft.loads:
        kind  = type(ld).__name__
        theta = f"{ld.theta_deg:8.1f}" if hasattr(ld, "theta_deg") else " " * 8
        src   = getattr(ld, "source", "-")
        lbl   = getattr(ld, "label", "")
        print(f"    {kind:<20} {ld.position:8.1f} {ld.magnitude:12.1f} {theta}  "
              f"[{src}] {lbl}")


# ===========================================================================
# FEM POST-PROCESS  (unchanged)
# ===========================================================================

def recover_diagrams(solver: SimpleFEMSolver, shaft_system: ShaftSystem):
    from axisforge.mesh.oneD.shaft.Elements.Timoshenko_Selective_Integration.timoshenko import TimoshenkoBeam
    beam = TimoshenkoBeam()

    n = len(solver.x_nodes)
    M_xz = np.zeros(n); M_xy = np.zeros(n)
    V_xz = np.zeros(n); V_xy = np.zeros(n)
    N_ax = np.zeros(n)
    done = set()

    for elem in solver.elements:
        dofs = [3*elem.idx_node_1, 3*elem.idx_node_1+1, 3*elem.idx_node_1+2,
                3*elem.idx_node_2, 3*elem.idx_node_2+1, 3*elem.idx_node_2+2]
        Ke   = beam.stiffness_element(elem)
        f_xz = Ke @ solver.d_total_xz[dofs]
        f_xy = Ke @ solver.d_total_xy[dofs]

        for local, node_id in ((0, elem.idx_node_1), (1, elem.idx_node_2)):
            if node_id in done:
                continue
            done.add(node_id)
            sl = slice(0, 3) if local == 0 else slice(3, 6)
            Naxz, Vxz, Mxz = f_xz[sl]
            _,    Vxy, Mxy = f_xy[sl]
            N_ax[node_id] = Naxz
            V_xz[node_id] = Vxz; M_xz[node_id] = Mxz
            V_xy[node_id] = Vxy; M_xy[node_id] = Mxy

    v_xz = np.array([solver.d_total_xz[3*i + 1] for i in range(n)])
    v_xy = np.array([solver.d_total_xy[3*i + 1] for i in range(n)])

    return dict(x=np.array(solver.x_nodes),
                M_xz=M_xz, M_xy=M_xy, V_xz=V_xz, V_xy=V_xy,
                v_xz=v_xz, v_xy=v_xy, N_ax=N_ax,
                T=solver.T_total, tau=solver.tau_total)


# ===========================================================================
# PLOT  (unchanged)
# ===========================================================================

INK = "#1a1a1a"

def plot_shaft(name, d, shaft_system):
    x = d["x"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(f"Shaft analysis - {name}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(x, d["M_xz"], color="#2166ac", lw=1.3, label="M_xz")
    ax.plot(x, d["M_xy"], color="#d62728", lw=1.3, ls="--", label="M_xy")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("Moment\n[N.mm]", fontsize=9)

    ax = axes[1]
    ax.plot(x, d["V_xz"], color="#2166ac", lw=1.3, label="V_xz")
    ax.plot(x, d["V_xy"], color="#d62728", lw=1.3, ls="--", label="V_xy")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("Shear\n[N]", fontsize=9)

    ax = axes[2]
    ax.plot(x, d["v_xz"]*1000, color="#2166ac", lw=1.3, label="v_xz")
    ax.plot(x, d["v_xy"]*1000, color="#d62728", lw=1.3, ls="--", label="v_xy")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("Deflection\n[um]", fontsize=9)

    ax = axes[3]
    ax.plot(x, d["T"], color=INK, lw=1.3, label="T(x)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel("Torque\n[N.m]", fontsize=9)
    ax.set_xlabel("x [mm]", fontsize=9)

    for ax in axes:
        for i, brg in enumerate(shaft_system.bearings):
            ax.axvline(brg.position, color="0.5", lw=0.8, ls=":",
                       label="bearing" if i == 0 else "")
        for i, ge in enumerate(shaft_system.gears):
            ax.axvline(ge.position, color="#33a02c", lw=0.9, ls="-.",
                       label="gear" if i == 0 else "")
        ax.grid(alpha=0.25, lw=0.5, color="0.7")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    banner("AxisForge - FAN-OUT branched system test  (ASYMMETRIC split)")
    print(f"  P = {P_W:.0f} W    n_in = {RPM_IN:.0f} rpm    "
          f"torque_split = {TORQUE_SPLIT_A:.2f} (A) / {TORQUE_SPLIT_B:.2f} (B)")

    all_pass = True

    S = build_branched_system()
    gearbox = S["gearbox"]
    sys1, sys2a, sys2b = S["sys1"], S["sys2a"], S["sys2b"]
    stage_a = S["stage_a"]
    stage_b = S["stage_b"]

    # ------------------------------------------------------------------
    # 0. sanity on the split itself
    # ------------------------------------------------------------------
    section("[ 0 ]  SPLIT SANITY")
    split_sum = TORQUE_SPLIT_A + TORQUE_SPLIT_B
    all_pass &= check("torque_split_A + torque_split_B", split_sum, 1.0)

    # ------------------------------------------------------------------
    # 1. RESOLVE
    # ------------------------------------------------------------------
    section("[ 1 ]  TOPOLOGY + RESOLVE")

    topo = gearbox._topology_errors()
    print(f"\n  topology check : {'OK' if not topo else 'ERRORS'}")
    if topo:
        for e in topo:
            print(f"      - {e}")
        print("\n  Aborting - topology invalid.")
        return

    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))
    print("  resolve()      : completed")

    omega    = RPM_IN * 2.0 * math.pi / 60.0
    T_source = P_W / omega

    print(f"\n  input torque (source)  : {T_source:8.3f} N.m")
    print(f"\n  {'shaft':<16}{'speed[rpm]':>12}{'T_out[N.m]':>13}{'rot':>6}"
          f"{'pos(y,z)[mm]':>20}")
    print(f"  {'-'*16}{'-'*12}{'-'*13}{'-'*6}{'-'*20}")
    for sh in (sys1, sys2a, sys2b):
        r = gearbox._resolved[id(sh)]
        y, z = r["shaft_position"]
        print(f"  {sh.name:<16}{sh.speed_rpm:>12.1f}{r['T_out_Nm']:>13.3f}"
              f"{int(r['rotation_dir']):>+6d}{f'({y:.1f}, {z:.1f})':>20}")

    T_out_a = gearbox._resolved[id(sys2a)]["T_out_Nm"]
    T_out_b = gearbox._resolved[id(sys2b)]["T_out_Nm"]

    exp_a = T_source * TORQUE_SPLIT_A * (60.0 / 20.0)   # branch A: u=3
    exp_b = T_source * TORQUE_SPLIT_B * (40.0 / 20.0)   # branch B: u=2

    print()
    all_pass &= check("T_out branch A (0.7 x u_a=3)", T_out_a, exp_a)
    all_pass &= check("T_out branch B (0.3 x u_b=2)", T_out_b, exp_b)

    # branch A/B ratio should reflect the split ratio, NOT the gear ratio
    # alone -- this is the check that actually catches "split silently
    # ignored / overwritten / averaged to 0.5" bugs.
    ratio_actual   = T_out_a / T_out_b
    ratio_expected = (TORQUE_SPLIT_A * 3.0) / (TORQUE_SPLIT_B * 2.0)
    all_pass &= check("T_out_a / T_out_b ratio", ratio_actual, ratio_expected)

    # ------------------------------------------------------------------
    # 2. LOAD SUMMATION — single pinion carries BOTH meshes at one station
    # ------------------------------------------------------------------
    section("[ 2 ]  LOAD SUMMATION ON INPUT SHAFT (shared pinion)")

    gm_loads  = [l for l in sys1.loads if getattr(l, "source", None) == "gear_mesh"]
    positions = sorted({round(l.position, 1) for l in gm_loads})

    print(f"\n  gear-mesh loads on {sys1.name} : {len(gm_loads)} total")
    print(f"  mesh position(s)                 : {positions} mm")
    print(f"  pinion z1 @55mm present          : "
          f"{'YES' if 55.0 in positions else 'MISSING'}")

    ft_loads = [l for l in sys1.radial_loads
                if l.source == "gear_mesh" and l.label == "z1:Ft"]
    fr_loads = [l for l in sys1.radial_loads
                if l.source == "gear_mesh" and l.label == "z1:Fr"]

    print(f"\n  Ft entries labelled 'z1:Ft'      : {len(ft_loads)}")
    print(f"  Fr entries labelled 'z1:Fr'      : {len(fr_loads)}")
    all_pass &= check("number of Ft entries on z1 (must be 2)", len(ft_loads), 2)

    for i, l in enumerate(ft_loads):
        print(f"      Ft[{i}] = {l.magnitude:8.1f} N  @ theta={l.theta_deg:6.1f} deg")

    # each mesh Ft under the ASYMMETRIC split should scale with its own
    # torque_split, not 0.5.
    rl_a = stage_a.gear1_w.rl      # pinion working pitch radius, mesh A
    rl_b = stage_b.gear1_w.rl      # pinion working pitch radius, mesh B
    ft_a_expected = TORQUE_SPLIT_A * T_source / (rl_a / 1000.0)
    ft_b_expected = TORQUE_SPLIT_B * T_source / (rl_b / 1000.0)
    print(f"\n  pinion pitch radius : mesh A rl={rl_a:.4f} mm   mesh B rl={rl_b:.4f} mm")

    # sort Ft loads by magnitude descending so [0] is always the bigger one
    # (branch A, since 0.7 > 0.3) regardless of internal ordering.
    ft_sorted = sorted(ft_loads, key=lambda l: abs(l.magnitude), reverse=True)
    if len(ft_sorted) == 2:
        all_pass &= check("Ft(A) = split_A*T/rl_a", abs(ft_sorted[0].magnitude), ft_a_expected)
        all_pass &= check("Ft(B) = split_B*T/rl_b", abs(ft_sorted[1].magnitude), ft_b_expected)

    fy = sum(getattr(l, "Fy", 0.0) for l in sys1.radial_loads
             if l.source == "gear_mesh")
    fz = sum(getattr(l, "Fz", 0.0) for l in sys1.radial_loads
             if l.source == "gear_mesh")
    net = math.hypot(fy, fz)
    print(f"\n  net radial force on pinion : Fy={fy:8.1f} N  Fz={fz:8.1f} N  "
          f"|F|={net:8.1f} N")
    print(f"  asymmetry produces bending : "
          f"{'YES (|F| > 0)' if net > 1.0 else 'NO (still cancels)'}")

    print("\n  full load dump per shaft:")
    for sh in (sys1, sys2a, sys2b):
        dump_shaft_loads(sh)

    # ------------------------------------------------------------------
    # 3. FEM
    # ------------------------------------------------------------------
    section("[ 3 ]  FEM SOLVE (all three shafts)")

    diagrams = {}
    for sh in (sys1, sys2a, sys2b):
        solver = SimpleFEMSolver(theory="timoshenko")
        solver.solve(sh)
        d = recover_diagrams(solver, sh)
        diagrams[sh.name] = d

        M_res = np.hypot(d["M_xz"], d["M_xy"])
        v_res = np.hypot(d["v_xz"], d["v_xy"]) * 1000
        finite = np.all(np.isfinite(d["v_xz"])) and np.all(np.isfinite(d["v_xy"]))
        print(f"\n  {sh.name:<16} nodes={len(d['x']):>4}   "
              f"max|M|={M_res.max():9.1f} N.mm   "
              f"max|v|={v_res.max():7.2f} um   "
              f"{'finite' if finite else 'NON-FINITE'}")
        all_pass &= finite

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    section("[ SUMMARY ]")
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED -- see [FAIL] lines above'}")

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    for sh in (sys1, sys2a, sys2b):
        plot_shaft(sh.name, diagrams[sh.name], sh)

    banner("Test complete - close plot windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()