"""
design_load_lifting_theta_sweep.py

Fixed b=20 mm. Sweeps the rate of theta variation along the face width:
    dtheta_dx = 0, 0.5, 1, 2, 5, 10  [deg/mm]

dtheta_dx=0 is the constant-theta reference (uniform distribution).

Output: equivalent (resultant) quantities only:
    |M|(x)     = sqrt(M_xz² + M_xy²)   [N·mm]
    sigma_b(x) = |M| / W(x)             [MPa]
    |v|(x)     = sqrt(v_xz² + v_xy²)   [µm]
    |V|(x)     = sqrt(V_xz² + V_xy²)   [N]

Two figures per shaft:
  Fig A — |M| and |v|  (side by side)
  Fig B — sigma_b and |V|  (side by side)
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
from axisforge.core.loads import RadialLoad, TorqueLoad, DistributedRadialLoad
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import GearElement, ShaftSystem
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalMeshLink, SpurHelicalGearSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver

# ===========================================================================
# PARAMETERS
# ===========================================================================

B           = 20.0    # fixed face width [mm]

# Cases: (label, theta_fn_factory)
# theta_fn_factory(theta_0, x_lo) -> callable(x) -> deg
def _linear(rate):
    """Linear: theta(x) = theta_0 + rate*(x - x_lo)"""
    return lambda theta_0, x_lo: (
        lambda x, t0=theta_0, xl=x_lo, r=rate: (t0 + r * (x - xl)) % 360.0
    )

def _quadratic(a):
    """Quadratic: theta(x) = theta_0 + a*(x - x_lo)**2"""
    return lambda theta_0, x_lo: (
        lambda x, t0=theta_0, xl=x_lo, _a=a: (t0 + _a * (x - xl)**2) % 360.0
    )

CASES = [
    ("dθ/dx=0 °/mm",    _linear(0)),
    ("dθ/dx=0.5 °/mm",  _linear(0.5)),
    ("dθ/dx=1 °/mm",    _linear(1)),
    ("dθ/dx=2 °/mm",    _linear(2)),
    ("quadratic a=0.05",           _quadratic(0.05)),
    ("quadratic a=0.1",            _quadratic(0.1)),
]
COLORS = ["#1a1a1a", "#2166ac", "#d62728", "#33a02c", "#ff7f00", "#984ea3"]

P_W         = 1000.0
RPM_IN      = 450.0
r_pulley    = 0.050
SHAFT_NAMES = ["shaft1(motor)", "shaft2", "shaft3(pulley)"]

GEAR_KW = dict(mn=2.0, x=0.0, b=B, alpha_n_deg=20.0, Ra=0.8,
               material_id="42CrMo4")


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
    W_local = np.array([shaft_system.shaft.W_at(x) for x in solver.x_nodes])
    # --- equivalent (resultant) quantities ---
    M_eq    = np.sqrt(M_xz**2 + M_xy**2)
    V_eq    = np.sqrt(V_xz**2 + V_xy**2)
    v_eq    = np.sqrt(v_xz**2 + v_xy**2) * 1000.0   # µm
    sigma_b = M_eq / W_local
    return dict(
        x       = np.array(solver.x_nodes),
        M_eq    = M_eq,
        V_eq    = V_eq,
        v_eq    = v_eq,
        sigma_b = sigma_b,
        T       = solver.T_total,
        tau     = solver.tau_total,
    )


def build_and_solve(label: str, theta_factory) -> dict:
    """
    Build, resolve and solve with b=20 mm.
    dtheta_dx: rate of angular variation [deg/mm] along the face width.
    dtheta_dx=0 -> constant theta (uniform distribution).
    """
    z1 = SpurHelicalGear(z=20, position=55.0, label="z1", **GEAR_KW)
    z2 = SpurHelicalGear(z=60, position=95.0, label="z2", **GEAR_KW)
    z3 = SpurHelicalGear(z=20, position=55.0, label="z3", **GEAR_KW)
    z4 = SpurHelicalGear(z=60, position=55.0, label="z4", **GEAR_KW)

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

    link1 = SpurHelicalMeshLink(sys1, ge_z1, sys2, ge_z2, stage1,
                                 phi_deg=270.0, distribute_loads=True, label="stage1")
    link2 = SpurHelicalMeshLink(sys2, ge_z3, sys3, ge_z4, stage2,
                                 phi_deg=270.0, distribute_loads=True, label="stage2")

    gearbox = SpurHelicalGearSystem([sys1, sys2, sys3], [link1, link2],
                                     label="load-lifting")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    # --- replace theta with the provided factory ---
    for sh in (sys1, sys2, sys3):
        new_loads = []
        for ld in sh._loads:
            if (isinstance(ld, DistributedRadialLoad)
                    and ld.source == "gear_mesh"
                    and not ld._theta_variable):
                theta_fn = theta_factory(ld.theta_deg, ld.x_lo)
                new_loads.append(DistributedRadialLoad(
                    ld.x_lo, ld.x_hi,
                    magnitude=ld._q_fn,
                    theta_deg=theta_fn,
                    label=ld.label,
                    source=ld.source,
                ))
            else:
                new_loads.append(ld)
        sh._loads = new_loads

    T3     = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    solvers = {}
    for sh in (sys1, sys2, sys3):
        s = SimpleFEMSolver(theory="timoshenko", distribute_gear_labels={"*"})
        s.solve(sh)
        solvers[sh.name] = s

    return {sh.name: recover_diagrams(solvers[sh.name], sh)
            for sh in (sys1, sys2, sys3)}


# ===========================================================================
# RUN SWEEP
# ===========================================================================

results = {}
for lbl, factory in CASES:
    print(f"  Solving case: {lbl} ...")
    results[lbl] = build_and_solve(lbl, factory)

print("All cases complete.\n")


# ===========================================================================
# PLOTS — two figures per shaft
# ===========================================================================

def _plot_shaft(shaft_name: str) -> None:
    # Fig A: |M| and |v|
    fig_a, (ax_M, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
    fig_a.suptitle(f"{shaft_name} — |M|  and  |v|  (b={B:.0f} mm, θ sweep)",
                   fontsize=12, fontweight="bold")

    # Fig B: sigma_b and |V|
    fig_b, (ax_s, ax_V) = plt.subplots(1, 2, figsize=(14, 5))
    fig_b.suptitle(f"{shaft_name} — σ_b  and  |V|  (b={B:.0f} mm, θ sweep)",
                   fontsize=12, fontweight="bold")

    for (lbl, _), color in zip(CASES, COLORS):
        d = results[lbl][shaft_name]
        x = d["x"]
        ax_M.plot(x, d["M_eq"],    color=color, lw=1.3, label=lbl)
        ax_v.plot(x, d["v_eq"],    color=color, lw=1.3, label=lbl)
        ax_s.plot(x, d["sigma_b"], color=color, lw=1.3, label=lbl)
        ax_V.plot(x, d["V_eq"],    color=color, lw=1.3, label=lbl)

    for ax, ylabel, title in [
        (ax_M, "|M| [N·mm]",  "|M| resultante"),
        (ax_v, "|v| [µm]",    "|v| deflexão"),
        (ax_s, "σ_b [MPa]",   "tensão de flexão"),
        (ax_V, "|V| [N]",     "|V| esforço transverso"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.grid(alpha=0.2, lw=0.4, color="0.7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False)

    fig_a.tight_layout(rect=[0, 0, 1, 0.95])
    fig_b.tight_layout(rect=[0, 0, 1, 0.95])


for shaft_name in SHAFT_NAMES:
    _plot_shaft(shaft_name)


# ===========================================================================
# PEAK VALUES TABLE
# ===========================================================================

print("=" * 95)
print(f"  PEAK VALUES — b={B:.0f} mm, dtheta/dx sweep")
print("=" * 95)
hdrs = [lbl for lbl, _ in CASES]
print(f"  {'shaft':<18}  {'qty':<14}  "
      + "  ".join(f"{h:>16}" for h in hdrs))
print("  " + "-" * 110)

for shaft_name in SHAFT_NAMES:
    for qty, fn in [
        ("|M|_max [N.mm]", lambda d: d["M_eq"].max()),
        ("sb_max [MPa]",   lambda d: d["sigma_b"].max()),
        ("|v|_max [um]",   lambda d: d["v_eq"].max()),
        ("|V|_max [N]",    lambda d: d["V_eq"].max()),
    ]:
        vals = [fn(results[lbl][shaft_name]) for lbl, _ in CASES]
        print(f"  {shaft_name:<18}  {qty:<14}  "
              + "  ".join(f"{v:>16.2f}" for v in vals))
    print()

print("=" * 95)

plt.show()