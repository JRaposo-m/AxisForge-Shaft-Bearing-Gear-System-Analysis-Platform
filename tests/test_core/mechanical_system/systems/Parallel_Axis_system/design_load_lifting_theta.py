"""
design_load_lifting_theta_sweep.py  (com convergência de malha + prints nodais)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.loads import RadialLoad, TorqueLoad, DistributedRadialLoad
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.solvers.mesh.convergence_solver import MeshConvergenceStudy

# ===========================================================================
# PARAMETERS
# ===========================================================================

B = 20.0

def _linear(rate):
    return lambda theta_0, x_lo: (
        lambda x, t0=theta_0, xl=x_lo, r=rate: (t0 + r * (x - xl)) % 360.0
    )

def _quadratic(a):
    return lambda theta_0, x_lo: (
        lambda x, t0=theta_0, xl=x_lo, _a=a: (t0 + _a * (x - xl)**2) % 360.0
    )

CASES = [
    ("dθ/dx=0 °/mm",     _linear(0)),
    ("dθ/dx=0.5 °/mm",   _linear(0.5)),
    ("dθ/dx=1 °/mm",     _linear(1)),
    ("dθ/dx=2 °/mm",     _linear(2)),
    ("quadratic a=0.05",  _quadratic(0.05)),
    ("quadratic a=0.1",   _quadratic(0.1)),
]
COLORS = ["#1a1a1a", "#2166ac", "#d62728", "#33a02c", "#ff7f00", "#984ea3"]

P_W         = 1000.0
RPM_IN      = 450.0
r_pulley    = 0.050
SHAFT_NAMES = ["shaft1(motor)", "shaft2", "shaft3(pulley)"]
GEAR_KW     = dict(mn=2.0, x=0.0, b=B, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

CONV_TOL        = 1e-3
CONV_MAX_LEVELS = 5
CONV_METRIC     = "sigma_b"

# ===========================================================================
# HELPERS
# ===========================================================================

def make_stepped_shaft(name, total_length, d_seat, d_body,
                        l_seat_a, l_seat_b, fillet_r, material_id="AISI_1045"):
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


# ===========================================================================
# BUILD — gearbox + theta substitution, devolve (sys1, sys2, sys3)
# ===========================================================================

def build_systems(theta_factory) -> tuple[ShaftSystem, ShaftSystem, ShaftSystem]:
    z1 = SpurHelicalGear(z=20, position=55.0, label="z1", **GEAR_KW)
    z2 = SpurHelicalGear(z=60, position=95.0, label="z2", **GEAR_KW)
    z3 = SpurHelicalGear(z=20, position=55.0, label="z3", **GEAR_KW)
    z4 = SpurHelicalGear(z=60, position=55.0, label="z4", **GEAR_KW)

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

    link1 = SpurHelicalMeshLink(sys1, ge_z1, sys2, ge_z2,
                                 SpurHelicalGearMeshing(z1, z2, label="stage1"),
                                 phi_deg=270.0, distribute_loads=True, label="stage1")
    link2 = SpurHelicalMeshLink(sys2, ge_z3, sys3, ge_z4,
                                 SpurHelicalGearMeshing(z3, z4, label="stage2"),
                                 phi_deg=270.0, distribute_loads=True, label="stage2")

    gearbox = SpurHelicalGearSystem([sys1, sys2, sys3], [link1, link2],
                                     label="load-lifting")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    # substituir theta nas DistributedRadialLoad de gear_mesh
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

    T3      = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope  = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    return sys1, sys2, sys3


# ===========================================================================
# CONVERGENCE + SOLVE — um ShaftSystem de cada vez
# ===========================================================================

def converge_and_solve(shaft_sys: ShaftSystem,
                        verbose: bool = False) -> tuple[object, object]:
    """
    Devolve (ShaftResults, MeshRefinementResult | None).
    MeshRefinementResult é None se não houver cargas distribuídas.
    """
    dist_labels = {"*"}
    theory      = "timoshenko"

    conv_result = None
    extra_nodes: list[float] = []

    if shaft_sys.distributed_radial_loads:
        probe = SimpleFEMSolver(theory=theory, distribute_gear_labels=dist_labels)
        study = MeshConvergenceStudy(probe,
                                      tol=CONV_TOL,
                                      max_levels=CONV_MAX_LEVELS,
                                      metric=CONV_METRIC)
        conv_result = study.run(shaft_sys)
        extra_nodes = conv_result.all_extra_nodes
        if verbose:
            conv_result.print_report()

    mesh = Mesh1D(shaft_sys)
    if extra_nodes:
        mesh.add_mandatory_positions(extra_nodes)

    solver = SimpleFEMSolver(theory=theory, distribute_gear_labels=dist_labels)
    solver.solve(shaft_sys, mesh=mesh)

    results = ShaftResultsReader(solver, shaft_sys).read()
    return results, conv_result


# ===========================================================================
# RUN SWEEP
# ===========================================================================

# all_results[case_label][shaft_name] = ShaftResults
# all_conv[case_label][shaft_name]    = MeshRefinementResult | None
all_results: dict[str, dict] = {}
all_conv:    dict[str, dict] = {}

for case_lbl, factory in CASES:
    print(f"  Solving: {case_lbl} ...")
    sys1, sys2, sys3 = build_systems(factory)

    res_case  = {}
    conv_case = {}
    for sh in (sys1, sys2, sys3):
        r, c = converge_and_solve(sh, verbose=False)
        res_case[sh.name]  = r
        conv_case[sh.name] = c

    all_results[case_lbl] = res_case
    all_conv[case_lbl]    = conv_case

print("All cases complete.\n")


# ===========================================================================
# PRINT — convergência por caso e veio
# ===========================================================================

print("\n" + "#" * 80)
print("  MESH CONVERGENCE REPORTS")
print("#" * 80)

for case_lbl, _ in CASES:
    for shaft_name in SHAFT_NAMES:
        c = all_conv[case_lbl][shaft_name]
        print(f"\n{'='*70}")
        print(f"  case: {case_lbl}  |  {shaft_name}")
        print(f"{'='*70}")
        if c is None:
            print("  (sem distributed_radial_loads — sem estudo de convergência)")
        else:
            c.print_report()


# ===========================================================================
# PRINT — resultados nodais por caso e veio
# ===========================================================================

print("\n" + "#" * 80)
print("  NODAL RESULTS")
print("#" * 80)

for case_lbl, _ in CASES:
    for shaft_name in SHAFT_NAMES:
        r = all_results[case_lbl][shaft_name]

        print(f"\n{'='*90}")
        print(f"  case: {case_lbl}  |  {shaft_name}  |  {len(r.x)} nodes")
        print(f"{'='*90}")
        print(f"  {'node':>5}  {'x [mm]':>9}  {'M [N·mm]':>12}  {'V [N]':>10}  "
              f"{'v [µm]':>9}  {'σ_b [MPa]':>11}  {'T [N·m]':>9}  "
              f"{'τ [MPa]':>9}  {'σ_eq [MPa]':>11}")
        print(f"  {'-'*5}  {'-'*9}  {'-'*12}  {'-'*10}  "
              f"{'-'*9}  {'-'*11}  {'-'*9}  {'-'*9}  {'-'*11}")

        for i, x in enumerate(r.x):
            sb  = r.sigma_b[i]
            tau = r.tau[i]
            seq = float(np.sqrt(sb**2 + 3.0 * tau**2))
            print(f"  {i:>5}  {x:>9.3f}  {r.M[i]:>12.2f}  {r.V[i]:>10.2f}  "
                  f"{r.v[i]*1000:>9.4f}  {sb:>11.4f}  "
                  f"{r.T[i]:>9.4f}  {tau:>9.4f}  {seq:>11.4f}")

        print(f"\n  PEAKS:")
        print(f"    M_max   = {r.M_max:>12.2f} N·mm  @ x={r.x_M_max:.3f} mm")
        print(f"    σ_b_max = {r.sigma_b_max:>12.4f} MPa   @ x={r.x_sigma_b_max:.3f} mm")
        print(f"    v_max   = {r.v_max*1000:>12.4f} µm    @ x={r.x_v_max:.3f} mm")
        print(f"    τ_max   = {r.tau_max:>12.4f} MPa   @ x={r.x_tau_max:.3f} mm")


# ===========================================================================
# PEAK VALUES TABLE
# ===========================================================================

print("\n" + "=" * 100)
print(f"  PEAK VALUES — b={B:.0f} mm, θ sweep")
print("=" * 100)
hdrs = [lbl for lbl, _ in CASES]
print(f"  {'shaft':<18}  {'quantity':<16}  " + "  ".join(f"{h:>18}" for h in hdrs))
print("  " + "-" * 130)

PEAK_FNS = [
    ("|M|_max [N·mm]", lambda r: r.M_max),
    ("σ_b_max [MPa]",  lambda r: r.sigma_b_max),
    ("v_max [µm]",     lambda r: r.v_max * 1000.0),
    ("|V|_max [N]",    lambda r: float(r.V.max())),
    ("τ_max [MPa]",    lambda r: r.tau_max),
]

for shaft_name in SHAFT_NAMES:
    for qty_lbl, fn in PEAK_FNS:
        vals = [fn(all_results[lbl][shaft_name]) for lbl, _ in CASES]
        print(f"  {shaft_name:<18}  {qty_lbl:<16}  "
              + "  ".join(f"{v:>18.3f}" for v in vals))
    print()

print("=" * 100)


# ===========================================================================
# PLOTS — M / V / v  (XZ | XY por veio)
# ===========================================================================

QUANTITIES_A = [
    ("M", "M_xz", "M_xy", "Moment [N·mm]"),
    ("V", "V_xz", "V_xy", "Shear [N]"),
    ("v", "v_xz", "v_xy", "Deflection [µm]"),
]

SCALE_A = {"v_xz": 1000.0, "v_xy": 1000.0}


def _plot_bending(shaft_name: str) -> None:
    nrows = len(QUANTITIES_A)
    fig = plt.figure(figsize=(16, 3.6 * nrows))
    fig.suptitle(f"{shaft_name} — M / V / v  (b={B:.0f} mm, θ sweep)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.45, wspace=0.30)

    for row, (title, key_xz, key_xy, ylabel) in enumerate(QUANTITIES_A):
        for col, (key, plane) in enumerate([(key_xz, "XZ"), (key_xy, "XY")]):
            ax = fig.add_subplot(gs[row, col])
            scale = SCALE_A.get(key, 1.0)
            for (lbl, _), color in zip(CASES, COLORS):
                r = all_results[lbl][shaft_name]
                ax.plot(r.x, getattr(r, key) * scale, color=color, lw=1.2, label=lbl)
            ax.set_title(f"{title} — {plane}", fontsize=9)
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.grid(alpha=0.2, lw=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, frameon=False, ncol=2)


# ===========================================================================
# PLOTS — σ_b / T / τ  (figura separada por veio)
# ===========================================================================

QUANTITIES_B = [
    ("Bending stress  σ_b", "sigma_b", "σ_b [MPa]"),
    ("Torque  T",           "T",       "T [N·m]"),
    ("Torsional shear  τ",  "tau",     "τ [MPa]"),
]


def _plot_stress(shaft_name: str) -> None:
    nrows = len(QUANTITIES_B)
    fig = plt.figure(figsize=(10, 3.8 * nrows))
    fig.suptitle(f"{shaft_name} — σ_b / T / τ  (b={B:.0f} mm, θ sweep)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.50)

    for row, (title, key, ylabel) in enumerate(QUANTITIES_B):
        ax = fig.add_subplot(gs[row, 0])
        for (lbl, _), color in zip(CASES, COLORS):
            r = all_results[lbl][shaft_name]
            ax.plot(r.x, getattr(r, key), color=color, lw=1.4, label=lbl)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False, ncol=3)


for shaft_name in SHAFT_NAMES:
    _plot_bending(shaft_name)
    _plot_stress(shaft_name)

plt.show()