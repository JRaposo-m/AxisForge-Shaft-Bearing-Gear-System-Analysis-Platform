"""
design_load_lifting_bsweep_converged.py

Mesmo sweep de b-values, mas agora:
  1. Para cada (shaft, b), corre MeshConvergenceStudy para obter extra_nodes.
  2. Injeta esses extra_nodes via Mesh1D.add_mandatory_positions no solve final.
  3. Usa ShaftResultsReader para extrair resultados (não recover_diagrams manual).
  4. Imprime relatório de convergência + tabela de picos.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.deep_groove import DeepGrooveBallFamily
from axisforge.core.machine_elements.bearings.families.roller_bearing.radial.subtypes.cylindrical_roller import CylindricalRollerFamily
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.loads import RadialLoad, TorqueLoad, AxialLoad
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.solvers.mesh.mesh_convergence_study import MeshConvergenceStudy

# ===========================================================================
# PARAMETERS
# ===========================================================================

B_VALUES    = [0, 1, 2, 5, 10, 15, 20]
COLORS      = ["#1a1a1a", "#2166ac", "#d62728", "#33a02c", "#ff7f00", "#984ea3", "#a65628"]
P_W         = 1000.0
RPM_IN      = 450.0
r_pulley    = 0.050
SHAFT_NAMES = ["shaft1(motor)", "shaft2", "shaft3(pulley)"]
GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

CONV_TOL        = 1e-6
CONV_MAX_LEVELS = 5
CONV_METRIC     = "sigma_b"   # "sigma_b" | "M_max" | "l2_M"

# ===========================================================================
# HELPERS (iguais ao original)
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
    return Bearing(
        d=20.0, D=47.0,
        bearing_type=BearingType.CYLINDRICAL_ROLLER,
        designation="N204",
        b=14.0, C=28_500.0, C0=19_000.0,
        arrangement="locating" if locating else "floating",
        contact_angle_deg=0.0,
        X=1.0, Y=0.0,
        label=label,
        position=position,
    )


# ===========================================================================
# BUILD + CONVERGENCE + SOLVE
# ===========================================================================

def build_gear_system(b: float):
    """Constrói e faz resolve() do gearbox. Devolve (gearbox, sys1, sys2, sys3)."""
    distribute = b > 0.0
    gear_kw    = dict(**GEAR_KW_BASE, b=b)

    z1 = SpurHelicalGear(z=20, position=55.0, label="z1", **gear_kw)
    z2 = SpurHelicalGear(z=60, position=95.0, label="z2", **gear_kw)
    z3 = SpurHelicalGear(z=20, position=55.0, label="z3", **gear_kw)
    z4 = SpurHelicalGear(z=60, position=55.0, label="z4", **gear_kw)

    s1 = make_stepped_shaft("shaft1", 120.0, 20.0, 25.0, 30.0, 30.0, 1.5)
    s2 = make_stepped_shaft("shaft2", 150.0, 20.0, 28.0, 35.0, 35.0, 2.0)
    s3 = make_stepped_shaft("shaft3", 150.0, 20.0, 25.0, 35.0, 35.0, 1.5)

    sys1 = ShaftSystem(s1, name="shaft1(motor)",  speed_rpm=450.0)
    sys2 = ShaftSystem(s2, name="shaft2",         speed_rpm=150.0)
    sys3 = ShaftSystem(s3, name="shaft3(pulley)", speed_rpm=50.0)

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

    sys1.shaft_origin_x = 0.0
    sys2.shaft_origin_x = sys1.shaft_origin_x + (z1.position - z2.position)
    sys3.shaft_origin_x = sys2.shaft_origin_x + (z3.position - z4.position)

    link1 = SpurHelicalMeshLink(sys1, ge_z1, sys2, ge_z2,
                                 SpurHelicalGearMeshing(z1, z2, label="stage1"),
                                 phi_deg=270.0, distribute_loads=distribute, label="stage1")
    link2 = SpurHelicalMeshLink(sys2, ge_z3, sys3, ge_z4,
                                 SpurHelicalGearMeshing(z3, z4, label="stage2"),
                                 phi_deg=270.0, distribute_loads=distribute, label="stage2")

    gearbox = SpurHelicalGearSystem([sys1, sys2, sys3], [link1, link2],
                                     label="load-lifting")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    T3     = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    return sys1, sys2, sys3, distribute


def converge_and_solve(shaft_system, distribute, verbose=False):
    dist_labels = {"*"} if distribute else None
    theory      = "timoshenko"

    extra_nodes: list[float] = []
    if shaft_system.distributed_radial_loads:
        # 1. solve global primeiro — obrigatório antes do MeshConvergenceStudy
        global_solver = SimpleFEMSolver(theory=theory,
                                        distribute_gear_labels=dist_labels)
        global_solver.solve(shaft_system)

        # 2. derivar intervals automaticamente
        intervals, skipped = MeshConvergenceStudy.intervals_from_shaft_system(shaft_system)
        if skipped and verbose:
            for s in skipped:
                print(f"  [skipped] {s}")

        if intervals:
            study = MeshConvergenceStudy(global_solver,
                                         gci_threshold=CONV_TOL,
                                         max_levels=CONV_MAX_LEVELS)
            conv_result = study.run(shaft_system, intervals)
            extra_nodes = conv_result.all_extra_nodes
            if verbose:
                conv_result.print_report()

    # 3. solve final com extra_nodes convergidos
    solver = SimpleFEMSolver(theory=theory, distribute_gear_labels=dist_labels)
    solver.solve(shaft_system, extra_mandatory=extra_nodes or None)
    return solver


def build_and_solve(b: float, verbose_conv: bool = False) -> dict[str, dict]:
    """
    Constrói, converge e resolve. Devolve dict[shaft_name -> ShaftResults].
    """
    sys1, sys2, sys3, distribute = build_gear_system(b)

    results = {}
    for shaft_sys in (sys1, sys2, sys3):
        solver  = converge_and_solve(shaft_sys, distribute, verbose=verbose_conv)
        reader  = ShaftResultsReader(solver, shaft_sys)
        results[shaft_sys.name] = reader.read()

    return results


# ===========================================================================
# RUN SWEEP
# ===========================================================================

print("Running b-sweep with mesh convergence...\n")
all_results: dict[int, dict[str, object]] = {}   # b -> {shaft_name -> ShaftResults}

for b in B_VALUES:
    print(f"  b={b:>3} mm ...", end="", flush=True)
    all_results[b] = build_and_solve(b, verbose_conv=False)
    print(" done")

print("\nAll cases complete.\n")


# ===========================================================================
# PEAK VALUES TABLE
# ===========================================================================

print("=" * 95)
print("  PEAK VALUES SWEEP — load-lifting gearbox")
print("=" * 95)
header = f"  {'shaft':<18}  {'quantity':<16}  " + "  ".join(f"b={b:>2}mm" for b in B_VALUES)
print(header)
print("  " + "-" * (len(header) - 2))

PEAK_QUANTITIES = [
    ("max|M| [N·mm]", lambda r: r.M_max),
    ("max σ_b [MPa]", lambda r: r.sigma_b_max),
    ("max v   [µm]",  lambda r: r.v_max * 1000.0),
    ("max|V|  [N]",   lambda r: float(r.V.max())),
    ("max τ   [MPa]", lambda r: r.tau_max),
]

for shaft_name in SHAFT_NAMES:
    for qty_label, fn in PEAK_QUANTITIES:
        vals = [fn(all_results[b][shaft_name]) for b in B_VALUES]
        row  = f"  {shaft_name:<18}  {qty_label:<16}  "
        row += "  ".join(f"{v:>7.2f}" for v in vals)
        print(row)
    print()

print("=" * 95)



# ===========================================================================
# PLOTS — bending diagrams (M, V, v) — figura por veio
# ===========================================================================

PLOT_QUANTITIES = [
    ("M",          "M_xz",  "M_xy",  "Moment [N·mm]", 1.0),
    ("V",          "V_xz",  "V_xy",  "Shear [N]",     1.0),
    ("deflection", "v_xz",  "v_xy",  "v [µm]",        1000.0),
]

def _plot_bending(shaft_name: str) -> None:
    nrows = len(PLOT_QUANTITIES)
    fig = plt.figure(figsize=(16, 3.6 * nrows))
    fig.suptitle(f"{shaft_name} — M / V / v  (b-sweep, converged mesh)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(nrows, 2, figure=fig, hspace=0.45, wspace=0.30)

    for row, (title, key_xz, key_xy, ylabel, scale) in enumerate(PLOT_QUANTITIES):
        for col, (key, plane) in enumerate([(key_xz, "XZ"), (key_xy, "XY")]):
            ax = fig.add_subplot(gs[row, col])
            for b, color in zip(B_VALUES, COLORS):
                r = all_results[b][shaft_name]
                ax.plot(r.x, getattr(r, key) * scale, color=color, lw=1.2, label=f"b={b}mm")
            ax.set_title(f"{title} — {plane}", fontsize=9)
            ax.set_xlabel("x [mm]", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.grid(alpha=0.2, lw=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, frameon=False, ncol=2)


# ===========================================================================
# PLOTS — stress diagrams (σ_b, T, τ) — figura separada por veio
# ===========================================================================

PLOT_SCALARS = [
    ("Bending stress  σ_b", "sigma_b", "σ_b [MPa]"),
    ("Torque  T",           "T",       "T [N·m]"),
    ("Torsional shear  τ",  "tau",     "τ [MPa]"),
]

def _plot_stress(shaft_name: str) -> None:
    nrows = len(PLOT_SCALARS)
    fig = plt.figure(figsize=(10, 3.8 * nrows))
    fig.suptitle(f"{shaft_name} — σ_b / T / τ  (b-sweep, converged mesh)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.50)

    for row, (title, key, ylabel) in enumerate(PLOT_SCALARS):
        ax = fig.add_subplot(gs[row, 0])
        for b, color in zip(B_VALUES, COLORS):
            r = all_results[b][shaft_name]
            ax.plot(r.x, getattr(r, key), color=color, lw=1.4, label=f"b={b}mm")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False, ncol=len(B_VALUES))


# --- render ---
for shaft_name in SHAFT_NAMES:
    _plot_bending(shaft_name)
    _plot_stress(shaft_name)


# ===========================================================================
# PRINT — nós e resultados nodais por veio e por b
# ===========================================================================

def print_nodal_results(b: float, shaft_name: str) -> None:
    r = all_results[b][shaft_name]

    print(f"\n{'='*80}")
    print(f"  {shaft_name}  |  b = {b} mm  |  {len(r.x)} nodes")
    print(f"{'='*80}")
    print(f"  {'node':>5}  {'x [mm]':>9}  {'M [N·mm]':>12}  {'V [N]':>10}  "
          f"{'v [µm]':>10}  {'σ_b [MPa]':>11}  {'T [N·m]':>9}  {'τ [MPa]':>9}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*9}  {'-'*9}")

    for i, x in enumerate(r.x):
        print(f"  {i:>5}  {x:>9.3f}  {r.M[i]:>12.2f}  {r.V[i]:>10.2f}  "
              f"{r.v[i]*1000:>10.4f}  {r.sigma_b[i]:>11.4f}  "
              f"{r.T[i]:>9.4f}  {r.tau[i]:>9.4f}")

    print(f"\n  PEAKS:")
    print(f"    M_max    = {r.M_max:>12.2f} N·mm  @ x = {r.x_M_max:.3f} mm")
    print(f"    σ_b_max  = {r.sigma_b_max:>12.4f} MPa   @ x = {r.x_sigma_b_max:.3f} mm")
    print(f"    v_max    = {r.v_max*1000:>12.4f} µm    @ x = {r.x_v_max:.3f} mm")
    print(f"    τ_max    = {r.tau_max:>12.4f} MPa   @ x = {r.x_tau_max:.3f} mm")


# ===========================================================================
# PRINT — nós de convergência (MeshRefinementResult) por veio e por b
# ===========================================================================

def print_convergence_nodes(b: float, shaft_name: str,
                             sys: ShaftSystem, distribute: bool) -> None:
    if not sys.distributed_radial_loads:
        print(f"\n  [{shaft_name} | b={b}mm]  sem distributed_radial_loads — sem estudo de convergência.")
        return

    dist_labels  = {"*"} if distribute else None
    probe_solver = SimpleFEMSolver(theory="timoshenko",
                                    distribute_gear_labels=dist_labels)
    study  = MeshConvergenceStudy(probe_solver,
                                   tol=CONV_TOL,
                                   max_levels=CONV_MAX_LEVELS,
                                   metric=CONV_METRIC)
    result = study.run(sys)

    print(f"\n{'='*80}")
    print(f"  CONVERGENCE NODES  |  {shaft_name}  |  b = {b} mm")
    print(f"{'='*80}")
    result.print_report()


# --- executa prints ---
# guarda os ShaftSystems para o relatório de convergência
# (precisamos reconstruir porque build_and_solve não os expõe)
# solução: expor build_gear_system separadamente e guardar sistemas

print("\n\n" + "#"*80)
print("  NODAL RESULTS")
print("#"*80)

for b in B_VALUES:
    for shaft_name in SHAFT_NAMES:
        print_nodal_results(b, shaft_name)

plt.show()