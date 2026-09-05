"""
design_convergence_study.py

Mesh convergence study for the load-lifting gearbox at a fixed face width b.

For each shaft and each convergence interval (gears, bearings, distributed
loads with b defined), runs MeshConvergenceStudy and plots:
  - per level: v = sqrt(v_xz^2 + v_xy^2) at the evaluation point(s) [um]
  - GCI values per level transition
  - convergence status

One figure per shaft, one subplot per interval.

UPDATED, this turn -- bearing definitions migrated to Bearing.assemble()
-----------------------------------------------------------------------
The old N204() factory built a Bearing via the flat, pre-rewrite
constructor (Bearing(d=..., bearing_type=BearingType.CYLINDRICAL_ROLLER,
arrangement="locating"/"floating", ...)). That constructor no longer
matches the current core/machine_elements/bearings architecture, where
every Bearing is built through Bearing.assemble(family, catalog, geometry,
analyses) -- see design_double_row_life_verification.py for the pattern
this was ported from.

Porting surfaced a real (not just syntactic) issue: CylindricalRollerFamily
rejects catalog.arrangement="locating" with a ValueError -- an N-type
roller bearing has no flange to react axial load, so it structurally
cannot be the locating bearing. The old script called N204(..., locating
=True) for the "a" position on every shaft; that call would now raise.

Per your choice, this is fixed by changing the topology, not by lying to
the family: the locating position on each shaft is now a deep-groove ball
bearing (DGBB6204, DeepGrooveBallFamily, arrangement="locating"), and the
floating position stays a cylindrical roller (N204, CylindricalRollerFamily,
arrangement="floating") -- same locating/floating split as
design_double_row_life_verification.py's DOUBLE_HOMO_SCENARIO, just
single-row instead of double-row.

Catalog numbers carried over as-is from the old N204() (b=14.0, C=1000,
C0=19_000.0) -- NOT re-derived from a real SKF N204 catalog page here, so
if C=1000 N was already a placeholder in your original script it still is
one now; only the construction API changed, not those values. The internal
roller geometry (Dwe, Lwe, Dpw, Z, s, n_s) is new -- the old flat
constructor never carried it (no line-contact solve was reachable from
it), so it did not exist to "carry over"; borrowed here from
design_double_row_life_verification.py's CR_GEOM, which is for the same
204-series envelope (d=20 mm, D=47 mm). The new DGBB6204 locating bearing
is entirely new (the old script had no ball bearing at all) -- its
geometry/capacity (BB_GEOM, C=12_700 N, C0=6_550 N) is also borrowed
unchanged from that reference script (standard SKF 6204 dynamic rating;
C0 illustrative). Adjust either block if you have real catalog data for
this specific design.

This script itself never calls the ISO/TS 16281 capacity/life solvers
(only SimpleFEMSolver + MeshConvergenceStudy), so none of the above
numbers feed into anything computed here -- they only need to be valid
enough for Bearing.assemble() to accept them and for bearing_extent() to
size the mesh-refinement exclusion window from `b`.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
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
from axisforge.solvers.mesh.convergence_solver import MeshConvergenceStudy

# ===========================================================================
# PARAMETERS
# ===========================================================================

B_STUDY    = 20.0          # single face width for the convergence study [mm]
P_W        = 1000.0
RPM_IN     = 450.0
r_pulley   = 0.050         # [m]

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")


GCI_THRESHOLD  = 0.01      # 1%
SAFETY_FACTOR  = 1.25
MAX_LEVELS     = 8

SHAFT_NAMES = ["shaft1(motor)", "shaft2", "shaft3(pulley)"]

# ---------------------------------------------------------------------------
# Bearing catalog + internal geometry -- see module docstring for what was
# carried over unchanged (N204_CATALOG) vs newly borrowed from
# design_double_row_life_verification.py (N204_GEOM, DGBB6204_CATALOG,
# DGBB6204_GEOM).
# ---------------------------------------------------------------------------

# N204 (floating, CylindricalRollerFamily) -- catalog numbers unchanged from
# the old flat N204() factory.
N204_CATALOG = dict(d=20.0, D=47.0, b=14.0, C=1000, C0=19_000.0, designation="N204")
# Internal roller geometry -- did not exist on the old flat Bearing; borrowed
# from design_double_row_life_verification.py's CR_GEOM (same 204-series
# envelope, d=20/D=47).
N204_GEOM = dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=60)

# DGBB6204 (locating, DeepGrooveBallFamily) -- new addition, replacing the
# N204(locating=True) call the CylindricalRollerFamily no longer accepts.
# Standard SKF 6204 dynamic rating C=12_700 N; C0 illustrative, both borrowed
# from design_double_row_life_verification.py's BB_GEOM / CR_6204.
DGBB6204_CATALOG = dict(d=20.0, D=47.0, b=14.0, C=12_700.0, C0=6_550.0, designation="6204")
DGBB6204_GEOM = dict(Dw=7.94, Dpw=33.5, Z=8, E=206_000.0, nu=0.3, s=0.010)

# ===========================================================================
# BUILD (same as design_load_lifting_bsweep.py, b fixed)
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


def N204(position, label) -> Bearing:
    """
    N-type cylindrical roller bearing -- floating only. CylindricalRollerFamily
    rejects arrangement="locating" (no flange to react axial load), so unlike
    the old flat N204(locating=...) factory, this one has no `locating` flag
    at all -- the locating position on the shaft is now DGBB6204() instead.
    """
    return Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(
            arrangement="floating",
            label=label,
            position=position,
            **N204_CATALOG,
        ),
        geometry=N204_GEOM,
        analyses={"line_contact": True},
    )


def DGBB6204(position, label) -> Bearing:
    """Deep-groove ball bearing -- the locating position on each shaft."""
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(
            arrangement="locating",
            label=label,
            position=position,
            **DGBB6204_CATALOG,
        ),
        geometry=DGBB6204_GEOM,
        analyses={"point_contact": True},
    )




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

    sys1.add_bearing(DGBB6204(20.0,  "brg1a")); sys1.add_bearing(N204(100.0, "brg1b"))
    sys2.add_bearing(DGBB6204(25.0,  "brg2a")); sys2.add_bearing(N204(125.0, "brg2b"))
    sys3.add_bearing(DGBB6204(25.0,  "brg3a")); sys3.add_bearing(N204(125.0, "brg3b"))

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
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1,
                    source_position=(0.0, 0.0))

    T3     = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))


    return {sh.name: sh for sh in (sys1, sys2, sys3)}


# ===========================================================================
# SOLVE GLOBAL + CONVERGENCE STUDY
# ===========================================================================

print(f"Building system (b = {B_STUDY} mm) ...")
shaft_systems = build_systems(B_STUDY)

global_solvers: dict[str, SimpleFEMSolver] = {}
for name, sys in shaft_systems.items():
    print(f"  Global solve: {name}")
    slv = SimpleFEMSolver(theory="timoshenko", distribute_gear_labels={"*"})
    slv.solve(sys)
    global_solvers[name] = slv

# per-shaft convergence study
conv_results: dict[str, object] = {}
skipped_all:  dict[str, list[str]] = {}

for name, sys in shaft_systems.items():
    print(f"\nConvergence study: {name}")
    intervals, skipped = MeshConvergenceStudy.intervals_from_shaft_system(sys)
    skipped_all[name] = skipped

    if skipped:
        print(f"  Skipped elements:")
        for msg in skipped:
            print(f"    [SKIP] {msg}")

    # excluir rolamentos -- deslocamento prescrito, refinamento sem sentido fisico
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
        print(f"  No intervals -- nothing to study.")
        conv_results[name] = None
        continue

    study = MeshConvergenceStudy(
        global_solver  = global_solvers[name],
        gci_threshold  = GCI_THRESHOLD,
        safety_factor  = SAFETY_FACTOR,
        max_levels     = MAX_LEVELS,
    )
    result = study.run(sys, intervals)
    conv_results[name] = result
    result.print_report()




# ===========================================================================
# PLOT -- convergence study: 3 subplots per interval (xz, xy, resultant)
# ===========================================================================

for shaft_name, result in conv_results.items():
    if result is None:
        continue

    records = list(result.per_load.values())
    if not records:
        continue

    n_intervals = len(records)
    fig = plt.figure(figsize=(14, 4 * n_intervals))
    fig.suptitle(
        f"Mesh convergence -- {shaft_name}  (b = {B_STUDY} mm)",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(n_intervals, 3, figure=fig, hspace=0.6, wspace=0.35)

    for row, rec in enumerate(records):
        levels = list(range(len(rec.levels)))

        # point_metrics_history: list of (f_xz, f_xy, f_res)
        m_xz = [m[0] for m in rec.point_metrics_history]
        m_xy = [m[1] for m in rec.point_metrics_history]
        m_res= [m[2] for m in rec.point_metrics_history]

        for col, (metrics, plane, color) in enumerate([
            (m_xz, "XZ", "#2166ac"),
            (m_xy, "XY", "#d62728"),
            (m_res,"resultant", "#1a1a1a"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(levels, metrics, marker="o", color=color,
                    lw=1.5, ms=5, label=f"v_{plane} [mm]")

            # GCI annotations
            for lvl_idx, gci_dict in enumerate(rec.gci_history):
                level_fine = lvl_idx + 2
                key = {"XZ": "xz", "XY": "xy", "resultant": "res"}[plane]
                gci_obj = gci_dict[key]
                if hasattr(gci_obj, "GCI_f_m") and not np.isnan(gci_obj.GCI_f_m):
                    gci_pct = gci_obj.GCI_f_m * 100.0
                    ax.annotate(
                        f"GCI={gci_pct:.3f}%",
                        xy=(level_fine, metrics[level_fine]),
                        xytext=(level_fine + 0.05, metrics[level_fine]),
                        fontsize=6,
                        color="#d62728" if gci_obj.GCI_f_m > GCI_THRESHOLD else "#33a02c",
                        va="center",
                    )

            ax.axhline(metrics[-1], color="0.6", lw=0.7, ls="--")

            status_color = "#33a02c" if rec.converged else "#d62728"
            status_txt   = "converged" if rec.converged else "NOT converged"
            ax.set_title(
                f"[{rec.label}] {plane}  x=[{rec.x_lo:.1f},{rec.x_hi:.1f}]  {status_txt}",
                fontsize=8, color=status_color,
            )
            ax.set_xlabel("refinement level", fontsize=7)
            ax.set_ylabel("v [mm]", fontsize=7)
            ax.set_xticks(levels)
            ax.set_xticklabels([f"L{l}" for l in levels], fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2, lw=0.4, color="0.7")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=6, frameon=False)

    plt.tight_layout()


# ===========================================================================
# PLOT -- global deflection diagrams per shaft
# ===========================================================================

from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    ShaftResultsReader, SimpleFEMResultsLibrary,
)

deflection_library = SimpleFEMResultsLibrary()

for shaft_name, sys in shaft_systems.items():
    slv = global_solvers[shaft_name]
    ShaftResultsReader(slv, sys).read(deflection_library)
    results = deflection_library.get(shaft_name)

    x    = results.x
    v_xz = results.v_xz    # [um]
    v_xy = results.v_xy
    v    = results.v

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Deflection diagrams -- {shaft_name}  (b = {B_STUDY} mm)",
                 fontsize=11, fontweight="bold")

    for ax, data, label, color in [
        (axes[0], v_xz, "v_xz [um]", "#2166ac"),
        (axes[1], v_xy, "v_xy [um]", "#d62728"),
        (axes[2], v,    "v resultant [um]", "#1a1a1a"),
    ]:
        ax.plot(x, data, color=color, lw=1.5, label=label)
        ax.axhline(0, color="0.7", lw=0.5)

        # bearing positions
        for b in sys.bearings:
            ax.axvline(b.position, color="0.5", lw=0.8, ls=":",
                       label="bearing" if b == sys.bearings[0] else "")

        # gear extents
        for ge in sys.gears:
            lo, hi = sys.gear_extent(ge)
            ax.axvspan(lo, hi, alpha=0.08, color="#33a02c",
                       label="gear extent" if ge == sys.gears[0] else "")
            ax.axvline(ge.position, color="#33a02c", lw=0.8, ls="-.")

        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.2, lw=0.4, color="0.7")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    axes[-1].set_xlabel("x [mm]", fontsize=8)
    plt.tight_layout()

plt.show()