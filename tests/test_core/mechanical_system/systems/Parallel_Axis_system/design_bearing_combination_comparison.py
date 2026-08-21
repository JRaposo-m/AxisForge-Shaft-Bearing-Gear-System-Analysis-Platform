"""
design_bearing_combination_comparison.py

Compares bearing-pair COMBINATIONS on the same load-lifting gearbox topology
(same shaft/gear pipeline as design_load_distribution_mixed_bearings.py):
loops over SCENARIOS swapping the locating bearing (single-row 6204 vs a
double-row 6204-pair), floating side is always NU204 -- isolates the effect
of the locating bearing alone.

Duplicated from design_load_distribution_mixed_bearings.py rather than
edited in place, per fixtures/README.md ("do not modify for a specific
case -- duplicate and rename").

Output: one comparison table per shaft, grouped by scenario.

Changelog (most recent first):
  - synced to the BallBearingResult/RollerBearingResult unification:
    REQUIRED_ATTRS (was _REQUIRED_ATTRS), families.ball/families.roller
    import paths, mr_result.outer_ok/.outer_n_iter/.outer_residual (was
    .ok/.n_iter/.residual), _contact_distribution() wraps a ball row in
    BallBearingResult(rows=[row]) before calling contact_distribution()
    (roller's own contact_distribution() still takes the raw row directly
    -- see roller_bearing_postprocessing.py's module docstring for why
    that side stays row-level).
  - capacity / Q_ei/Q_ee / Ka now computed via RollingBearingSolver.
    postprocess_and_record() for both single-row and double-row bearings,
    instead of ad hoc calls (_stiffness() removed). Two behavior changes:
    double-row Ka is now a real value (was hardcoded None), and double-row
    Q_ci/Q_ce is now a single whole-bearing value (was a per-row list of
    2, numerically identical). One manual step remains: capacity for the
    double-row label is patched in afterward via the real DR6204()
    Bearing, since the duck-typed multi-row view has no .family.
  - double-row scenario solved through RollingBearingSolver's own
    multi-row dispatch (_make_dr_multirow_view()) instead of a separate
    hand-rolled ISO16281MultiRowBallSolver call -- one solve() per shaft.
  - RollingBearingSolver.solve() returns a LIST per label (Option A) --
    every value indexed [0] for a single-row bearing.
  - Q_ci/Q_ce switched from bearing.family.dynamic_capacity() (whole-
    bearing Ca) to per_element_dynamic_capacity() (Q_ci/Q_ce) -- the
    correct per-element pairing for Q_ei/Q_ee.
  - Q_ei/Q_ee added, per row, via multirow_dynamic_equivalent_load()
    (generalized to any row count i, not hardcoded to 2).
  - illustrative AxialLoad on shaft3 removed -- was silently giving
    shaft3 Fa != 0 even in the "baseline" scenario.
  - thrust single-row scenario removed; the multi-row scenario switched
    from thrust (row-0-only probe) to a real 2-row radial 6204 double-row
    bearing, solved for real via ISO16281MultiRowBallSolver.
  - comparison table regrouped: scenario outer, shaft/slot inner.

Known gaps (not implemented here):
  - roller (line-contact) Q_ei/Q_ee -- LaminaDynamicEquivalentLoad exists
    now (roller_bearing_postprocessing.py), but it's per-lamina (q_kei/
    q_kee arrays, not a scalar per row like ball's Q_ei/Q_ee), so it
    doesn't fit this table's per-row-scalar format as-is -- left None,
    not wired in.
  - double-row "global" stiffness beyond row-0's representative Ka.
  - L10 life (P = X*Fr + Y*Fa) -- out of scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.deep_groove_ball import DeepGrooveBallFamily
from axisforge.core.machine_elements.Bearings.families.roller_bearing.radial.subtypes.cylindrical_roller import CylindricalRollerFamily
from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.loads import RadialLoad, TorqueLoad
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    GearElement, ShaftSystem,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import (
    SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    ShaftResultsReader,
    SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_solver import (
    ISO16281BallSolver,
    REQUIRED_ATTRS,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
    BallBearingResult,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import (
    RollerBearingResult,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    BearingResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_postprocessing import (
    contact_distribution as ball_contact_distribution,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_postprocessing import (
    contact_distribution as roller_contact_distribution,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.rolling_bearing_solver import (
    RollingBearingSolver,
)
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_grade import Grader


# ===========================================================================
# PARAMETERS
# ===========================================================================

B_STUDY  = 20.0
P_W      = 1000.0
RPM_IN   = 450.0
r_pulley = 0.050          # [m]

GEAR_KW_BASE = dict(mn=2.0, x=0.0, alpha_n_deg=20.0, Ra=0.8, material_id="42CrMo4")

SOLVER_TOL = 1.0e-6
GEAR_GRADE = "grade_3"

# Every locating bearing here is shaft-mounted (inner ring rotates, outer
# stationary) -- one fixed pair of flags for every bearing/row.
INNER_ROTATING = True
OUTER_ROTATING = False

# --- radial deep-groove ball, 6204 (d=20, D=47, B=14) ---
BB_GEOM = dict(Dw=7.94, Dpw=33.5, Z=8, s=0.010, E=206_000.0, nu=0.3)
CR_6204 = 12_700.0

# --- floating cylindrical roller, NU204 ---
CR_GEOM = dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40)
CR_NU204 = 28_500.0

# --- double-row deep-groove ball -- illustrative placeholder, 2 rows of the
# same geometry as BB_GEOM, tested against this system's real Fr.
DR_ROW_GEOM = dict(Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
                   E=BB_GEOM["E"], nu=BB_GEOM["nu"], s=BB_GEOM["s"])
DR_CATALOG = dict(d=20.0, D=47.0, b=28.0, C=CR_6204, C0=6_550.0,
                  designation="DR6204-illustrative")


# ===========================================================================
# BEARING BUILDERS -- one per bearing kind, each returning an assembled
# Bearing via Bearing.assemble(). Locating builders are swapped per
# scenario; CRNU204 (floating) is constant across every scenario.
# ===========================================================================

def BB6204(position, label) -> Bearing:
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(d=20.0, D=47.0, designation="6204", b=14.0,
                               C=CR_6204, C0=6_550.0, arrangement="locating",
                               label=label, position=position),
        geometry=dict(Dw=BB_GEOM["Dw"], Dpw=BB_GEOM["Dpw"], Z=BB_GEOM["Z"],
                      E=BB_GEOM["E"], nu=BB_GEOM["nu"], s=BB_GEOM["s"]),
        analyses={"point_contact": True},
    )


def CRNU204(position, label) -> Bearing:
    return Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=BearingCatalog(d=20.0, D=47.0, designation="NU204", b=14.0,
                               C=CR_NU204, C0=22_000.0, arrangement="floating",
                               label=label, position=position),
        geometry=dict(Dwe=CR_GEOM["Dwe"], Lwe=CR_GEOM["Lwe"], Dpw=CR_GEOM["Dpw"],
                      Z=CR_GEOM["Z"], s=CR_GEOM["s"], n_s=CR_GEOM["n_s"]),
        analyses={"line_contact": True},
    )


# DR_ROW_BEARINGS -- the 2 rows' full Bearing objects (position-independent,
# built once), so per_element_dynamic_capacity() can be called on each row
# directly rather than guessed at via a stripped SimpleNamespace.
def _row_attrs(b: Bearing) -> dict:
    return {a: getattr(b, a) for a in REQUIRED_ATTRS}


DR_ROW_BEARINGS = [
    Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(arrangement="locating", label=f"dr-row{j}",
                               position=0.0, **DR_CATALOG),
        geometry=DR_ROW_GEOM,
        analyses={"point_contact": True},
    )
    for j in range(2)
]

# DR_ROWS -- the 2 rows' REQUIRED_ATTRS dicts for ISO16281MultiRowBallSolver.
DR_ROWS = [_row_attrs(b) for b in DR_ROW_BEARINGS]


def DR6204(position, label) -> Bearing:
    """
    Double-row 6204's FEM-facing Bearing -- single-row-shaped (one node;
    FEM has no notion of rows). The solver does NOT see this object for the
    double-row scenario's locating slot -- it's handed a separate duck-typed
    multi-row view instead, see _make_dr_multirow_view().
    """
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(arrangement="locating", label=label,
                               position=position, **DR_CATALOG),
        geometry=DR_ROW_GEOM,
        analyses={"point_contact": True},
    )


# ===========================================================================
# SCENARIOS -- (name, locating_builder). Floating side is always CRNU204.
# ===========================================================================

DOUBLE_ROW_SCENARIO = "radial DOUBLE-row locating (2x 6204-row) + floating roller (NU204)  [REAL multi-row solve]"

SCENARIOS = [
    ("radial single-row locating (6204) + floating roller (NU204)  [baseline]", BB6204),
    (DOUBLE_ROW_SCENARIO,                                                       DR6204),
]


# ===========================================================================
# BUILD HELPERS
# ===========================================================================

def make_stepped_shaft(name, total_length, d_seat, d_body,
                       l_seat_a, l_seat_b, fillet_r, material_id="AISI_1045"):
    l_body = total_length - l_seat_a - l_seat_b
    sh = Shaft(label=name)
    sh.add_section(ShaftSection(length=l_seat_a, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatA"))
    sh.add_section(ShaftSection(
        length=l_body, diameter=d_body, material_id=material_id, label=f"{name}-body",
        shoulder_left=Shoulder(fillet_radius=fillet_r, diameter_large=d_body, diameter_small=d_seat),
        shoulder_right=Shoulder(fillet_radius=fillet_r, diameter_large=d_body, diameter_small=d_seat),
    ))
    sh.add_section(ShaftSection(length=l_seat_b, diameter=d_seat,
                                material_id=material_id, label=f"{name}-seatB"))
    return sh


def build_systems(b: float, locating_builder) -> dict[str, ShaftSystem]:
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
    sys3 = ShaftSystem(s3, name="shaft3(pulley)", speed_rpm=50.0)

    sys1.shaft_origin_x = 0.0
    sys2.shaft_origin_x = sys1.shaft_origin_x + (z1.position - z2.position)
    sys3.shaft_origin_x = sys2.shaft_origin_x + (z3.position - z4.position)

    sys1.add_bearing(locating_builder(20.0,  "brg1a")); sys1.add_bearing(CRNU204(100.0, "brg1b"))
    sys2.add_bearing(locating_builder(25.0,  "brg2a")); sys2.add_bearing(CRNU204(125.0, "brg2b"))
    sys3.add_bearing(locating_builder(25.0,  "brg3a")); sys3.add_bearing(CRNU204(125.0, "brg3b"))

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
                                    label="load-lifting-bearing-comparison")
    gearbox.resolve(P_W, RPM_IN, rotation_dir_source=1, source_position=(0.0, 0.0))

    T3      = gearbox._resolved[id(sys3)]["T_out_Nm"]
    F_rope  = T3 / r_pulley
    sys3.add_load(RadialLoad(95.0, F_rope, theta_deg=180.0, label="rope-tension"))
    # Fa=0 on every shaft here, same as design_load_distribution_mixed_bearings.py.

    T_z1_in = gearbox._resolved[id(sys1)]["T_out_Nm"]
    sys1.add_load(TorqueLoad(0.0,  -T_z1_in, source="user", label="motor-input"))
    sys3.add_load(TorqueLoad(95.0,  T3,      source="user", label="pulley-load-resistance"))

    return {sh.name: sh for sh in (sys1, sys2, sys3)}


def gear_grade_nodes(shaft_sys: ShaftSystem, grade: str = GEAR_GRADE) -> list[float]:
    base_nodes = Mesh1D(shaft_sys).x_nodes
    extra: list[float] = []
    for ge in shaft_sys.gears:
        lo, hi = shaft_sys.gear_extent(ge)
        extra += Grader(lo, hi, base_nodes).get_grade(grade)
    return extra


def _contact_distribution(b_obj, res):
    """
    res is bundle.load_distribution[0] -- a bare per-row result. Both
    ball's and roller's contact_distribution() now take the
    BallBearingResult/RollerBearingResult wrapper and return a list (one
    per row) -- wrap res in a length-1 container and unwrap [0] either way.
    """
    if b_obj.bearing_type == BearingType.CYLINDRICAL_ROLLER:
        return roller_contact_distribution(b_obj, RollerBearingResult(rows=[res]))[0]
    return ball_contact_distribution(b_obj, BallBearingResult(rows=[res]))[0]


def _capacity_Cr(b_obj: Bearing) -> float:
    """Cr forwarded to per_element_dynamic_capacity() -- CR_6204 or CR_NU204 by bearing type."""
    return CR_NU204 if b_obj.bearing_type == BearingType.CYLINDRICAL_ROLLER else CR_6204


def _per_element_capacity(b_obj: Bearing) -> tuple[float, float]:
    """(Q_ci, Q_ce) -- per-element dynamic capacity, ISO/TS 16281 S4.3.1.2, uniform for ball and roller."""
    return b_obj.family.per_element_dynamic_capacity(b_obj, Cr=_capacity_Cr(b_obj))


def _make_dr_multirow_view(label: str) -> SimpleNamespace:
    """
    Duck-typed multi-row stand-in for the double-row scenario's locating
    slot -- used ONLY for RollingBearingSolver (.rows/.i/.bearing_type/
    .arrangement/.label are all it reads). The FEM still sees the real
    single-row DR6204() Bearing.
    """
    return SimpleNamespace(
        rows=DR_ROWS, i=len(DR_ROWS), label=label,
        bearing_type=BearingType.DEEP_GROOVE_BALL, arrangement="locating",
    )


def _format_double_row_result(node, bundle, mr_result) -> dict:
    """
    Formats an already-solved, already-postprocessed double-row bundle into
    the same dict shape the print loop / plots expect. Pure report
    formatting: capacity/Q_ei/Q_ee/Ka come from `bundle` (populated by
    RollingBearingSolver.postprocess_and_record()); only Q_j/phi_global/
    Q_max/n_loaded/dist_traces (this script's own table/plot fields) are
    computed here. `mr_result` (bundle.extra["multirow_result"]) is now a
    BallBearingResult (via .multirow()) -- its outer-solve diagnostics are
    named outer_ok/outer_n_iter/outer_residual, not ok/n_iter/residual.
    """
    row_results = bundle.load_distribution
    row_eq      = bundle.dynamic_equivalent_load

    # bundle.capacity is a single (Q_ci, Q_ce) pair (see library.py) --
    # wrapped in a length-1 list for shape symmetry with the single-row
    # convention. Not a numeric change: DR_ROW_BEARINGS and DR6204() share
    # identical geometry by construction.
    Q_ci, Q_ce = bundle.capacity
    Q_ci, Q_ce = [Q_ci], [Q_ce]

    phi_Fr = row_results[0].phi_Fr

    Q_max       = 0.0
    n_loaded    = 0
    Z_total     = 0
    dist_traces = []
    for j, (row_geom, row_res) in enumerate(zip(DR_ROWS, row_results)):
        Q_j         = row_geom["cp"] * row_res.delta_j ** 1.5
        phi_global  = (row_geom["phi_j"] + phi_Fr) % (2.0 * np.pi)
        Q_max       = max(Q_max, float(Q_j.max()))
        n_loaded   += int((Q_j > 0).sum())
        Z_total    += int(row_geom["Z"])
        dist_traces.append((phi_global, Q_j, f"row{j}"))

    return dict(
        kind="deep_groove_ball_double_row [REAL 2-row solve, via RollingBearingSolver]",
        Fr=node.Fr, Fa=node.Fa,
        Q_max=Q_max, n_loaded=n_loaded, Z=Z_total,
        Ka=bundle.stiffness.Ka, ok=mr_result.outer_ok,
        delta_r=mr_result.delta_r, delta_a=mr_result.delta_a, psi=mr_result.psi,
        dist_traces=dist_traces,
        # outer (multi-row) load-split convergence diagnostics -- None for
        # an ordinary single-row bearing (no outer solve to report).
        mr_n_iter=mr_result.outer_n_iter, mr_residual=mr_result.outer_residual,
        mr_f_r=mr_result.f_r, mr_f_a=mr_result.f_a,
        Q_ei=[e.Q_ei for e in row_eq], Q_ee=[e.Q_ee for e in row_eq],
        Q_ci=Q_ci, Q_ce=Q_ce,
    )


# ===========================================================================
# PLOTS -- read the SAME `rows` dict the printed table already built, no
# re-solving.
#   plot_polar_comparison() -- one figure per shaft, (slot x scenario) grid
#     of polar Q_j-vs-phi_global plots; double-row's locating subplot shows
#     both rows' traces overlaid.
#   plot_qmax_comparison() -- grouped bar chart, Q_max per scenario, one
#     subplot per shaft's locating bearing.
# ===========================================================================

def plot_polar_comparison(rows: dict[tuple[str, str], dict[str, dict]]) -> None:
    shaft_names = sorted({shaft_name for shaft_name, _ in rows})
    slots       = ["locating (a)", "floating (b)"]
    scenario_names = [s for s, _ in SCENARIOS]

    for shaft_name in shaft_names:
        fig, axes = plt.subplots(len(slots), len(scenario_names),
                                 figsize=(5 * len(scenario_names), 5 * len(slots)),
                                 subplot_kw={"projection": "polar"})
        fig.suptitle(f"Load distribution comparison -- {shaft_name}  (ISO/TS 16281)",
                    fontsize=11, fontweight="bold")

        for i, slot in enumerate(slots):
            by_scenario = rows.get((shaft_name, slot), {})
            for j, scenario_name in enumerate(scenario_names):
                ax = axes[i, j] if len(slots) > 1 else axes[j]
                r  = by_scenario.get(scenario_name)
                if r is None:
                    ax.set_visible(False)
                    continue

                colors = ["#2166ac", "#d62728"]   # row0, row1 (if 2 rows)
                for k, (phi, Q, row_label) in enumerate(r["dist_traces"]):
                    phi_c = np.append(phi, phi[0])
                    Q_c   = np.append(Q,   Q[0])
                    color = colors[k % len(colors)]
                    lbl   = row_label if row_label is not None else None
                    ax.plot(phi_c, Q_c, color=color, lw=1.4, marker="o", ms=3, label=lbl)
                    ax.fill(phi_c, Q_c, color=color, alpha=0.10)

                if any(lbl for _, _, lbl in r["dist_traces"]):
                    ax.legend(fontsize=6, frameon=False, loc="upper right")

                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
                ax.tick_params(labelsize=6)
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                short = scenario_name.split("(")[0].strip()
                ax.set_title(f"{short}\n{slot}\nQ_max={r['Q_max']:.0f} N  "
                            f"n_loaded={r['n_loaded']}/{r['Z']}",
                            fontsize=7)

        plt.tight_layout()
        plt.show()


def plot_qmax_comparison(rows: dict[tuple[str, str], dict[str, dict]]) -> None:
    shaft_names = sorted({shaft_name for shaft_name, slot in rows if slot == "locating (a)"})
    scenario_names = [s for s, _ in SCENARIOS]
    short_names = [s.split("(")[0].strip() for s in scenario_names]

    fig, axes = plt.subplots(1, len(shaft_names), figsize=(5 * len(shaft_names), 4.5))
    if len(shaft_names) == 1:
        axes = [axes]
    fig.suptitle("Q_max comparison -- locating bearing, by scenario", fontsize=11, fontweight="bold")

    for ax, shaft_name in zip(axes, shaft_names):
        by_scenario = rows.get((shaft_name, "locating (a)"), {})
        vals  = [by_scenario[s]["Q_max"] if s in by_scenario else 0.0 for s in scenario_names]
        bars  = ax.bar(short_names, vals, color=["#4c72b0", "#c44e52", "#55a868"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}",
                   ha="center", va="bottom", fontsize=8)
        ax.set_title(shaft_name, fontsize=9)
        ax.set_ylabel("Q_max [N]", fontsize=8)
        ax.tick_params(labelsize=7, axis="x", rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2, lw=0.4, axis="y")

    plt.tight_layout()
    plt.show()


# ===========================================================================
# MAIN -- one comparison table per shaft, scenarios as columns
# ===========================================================================

W = 100

if __name__ == "__main__":
    library = SimpleFEMResultsLibrary()

    # rows[(shaft_name, bearing_slot)] = {scenario_name: row_dict}
    rows: dict[tuple[str, str], dict[str, dict]] = {}

    for scenario_name, locating_builder in SCENARIOS:
        print(f"\n=== scenario: {scenario_name} ===")

        shaft_systems = build_systems(B_STUDY, locating_builder)

        for name, shaft_sys in shaft_systems.items():
            bearings    = {b.label: b for b in shaft_sys.bearings}
            extra_nodes = gear_grade_nodes(shaft_sys)

            fem = SimpleFEMSolver()
            fem.solve(shaft_sys, extra_mandatory=extra_nodes)
            ShaftResultsReader(fem, shaft_sys).read(library)

            # Solver-facing bearings dict -- same as `bearings` except the
            # double-row scenario's locating label is swapped for a
            # duck-typed multi-row view, so RollingBearingSolver's own
            # multi-row dispatch picks it up automatically.
            bearings_for_solver = dict(bearings)
            dr_label = None
            if scenario_name == DOUBLE_ROW_SCENARIO:
                dr_label = next(lbl for lbl in bearings if lbl.endswith("a"))
                bearings_for_solver[dr_label] = _make_dr_multirow_view(dr_label)

            # catalog -- per-bearing kwargs for postprocess_and_record().
            # "capacity" is omitted for dr_label: the duck-typed multi-row
            # view has no .family, so it's patched in manually afterward
            # using the real DR6204() Bearing instead. "dynamic_equivalent_
            # load" is omitted for the roller (see "Known gaps" above).
            catalog: dict[str, dict] = {}
            for lbl, b_real in bearings.items():
                entry: dict = {}
                if lbl != dr_label:
                    entry["capacity"] = {"Cr": _capacity_Cr(b_real)}
                if b_real.bearing_type != BearingType.CYLINDRICAL_ROLLER:
                    entry["dynamic_equivalent_load"] = dict(
                        inner_rotating=INNER_ROTATING, outer_rotating=OUTER_ROTATING,
                    )
                catalog[lbl] = entry

            # ONE call per shaft: solves (internally, since load_distribution
            # is not pre-supplied) AND records capacity / dynamic_equivalent_
            # load / stiffness for every bearing on the shaft.
            results_lib = BearingResultsLibrary()
            solver      = RollingBearingSolver(tol=SOLVER_TOL)
            results_lib = solver.postprocess_and_record(
                shaft_sys, bearings_for_solver, library, catalog,
                results=results_lib,
            )

            if dr_label is not None:
                # Manual capacity patch, via the real DR6204() Bearing
                # (bearings[dr_label], not the duck-typed solver view).
                results_lib.set_capacity(
                    dr_label, _per_element_capacity(bearings[dr_label]),
                )

            shaft_res     = library.get(name)
            node_by_label = {n.label: n for n in shaft_res.bearing_nodes}

            for lbl in bearings:
                node   = node_by_label[lbl]
                b_obj  = bearings[lbl]
                slot   = "locating (a)" if lbl.endswith("a") else "floating (b)"
                key    = (name, slot)
                bundle = results_lib.get(lbl)

                if scenario_name == DOUBLE_ROW_SCENARIO and slot == "locating (a)":
                    mr_result = bundle.extra["multirow_result"]
                    rows.setdefault(key, {})[scenario_name] = _format_double_row_result(
                        node, bundle, mr_result,
                    )
                    continue

                res      = bundle.load_distribution[0]
                dist     = _contact_distribution(b_obj, res)
                Q_arr    = dist[:, 1]
                n_loaded = int((Q_arr > 0).sum())
                Q_max    = float(Q_arr.max())
                stiff    = bundle.stiffness

                Q_ci, Q_ce = bundle.capacity
                Q_ci, Q_ce = [Q_ci], [Q_ce]

                if bundle.dynamic_equivalent_load is not None:
                    eq = bundle.dynamic_equivalent_load[0]
                    Q_ei, Q_ee = [eq.Q_ei], [eq.Q_ee]
                else:
                    Q_ei, Q_ee = None, None

                rows.setdefault(key, {})[scenario_name] = dict(
                    kind=b_obj.family.name,
                    Fr=node.Fr, Fa=node.Fa,
                    Q_max=Q_max, n_loaded=n_loaded, Z=b_obj.Z,
                    Ka=stiff.Ka, ok=res.ok,
                    delta_r=res.delta_r, delta_a=res.delta_a, psi=res.psi,
                    dist_traces=[(dist[:, 0], dist[:, 1], None)],
                    mr_n_iter=None, mr_residual=None, mr_f_r=None, mr_f_a=None,
                    Q_ei=Q_ei, Q_ee=Q_ee,
                    Q_ci=Q_ci, Q_ce=Q_ce,
                )

    # comparison table -- grouped by SCENARIO (outer), shaft/slot (inner)
    print(f"\n{'='*W}")
    print("  BEARING COMBINATION COMPARISON")
    print(f"{'='*W}")

    shaft_slot_keys = sorted(rows.keys())

    for scenario_name, _ in SCENARIOS:
        print(f"\n{'='*W}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"{'='*W}")

        for shaft_name, slot in shaft_slot_keys:
            r = rows[(shaft_name, slot)].get(scenario_name)
            if r is None:
                continue
            if r["Ka"] is None:
                ka_txt = "n/a"
            elif r["Ka"] == float("inf"):
                ka_txt = "inf"
            else:
                ka_txt = f"{r['Ka']:.3e}"
            print(f"\n-- {shaft_name} / {slot} {'-'*max(W-14-len(shaft_name)-len(slot), 4)}")
            print(f"      kind={r['kind']:<28}  ok={r['ok']}")
            print(f"      Fr={r['Fr']:>9.1f} N   Fa={r['Fa']:>9.1f} N   "
                  f"Q_max={r['Q_max']:>9.1f} N   n_loaded={r['n_loaded']}/{r['Z']}")
            print(f"      delta_r={r['delta_r']:>10.5f} mm   delta_a={r['delta_a']:>10.5f} mm   "
                  f"psi={np.degrees(r['psi']):>9.5f} deg")
            print(f"      Ka={ka_txt}")

            if r["mr_n_iter"] is not None:
                f_r_txt = ", ".join(f"{v:.4f}" for v in r["mr_f_r"])
                f_a_txt = ", ".join(f"{v:.4f}" for v in r["mr_f_a"])
                print(f"      [multirow outer solve]  ok={r['ok']}   "
                      f"n_iter={r['mr_n_iter']}   residual={r['mr_residual']:.3e}")
                print(f"          f_r (per row) = [{f_r_txt}]")
                print(f"          f_a (per row) = [{f_a_txt}]")

            qci_txt = ", ".join(f"{v:.1f}" for v in r["Q_ci"])
            qce_txt = ", ".join(f"{v:.1f}" for v in r["Q_ce"])
            print(f"      Q_ci (per row) = [{qci_txt}] N   "
                  f"Q_ce (per row) = [{qce_txt}] N")

            if r["Q_ei"] is not None:
                qei_txt = ", ".join(f"{v:.1f}" for v in r["Q_ei"])
                qee_txt = ", ".join(f"{v:.1f}" for v in r["Q_ee"])
                print(f"      Q_ei (per row) = [{qei_txt}] N   "
                      f"Q_ee (per row) = [{qee_txt}] N")

    plot_polar_comparison(rows)
    plot_qmax_comparison(rows)

    print(f"\n{'='*W}")
    print("  done -- see module docstring before trusting these numbers.")
    print(f"{'='*W}\n")