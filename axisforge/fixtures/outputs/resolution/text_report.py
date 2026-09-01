"""
fixtures/outputs/resolution/text_report.py

THE principal writer for the whole Resolution domain -- mirrors
fixtures/outputs/construction/text_report.py's own role, one level up:
Resolution has only one domain today (shaft_fem), so content-block
building and file-writing live together in this one module instead of
being split across per-domain subpackages the way Construction's four
domains are -- nothing to gain from that split yet with only one thing
to report. If a second Resolution domain is ever added, split then,
the same way Construction did.

Reads axisforge.solvers.machine_elements.shaft.oneD_analysis.static.
static_analysis.ShaftResults (via SimpleFEMResultsLibrary) -- its own
docstring documents FOUR sections:
  1. MESH                      -- x_nodes, elements
  2. FEM SOLUTION (raw)        -- K, d_total_xz/xy, f_xz/xy_ext/total/
                                   reaction, free/constrained_dofs,
                                   T_total, tau_total, torsion_contributions
  3. POST-PROCESSED QUANTITIES -- x, M_xz/xy/M, V_xz/xy/V, v_xz/xy/v, T,
                                   d, W, Wt, sigma_b, tau (all length n),
                                   plus the *_max/x_*_max governing
                                   values, plus the bearing REACTION
                                   summary arrays (bearing_positions,
                                   R_xz, R_xy, R, R_axial)
  4. BEARING NODE DATA          -- one BearingNodeData per bearing,
                                    complete FEM nodal state (displace-
                                    ments, rotations, forces, moments,
                                    seat misalignment)

This report includes sections 3 and 4 in full, plus x_nodes/elements
count from section 1. It deliberately does NOT print section 2's raw
solver arrays (K -- a (3n, 3n) matrix; the full-length d_total_xz/xy,
f_xz/xy_ext/total/reaction vectors; free/constrained_dofs index lists)
-- those are solver-internal state sized for programmatic downstream
consumers (ISO 16281, Newton-Raphson coupling), not something anyone
reads row-by-row in a text report; dumping a 3n x 3n matrix to a .txt
would not be legible at any n worth solving. T_total/tau_total ARE
covered regardless -- they are literally the same arrays as the
post-processed T/tau columns already printed (ShaftResultsReader.read()
sets `T = solver.T_total` and `tau = tau_arr = solver.tau_total`
verbatim, no copy). torsion_contributions (per-node breakdown of which
torque source contributes how much at each node) is also not
included -- a nested list[list[dict]], genuinely useful for debugging a
torsion diagram but not yet given a table shape here; flag if wanted
and it gets its own block.

Per-node arrays (section 3) are split into three grouped tables rather
than one wide 16-column table, purely for legibility in a fixed-width
text viewer:
  BENDING & SHEAR       : x, M_xz, M_xy, M, V_xz, V_xy, V
  DEFLECTION & TORSION  : x, v_xz, v_xy, v, T
  SECTION & STRESS      : x, d, W, Wt, sigma_b, tau
Units follow ShaftResults' own docstring exactly, including the one
easy-to-miss mismatch: M_xz/M_xy/M are N.mm, but T is N.m -- kept
distinct in the column headers below, not normalised to one unit,
so a reader cross-checking against ShaftResults' own docstring sees
the same units in both places.

Bearing data (section 4) becomes TWO things, since ShaftResults itself
keeps them as two separate stored quantities computed from two
different solver arrays -- not merged into one, so as not to imply
they are the same measurement read twice:
  BEARING REACTIONS (summary table) -- bearing_positions/R_xz/R_xy/R/
    R_axial, derived from solver.f_xz_reaction/f_xy_reaction (reaction
    vector, non-zero only at constrained DOFs). No label array is
    stored alongside these in ShaftResults itself -- labelled here by
    reading bearing_nodes[i].label at the same index (both arrays are
    built from the same `for b in self._sys.bearings` loop, in the
    same reader instance, so index-aligned; see static_analysis.py's
    own ShaftResultsReader.read()).
  BEARING NODE DATA (two tables, one row per bearing) -- the full
    BearingNodeData per bearing_nodes entry, split the same way
    section 3's per-node arrays are split above (one 13-column row per
    bearing does not read well in a fixed-width text viewer):
      DISPLACEMENTS & ROTATIONS -- position, u, v_xz, v_xy, theta_xz,
        theta_xy, psi_xz, psi_xy (seat misalignment kept here, it is
        an angle like theta, not a load).
      LOADS (total force vector) -- position, Fr_xz, Fr_xy, Fr, Fa,
        M_xz, M_xy, derived from solver.f_xz_total/f_xy_total (TOTAL
        nodal force, not the reaction vector -- numerically close to
        the reactions table above at a bearing node with no other load
        applied exactly there, but not guaranteed identical, and
        computed from a different array).

Writes with encoding="utf-8" explicitly, same reason as every other
report writer in this package. No shared _io.py-style helper module --
same explicit preference already applied to fixtures/outputs/
construction/*: this writer owns its own RULE constant and its own
write() call.

Dependency (fixtures/solvers only, read-only access -- no core
modification):
  axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis
      SimpleFEMResultsLibrary, ShaftResults, BearingNodeData (not
      imported here -- this module only ever READS objects handed to
      it, it never builds them; see fixtures/solvers/fem_simple.py's
      own solve_system() for that)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

RULE = "=" * 72
SUB = "-" * 72

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
        SimpleFEMResultsLibrary, ShaftResults, BearingNodeData,
    )


# ---------------------------------------------------------------------------
# Table formatting -- small internal helper, no third-party dependency.
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    """
    Fixed-width ASCII table: `headers` and every row in `rows` are
    already-formatted strings (caller controls precision/units per
    column); this only pads columns to the widest cell (header or any
    row) and joins with 2-space gutters, plus a rule line under the
    header. Returns "(no data)" if `rows` is empty.
    """
    if not rows:
        return "  (no data)"
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    def fmt_row(cells: list[str]) -> str:
        return "  " + "  ".join(c.rjust(w) for c, w in zip(cells, widths))
    lines = [fmt_row(headers), "  " + "  ".join("-" * w for w in widths)]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

def summary_block(result: "ShaftResults") -> str:
    """Governing values + their axial location: M_max, v_max,
    sigma_b_max, tau_max. The four *_max/x_*_max scalar pairs
    ShaftResults itself stores (no V_max -- not a field on the class)."""
    return "\n".join([
        f"  M_max          : {result.M_max:12.2f} N.mm  @ x={result.x_M_max:8.2f} mm",
        f"  v_max          : {result.v_max:12.4f} mm    @ x={result.x_v_max:8.2f} mm",
        f"  sigma_b_max    : {result.sigma_b_max:12.2f} MPa   @ x={result.x_sigma_b_max:8.2f} mm",
        f"  tau_max        : {result.tau_max:12.2f} MPa   @ x={result.x_tau_max:8.2f} mm",
    ])


def bending_shear_table(result: "ShaftResults") -> str:
    """Per-node table: x, M_xz, M_xy, M [N.mm]; V_xz, V_xy, V [N]."""
    headers = ["x[mm]", "M_xz[N.mm]", "M_xy[N.mm]", "M[N.mm]",
               "V_xz[N]", "V_xy[N]", "V[N]"]
    rows = [
        [f"{x:.1f}", f"{mxz:.1f}", f"{mxy:.1f}", f"{m:.1f}",
         f"{vxz:.1f}", f"{vxy:.1f}", f"{v:.1f}"]
        for x, mxz, mxy, m, vxz, vxy, v in zip(
            result.x_nodes, result.M_xz, result.M_xy, result.M,
            result.V_xz, result.V_xy, result.V,
        )
    ]
    return _table(headers, rows)


def deflection_torsion_table(result: "ShaftResults") -> str:
    """Per-node table: v_xz, v_xy, v [mm]; T [N.m] -- note T's unit
    differs from the bending-moment table's N.mm, per ShaftResults'
    own docstring."""
    headers = ["x[mm]", "v_xz[mm]", "v_xy[mm]", "v[mm]", "T[N.m]"]
    rows = [
        [f"{x:.1f}", f"{vxz:.4f}", f"{vxy:.4f}", f"{v:.4f}", f"{t:.3f}"]
        for x, vxz, vxy, v, t in zip(
            result.x_nodes, result.v_xz, result.v_xy, result.v, result.T,
        )
    ]
    return _table(headers, rows)


def section_stress_table(result: "ShaftResults") -> str:
    """Per-node table: d [mm]; W, Wt [mm^3]; sigma_b, tau [MPa]."""
    headers = ["x[mm]", "d[mm]", "W[mm^3]", "Wt[mm^3]",
               "sigma_b[MPa]", "tau[MPa]"]
    rows = [
        [f"{x:.1f}", f"{d:.2f}", f"{w:.1f}", f"{wt:.1f}",
         f"{sb:.2f}", f"{t:.2f}"]
        for x, d, w, wt, sb, t in zip(
            result.x_nodes, result.d, result.W, result.Wt,
            result.sigma_b, result.tau,
        )
    ]
    return _table(headers, rows)


def bearing_reactions_table(result: "ShaftResults") -> str:
    """
    Per-bearing table from the REACTION-vector-derived summary arrays
    (bearing_positions, R_xz, R_xy, R, R_axial) -- labelled by reading
    bearing_nodes[i].label at the same index (see this module's own
    top docstring for why that's a safe, index-aligned lookup rather
    than a guess).
    """
    labels = [n.label for n in result.bearing_nodes]
    headers = ["label", "position[mm]", "R_xz[N]", "R_xy[N]", "R[N]", "R_axial[N]"]
    rows = []
    for i, pos in enumerate(result.bearing_positions):
        label = labels[i] if i < len(labels) else f"(bearing {i})"
        rows.append([
            label, f"{pos:.2f}",
            f"{result.R_xz[i]:.2f}", f"{result.R_xy[i]:.2f}",
            f"{result.R[i]:.2f}", f"{result.R_axial[i]:.2f}",
        ])
    return _table(headers, rows)


def bearing_node_kinematics_table(result: "ShaftResults") -> str:
    """
    Per-bearing table: position, displacements (u, v_xz, v_xy), shaft
    slope at the node (theta_xz, theta_xy), and seat misalignment
    angle (psi_xz, psi_xy) -- the "how does the shaft move/tilt at
    this bearing" half of BearingNodeData. Split from the loads table
    below for the same legibility reason bending/deflection/section
    are three separate tables above: one 13-column row per bearing
    does not read well in a fixed-width text viewer.
    """
    headers = ["label", "position[mm]", "u[mm]", "v_xz[mm]", "v_xy[mm]",
               "theta_xz[rad]", "theta_xy[rad]", "psi_xz[rad]", "psi_xy[rad]"]
    rows = [
        [n.label, f"{n.position:.2f}", f"{n.u:.4f}", f"{n.v_xz:.4f}", f"{n.v_xy:.4f}",
         f"{n.theta_xz:.6f}", f"{n.theta_xy:.6f}", f"{n.psi_xz:.6f}", f"{n.psi_xy:.6f}"]
        for n in result.bearing_nodes
    ]
    return _table(headers, rows)


def bearing_node_loads_table(result: "ShaftResults") -> str:
    """
    Per-bearing table: radial/axial force and bending moment at the
    bearing node, from the TOTAL force vector (solver.f_xz_total/
    f_xy_total) -- see this module's own top docstring for how this
    differs from bearing_reactions_table() above (REACTION vector,
    non-zero only at constrained DOFs; numerically close but not the
    same computed quantity).
    """
    headers = ["label", "position[mm]", "Fr_xz[N]", "Fr_xy[N]", "Fr[N]",
               "Fa[N]", "M_xz[N.mm]", "M_xy[N.mm]"]
    rows = [
        [n.label, f"{n.position:.2f}", f"{n.Fr_xz:.2f}", f"{n.Fr_xy:.2f}",
         f"{n.Fr:.2f}", f"{n.Fa:.2f}", f"{n.M_xz:.2f}", f"{n.M_xy:.2f}"]
        for n in result.bearing_nodes
    ]
    return _table(headers, rows)


def shaft_result_block(result: "ShaftResults") -> str:
    """Combines every content block above into one shaft's full
    Resolution section, in the order: summary, bending & shear,
    deflection & torsion, section & stress, bearing reactions, bearing
    node data."""
    sections = [
        f"  nodes          : {len(result.x_nodes)}",
        "",
        summary_block(result),
        "",
        "BENDING & SHEAR",
        SUB,
        bending_shear_table(result),
        "",
        "DEFLECTION & TORSION",
        SUB,
        deflection_torsion_table(result),
        "",
        "SECTION & STRESS",
        SUB,
        section_stress_table(result),
        "",
        "BEARING REACTIONS",
        SUB,
        bearing_reactions_table(result),
        "",
    ]
    if result.bearing_nodes:
        sections.append(f"BEARING NODE DATA ({len(result.bearing_nodes)}) -- DISPLACEMENTS & ROTATIONS")
        sections.append(SUB)
        sections.append(bearing_node_kinematics_table(result))
        sections.append("")
        sections.append(f"BEARING NODE DATA ({len(result.bearing_nodes)}) -- LOADS (total force vector)")
        sections.append(SUB)
        sections.append(bearing_node_loads_table(result))
    else:
        sections.append("BEARING NODE DATA: (none)")
    return "\n".join(sections)


def write_resolution_report(
    library: "SimpleFEMResultsLibrary",
    system: "SpurHelicalGearSystem",
    path: "str | Path",
    title: str = "",
) -> str:
    """
    Write ONE .txt covering the whole Resolution domain: one
    "SHAFT: <name>" section per shaft in system.shafts (system's own
    order), each holding shaft_result_block() for that shaft's result
    in `library`, or "(no result)" if library.get_or_none(name) is
    None. Returns the written text, so a caller that wants to
    inspect/verify it doesn't have to re-open the file it just wrote.

    Parameters
    ----------
    library : SimpleFEMResultsLibrary
        Already solved (e.g. via solve_system()). Purely reads
        `library` -- does not solve or validate it.
    system : SpurHelicalGearSystem
        Supplies the shaft ORDER and NAMES this report walks --
        library itself has no ordering guarantee beyond insertion, and
        a shaft system's own shaft order is the more meaningful one to
        read a report in (matches every Construction report writer's
        own convention).
    path : str | Path
        The .txt file to write (parent directory created if missing).
        A relative path resolves against the process's current working
        directory -- NOT this module's location, and not the calling
        script's location either. To have the report always land next
        to the script that built it, build an absolute path at the
        call site: Path(__file__).resolve().parent / "report.txt".
    title : str
        Header title. Defaults to system.label or "Resolution report".
    """
    header_title = title or system.label or "Resolution report"
    sections = [RULE, header_title.center(72), RULE, ""]

    for ss in system.shafts:
        sections.append(RULE)
        sections.append(f"SHAFT: {ss.name}")
        sections.append(RULE)

        result = library.get_or_none(ss.name)
        if result is None:
            sections.append("  (no result -- not solved, or solved into a "
                             "different library)")
        else:
            sections.append(shaft_result_block(result))
        sections.append("")

    sections.append(RULE)
    text = "\n".join(sections) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text