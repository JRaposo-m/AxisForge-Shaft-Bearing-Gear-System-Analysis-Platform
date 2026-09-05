"""
fixtures/studies/shafts/fem_studies/outputs/comparison_report.py

Report writer for a TWO-SOLVE comparison of the same SpurHelicalGearSystem
under Resolution -- e.g. shaft_fem.timoshenko_rigid vs
shaft_fem.euler_bernoulli_rigid, but genuinely label-agnostic: this
module never assumes which two configurations produced the two
ShaftResults it is handed, only that they were solved against the SAME
`system` (same shaft names, same node positions) so their per-shaft
quantities line up index-for-index. Comparing two ShaftResults that
were NOT solved on the same system (different mesh, different loads)
would silently produce a meaningless table -- this module does not
guard against that; it is the caller's responsibility, the same way
check_resolution_fem_compare.py solves ONE `system` object twice rather
than trusting two separately-built systems to match.

Sibling of resolution_report.py in this same fem_studies/ package --
NOT a subfolder-per-analysis split. Both report shapes (single-solve,
comparison) belong to the same Resolution/shaft_fem domain and read the
same ShaftResults shape; splitting further into single/ and comparison/
subdirectories would separate two files that are read together far more
often than either is read alone. One file per report SHAPE, one folder
per domain -- convergence_report.py (once that study exists) joins here
the same way, not under its own subfolder.

Compares TWO things now, not one: comparison_table() -- the governing
scalars (sigma_b_max, v_max, each bearing's Fr/Fa) resolution_report.py's
own summary_block()/bearing_reactions_table() already treat as "worth a
table row" -- AND three per-node tables (bending_shear_comparison_table,
deflection_comparison_table, stress_comparison_table), one row per mesh
node, mirroring resolution_report.py's own three per-node tables but
with both sides plus delta/delta% instead of one column set. The
per-node tables were added because the governing MAXIMA alone hide
where along the shaft the two theories actually diverge, and by how
much away from the worst point -- e.g. Euler-Bernoulli vs Timoshenko
deflection typically diverges MORE the shorter/stubbier the local span,
which a single v_max row cannot show if the worst point happens to sit
on a slender section while a stubby one nearby diverges harder in
relative terms. Bending moment and shear (V) are included as much for
cross-validation as for the theory comparison itself: M(x)/V(x) come
from static equilibrium on the same loads/positions and are expected to
match almost exactly regardless of beam theory (small differences can
still appear at nodes straddling a distributed gear-mesh load, if the
two solves used different Gauss integration orders there) -- a large
M/V delta at a node is a stronger signal of a genuine bug (mismatched
loads, mismatched mesh) than any deflection delta could be, since
deflection is SUPPOSED to differ between theories. This module does not
encode any of that as a pass/fail check or a flag column -- it only
tabulates delta/delta%, the same neutral, non-judgmental stance
resolution_report.py itself takes (it reports numbers, it does not
grade them). Reading the numbers for a physical sanity check (e.g. a
Timoshenko-vs-Euler-Bernoulli deflection ordering that should never
invert) is on the caller/reader, same as it was in
check_resolution_fem_compare.py's own console output.

Node-count mismatch between the two sides is SHOWN, not hidden or
raised on -- see shaft_comparison_block()'s own docstring, and
_zip_nodes()'s own docstring for what happens to the per-node tables
specifically when it occurs (they are skipped with a one-line notice,
never zipped short and mislabelled). A mismatch usually means the two
solves did not, in fact, share a mesh (different extra_mandatory
refinement, or -- the case this module cannot rule out by itself --
two different `system` objects passed to the two solve_system() calls
upstream), and the reader should see that flag before trusting any
delta% column at all.

Writes with encoding="utf-8" explicitly, same reason as every other
report writer in this package. Duplicates resolution_report.py's own
_table()/RULE/SUB rather than importing them -- same explicit "no
shared _io.py-style helper" preference that module's own docstring
states, applied consistently here.

Dependency (results/ and fixtures/ only, read-only access -- no core
modification):
  axisforge.results.fem_results.shaft_results
      ShaftResults -- the result SHAPE. TYPE_CHECKING-only: this module
      only ever reads objects handed to it, it never builds them.
  axisforge.fixtures.studies.shafts.fem_studies.results_library
      RigidBearingFEMResultsLibrary -- the registry this report walks,
      once per side of the comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

RULE = "=" * 88
SUB = "-" * 88

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.results.fem_results.shaft_results import ShaftResults
    from axisforge.fixtures.studies.shafts.fem_studies.results_library import (
        RigidBearingFEMResultsLibrary,
    )


# ---------------------------------------------------------------------------
# Table formatting -- duplicated from resolution_report.py on purpose;
# see this module's own top docstring.
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width ASCII table -- identical contract to
    resolution_report.py's own _table(): headers/rows are
    already-formatted strings, columns pad to the widest cell, 2-space
    gutters, rule line under the header. "(no data)" if `rows` is
    empty."""
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


def _delta_pct(a: float, b: float, atol: float = 1e-6) -> float:
    """
    % change of b relative to a, guarded against the division-by-
    near-zero blow-up: M/V/Fa at a free end or a non-driven bearing DOF
    is PHYSICALLY zero, but the solver returns floating-point noise
    like -1.8e-13, not an exact 0.0. Dividing that noise by itself
    (or by another noise value) amplifies into nonsense like -99.97%
    or inf% -- a false signal that has nothing to do with beam theory
    and everything to do with FEM round-off, and it drowns out the
    real deltas the report exists to show.

    Absolute-error check FIRST: if BOTH |a| and |b| are already within
    `atol` of zero, they are the SAME zero as far as this report is
    concerned, and this returns 0.0 without ever computing a ratio.
    atol=1e-6 is well below any physically meaningful M[N.mm]/V[N]/
    v[mm]/sigma_b[MPa]/Fr,Fa[N] value in this platform's shaft studies
    (all of those sit at 1e-4 or larger when genuinely non-zero) and
    well above typical FEM solve noise (~1e-10 to 1e-13) -- so it
    separates "numerical zero" from "small but real" correctly for
    every quantity this module tables, without needing a different
    atol per column.

    Only past that guard does the original relative-change logic run:
    a == 0.0 (and |b| > atol, a genuine zero-to-nonzero jump, not
    noise) still returns inf -- that IS unmeasurable in relative terms,
    and reporting a fabricated 0% there would hide a real jump instead
    of a fake one.
    """
    if abs(a) <= atol and abs(b) <= atol:
        return 0.0
    if a == 0.0:
        return 0.0 if b == 0.0 else float("inf")
    return (b - a) / abs(a) * 100.0


def _zip_nodes(
    result_a: "ShaftResults",
    result_b: "ShaftResults",
) -> list[tuple[float, int, int]] | None:
    """
    Index-aligned (x, i_a, i_b) triples for every node, ASSUMING both
    results share the same mesh (same x_nodes, same order) -- the same
    assumption comparison_table() already makes for bearing_nodes.
    Returns None instead of zipping short when the node COUNTS differ,
    so a per-node table is never silently built from mismatched rows;
    callers must check for None and print a notice instead of a table
    (see the three *_comparison_table() functions below). Does NOT
    check that x values themselves match one-for-one when counts are
    equal -- two same-length but differently-refined meshes would pass
    this check and still produce a meaningless table; see this
    module's own top docstring on why detecting that fully is out of
    scope here.
    """
    if len(result_a.x_nodes) != len(result_b.x_nodes):
        return None
    return list(zip(result_a.x_nodes, range(len(result_a.x_nodes)), range(len(result_b.x_nodes))))


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

def comparison_table(
    result_a: "ShaftResults",
    result_b: "ShaftResults",
    label_a: str,
    label_b: str,
) -> str:
    """
    One row per governing quantity: sigma_b_max, v_max, then each
    bearing's Fr/Fa (bearing_nodes zipped index-for-index between the
    two results -- same "same system, same bearing order" assumption
    this module's own top docstring states). Columns: quantity,
    `label_a` value, `label_b` value, delta (b - a), delta %.
    """
    headers = ["quantity", label_a, label_b, "delta", "delta %"]
    rows = [
        ["sigma_b_max [MPa]", f"{result_a.sigma_b_max:.4f}",
         f"{result_b.sigma_b_max:.4f}",
         f"{result_b.sigma_b_max - result_a.sigma_b_max:.4f}",
         f"{_delta_pct(result_a.sigma_b_max, result_b.sigma_b_max):.2f}%"],
        ["v_max [mm]", f"{result_a.v_max:.4f}", f"{result_b.v_max:.4f}",
         f"{result_b.v_max - result_a.v_max:.4f}",
         f"{_delta_pct(result_a.v_max, result_b.v_max):.2f}%"],
    ]
    for na, nb in zip(result_a.bearing_nodes, result_b.bearing_nodes):
        rows.append([
            f"{na.label} Fr [N]", f"{na.Fr:.4f}", f"{nb.Fr:.4f}",
            f"{nb.Fr - na.Fr:.4f}", f"{_delta_pct(na.Fr, nb.Fr):.2f}%",
        ])
        rows.append([
            f"{na.label} Fa [N]", f"{na.Fa:.4f}", f"{nb.Fa:.4f}",
            f"{nb.Fa - na.Fa:.4f}", f"{_delta_pct(na.Fa, nb.Fa):.2f}%",
        ])
    return _table(headers, rows)


def bending_shear_comparison_table(
    result_a: "ShaftResults",
    result_b: "ShaftResults",
    label_a: str,
    label_b: str,
) -> str:
    """
    Per-node table: x, M[N.mm] and V[N] (resultant magnitude, not split
    xz/xy -- see this module's own top docstring for why this table is
    here at all: cross-validation, not theory comparison, since M/V are
    expected to match almost exactly regardless of beam theory). "(mesh
    mismatch -- node counts differ, skipped)" instead of a table if
    _zip_nodes() returns None.
    """
    pairs = _zip_nodes(result_a, result_b)
    if pairs is None:
        return (f"  (mesh mismatch -- {len(result_a.x_nodes)} nodes "
                f"({label_a}) vs {len(result_b.x_nodes)} nodes "
                f"({label_b}), skipped)")
    headers = ["x[mm]", f"M[N.mm] ({label_a})", f"M[N.mm] ({label_b})", "M delta %",
               f"V[N] ({label_a})", f"V[N] ({label_b})", "V delta %"]
    rows = [
        [f"{x:.1f}", f"{result_a.M[ia]:.1f}", f"{result_b.M[ib]:.1f}",
         f"{_delta_pct(result_a.M[ia], result_b.M[ib]):.2f}%",
         f"{result_a.V[ia]:.1f}", f"{result_b.V[ib]:.1f}",
         f"{_delta_pct(result_a.V[ia], result_b.V[ib]):.2f}%"]
        for x, ia, ib in pairs
    ]
    return _table(headers, rows)


def deflection_comparison_table(
    result_a: "ShaftResults",
    result_b: "ShaftResults",
    label_a: str,
    label_b: str,
) -> str:
    """
    Per-node table: x, v[mm] (resultant deflection) from both sides,
    delta, delta % -- the table most likely to show the actual
    Timoshenko-vs-Euler-Bernoulli (or whichever two theories) physics
    difference, node by node rather than only at the single worst
    point v_max already reports. "(mesh mismatch...)" instead of a
    table if _zip_nodes() returns None.
    """
    pairs = _zip_nodes(result_a, result_b)
    if pairs is None:
        return (f"  (mesh mismatch -- {len(result_a.x_nodes)} nodes "
                f"({label_a}) vs {len(result_b.x_nodes)} nodes "
                f"({label_b}), skipped)")
    headers = ["x[mm]", f"v[mm] ({label_a})", f"v[mm] ({label_b})", "delta[mm]", "delta %"]
    rows = [
        [f"{x:.1f}", f"{result_a.v[ia]:.4f}", f"{result_b.v[ib]:.4f}",
         f"{result_b.v[ib] - result_a.v[ia]:.4f}",
         f"{_delta_pct(result_a.v[ia], result_b.v[ib]):.2f}%"]
        for x, ia, ib in pairs
    ]
    return _table(headers, rows)


def stress_comparison_table(
    result_a: "ShaftResults",
    result_b: "ShaftResults",
    label_a: str,
    label_b: str,
) -> str:
    """
    Per-node table: x, sigma_b[MPa] (bending stress) from both sides,
    delta, delta %. Stress differences between theories at a given node
    are a direct consequence of the M(x) values already cross-validated
    in bending_shear_comparison_table() being fed through the SAME
    section modulus W(x) either side (geometry doesn't change with beam
    theory) -- so a stress delta here should already be explained by
    whatever small M delta showed up there, not an independent source of
    difference. "(mesh mismatch...)" instead of a table if _zip_nodes()
    returns None.
    """
    pairs = _zip_nodes(result_a, result_b)
    if pairs is None:
        return (f"  (mesh mismatch -- {len(result_a.x_nodes)} nodes "
                f"({label_a}) vs {len(result_b.x_nodes)} nodes "
                f"({label_b}), skipped)")
    headers = ["x[mm]", f"sigma_b[MPa] ({label_a})", f"sigma_b[MPa] ({label_b})",
               "delta[MPa]", "delta %"]
    rows = [
        [f"{x:.1f}", f"{result_a.sigma_b[ia]:.2f}", f"{result_b.sigma_b[ib]:.2f}",
         f"{result_b.sigma_b[ib] - result_a.sigma_b[ia]:.2f}",
         f"{_delta_pct(result_a.sigma_b[ia], result_b.sigma_b[ib]):.2f}%"]
        for x, ia, ib in pairs
    ]
    return _table(headers, rows)


def shaft_comparison_block(
    name: str,
    result_a: "ShaftResults",
    result_b: "ShaftResults",
    label_a: str,
    label_b: str,
) -> str:
    """
    One shaft's full comparison section: node counts for both sides,
    shown even when equal so a mismatch is visible right above every
    table it would otherwise silently undermine (see this module's own
    top docstring) -- then comparison_table() (governing scalars), then
    the three per-node tables (bending & shear, deflection, stress), in
    that order to match resolution_report.py's own per-node table
    ordering.
    """
    return "\n".join([
        f"  nodes ({label_a}) : {len(result_a.x_nodes)}",
        f"  nodes ({label_b}) : {len(result_b.x_nodes)}",
        "",
        comparison_table(result_a, result_b, label_a, label_b),
        "",
        "BENDING & SHEAR (cross-validation -- expected to match closely)",
        SUB,
        bending_shear_comparison_table(result_a, result_b, label_a, label_b),
        "",
        "DEFLECTION",
        SUB,
        deflection_comparison_table(result_a, result_b, label_a, label_b),
        "",
        "SECTION & STRESS",
        SUB,
        stress_comparison_table(result_a, result_b, label_a, label_b),
    ])


def write_comparison_report(
    library_a: "RigidBearingFEMResultsLibrary",
    library_b: "RigidBearingFEMResultsLibrary",
    system: "SpurHelicalGearSystem",
    path: "str | Path",
    label_a: str = "a",
    label_b: str = "b",
    title: str = "",
) -> str:
    """
    Write ONE .txt comparing two already-solved libraries for the SAME
    `system` -- one "SHAFT: <name>" section per shaft in system.shafts
    (system's own order, matching resolution_report.py's own
    write_resolution_report() convention), each holding
    shaft_comparison_block() for that shaft, or "(missing from: ...)"
    naming whichever side(s) have no result for that name. Returns the
    written text, so a caller that wants to inspect/verify it doesn't
    have to re-open the file it just wrote.

    Parameters
    ----------
    library_a, library_b : RigidBearingFEMResultsLibrary
        Already solved -- e.g. via two separate solve_system() calls
        on the SAME `system` object, one per shaft_fem capability (see
        check_resolution_fem_compare.py). Purely reads both -- does
        not solve or validate either.
    system : SpurHelicalGearSystem
        Supplies the shaft ORDER and NAMES this report walks, same
        reasoning as write_resolution_report()'s own `system`
        parameter.
    path : str | Path
        The .txt file to write (parent directory created if missing).
        A relative path resolves against the process's current working
        directory -- build an absolute path at the call site
        (Path(__file__).resolve().parent / "report.txt") to have the
        report always land next to the script that built it.
    label_a, label_b : str
        Column headers identifying which side is which in every table
        -- e.g. "timoshenko" / "euler_bernoulli". Purely cosmetic, but
        deliberately required rather than defaulted to something
        physics-specific, since this module makes no assumption about
        what the two sides are (see this module's own top docstring).
    title : str
        Header title. Defaults to "<system.label> -- comparison", or
        "Resolution comparison report" if system.label is empty.
    """
    header_title = title or (
        f"{system.label} -- comparison" if system.label
        else "Resolution comparison report"
    )
    sections = [RULE, header_title.center(88), RULE, ""]

    for ss in system.shafts:
        sections.append(RULE)
        sections.append(f"SHAFT: {ss.name}")
        sections.append(RULE)

        result_a = library_a.get_or_none(ss.name)
        result_b = library_b.get_or_none(ss.name)
        if result_a is None or result_b is None:
            missing = []
            if result_a is None:
                missing.append(label_a)
            if result_b is None:
                missing.append(label_b)
            sections.append(f"  (missing from: {', '.join(missing)})")
        else:
            sections.append(shaft_comparison_block(ss.name, result_a, result_b, label_a, label_b))
        sections.append("")

    sections.append(RULE)
    text = "\n".join(sections) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text