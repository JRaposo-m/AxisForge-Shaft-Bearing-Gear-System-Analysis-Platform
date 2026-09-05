"""
axisforge/fixtures/studies/shafts/convergence_studies/outputs/convergence_report.py

CONTENT ONLY -- same discipline as fem_studies/outputs/resolution_report.py
and fem_studies/outputs/comparison_report.py: every function here builds
and returns a text block, none of them opens a file. The one module
that actually writes is fixtures/studies/text_report.py
(write_studies_report()), which imports shaft_convergence_block() from
here the same way it already imports shaft_result_block() and
shaft_comparison_block() from their own sibling packages.

Unlike comparison_report.py, this module does NOT also keep its own
write_convergence_report(). comparison_report.py's standalone writer
predates the "one writer for the whole Studies domain" rule that
resolution_report.py's own split already documents (write_resolution_report()
was removed from it, not kept as a second option) -- this module follows
that later, stricter version of the rule from the start: text_report.py
is the only place a convergence .txt gets written, there is no
"just a convergence report, standalone" entry point here.

_table() is duplicated from resolution_report.py/comparison_report.py by
the same explicit convention those two document for each other -- no
shared _io.py-style helper across sibling report modules in this
platform.

Content shape, per shaft (one MeshRefinementResult):
  - a one-line summary: how many of its intervals converged out of how
    many were attempted (result.per_load.values()).
  - one block per interval (result.per_load, insertion order --
    MeshConvergenceStudy.intervals_from_shaft_system()'s own order,
    itself gears-then-external_distributed-then-bearings, filtered by
    whatever `regions` run_convergence() was called with): the interval
    label/span, CONVERGED/NOT CONVERGED status, final node count, a
    per-level history table (metric value at every grade tried, plus
    GCI_f_m percentage once 3+ levels exist -- "--" before that, "n/a"
    for a _DummyGCI, i.e. a non-uniform refinement ratio, see
    convergence_solver.py's own RichardsonGCI/_DummyGCI docstrings), a
    full Richardson-detail table (r, p, e_m_c, e_f_m, GCI_m_c, GCI_f_m,
    f_h0 -- one row per level-transition per plane), and a one-line
    Richardson-extrapolated final estimate for the resultant
    displacement (f_h0, with its observed order p).
  - a closing line listing result.all_extra_nodes -- the union of every
    interval's converged (or last-tried) node positions, exactly the
    list a caller would forward to Mesh1D(..., extra_mandatory=...) to
    actually apply this study's refinement to the resolution mesh.

GCI_f_m (fine-vs-medium) is the percentage shown in
levels_history_table(), not GCI_m_c (medium-vs-coarse) -- GCI_f_m is
the error estimate on the LAST (finest) pair of levels tried, i.e. the
number that answers "how much would the answer still move if I refined
once more from here", which is the number RichardsonGCI.converged
actually gates on together with GCI_m_c. That table stays intentionally
narrow (it is the one meant to be skimmed level-by-level); everything
else RichardsonGCI computes -- r, p, e_m_c, e_f_m, GCI_m_c, f_h0 -- now
lives in gci_detail_table() instead of being left only on the object
itself (an earlier version of this module said exactly that: "GCI_m_c
stays available on the RichardsonGCI object... this report just
doesn't table it" -- corrected here on explicit request: everything a
RichardsonGCI computes, including the observed order of convergence p,
belongs in the written report, not just in memory on an object nobody
downstream inspects).

Precision was widened as a deliberate DIAGNOSTIC step (not cosmetic):
f_xz/f_xy/f_res now print at 10 decimals (was 5), percentages at 6
decimals (was 3), and gci_detail_table() gained delta_m_c/delta_f_m --
the RAW signed mm differences between levels, in scientific notation.
Two levels that looked bit-for-bit identical at 5 decimals in an
earlier run (a real case: two DistributedRadialLoad convergence checks
both showed p noticeably above 2, the theoretical order for this
platform's 2-node beam elements, while every gear-interval check
landed almost exactly on p~=2) turned out to hide their actual
per-level change past the 5th digit -- and RichardsonGCI.p is computed
directly from the ratio of exactly those two raw deltas, so when they
are both tiny and comparable to floating-point/solver noise, p becomes
numerically unstable and can land far from the true order without that
meaning genuine super-convergence. This wider precision is what lets a
reader confirm, from the report alone, whether a suspicious p is real
signal or noise -- see gci_detail_table()'s own docstring for the full
reasoning.

Dependency (results/ only, read-only access -- no core modification):
  axisforge.results.fem_results.convergence_results
      ConvergenceRecord, MeshRefinementResult -- the result SHAPES this
      module formats. TYPE_CHECKING-only: this module reads fields off
      instances handed to it, it never constructs one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.convergence_results import (
        ConvergenceRecord,
        MeshRefinementResult,
    )

__all__ = [
    "levels_history_table",
    "gci_detail_table",
    "interval_block",
    "shaft_convergence_block",
]


# ---------------------------------------------------------------------------
# Table helper -- duplicated by convention, see this module's own docstring.
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width ASCII table -- one column per header, left-aligned."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def _fmt_gci_pct(value: float) -> str:
    """value is a fraction (0.01 = 1%%), formatted as a percentage string,
    SIGNED (e_m_c/e_f_m carry a meaningful sign -- direction of change
    between levels -- and it is not stripped here). NaN (a _DummyGCI's
    GCI_f_m -- non-uniform refinement ratio -- or a field _DummyGCI
    does not have at all) prints as 'n/a', never as the literal string
    'nan'.

    6 decimal places (not 3) -- widened specifically so a percentage
    like 0.000012%% doesn't round down to the same 0.000%% as a genuine
    zero; see gci_detail_table()'s own docstring for why telling those
    two apart matters when p looks suspicious."""
    if value != value:  # NaN check without importing math for one use
        return "n/a"
    return f"{value * 100:.6f}%"


def _fmt_num(value: float, fmt: str = ".4f") -> str:
    """Plain numeric formatter (r, p, f_h0 -- NOT percentages) with the
    same NaN-safety as _fmt_gci_pct, for fields _DummyGCI does not
    carry at all (getattr(..., float('nan')) upstream)."""
    if value != value:
        return "n/a"
    return format(value, fmt)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

def levels_history_table(rec: "ConvergenceRecord") -> str:
    """
    Per-level metric history for one interval: the point metric at
    every grade tried, plus GCI_f_m (fine-vs-medium, see this module's
    own top docstring for why not GCI_m_c) once 3+ levels exist.

    rec.gci_history[i] is the GCI computed from the triple of levels
    ending at rec.levels[i + 2] (Richardson needs 3 consecutive levels,
    and MeshConvergenceStudy._converge_one_load() only appends to
    gci_history once len(history) >= 3) -- so a level's GCI columns
    stay "--" for its first two entries, never a KeyError/IndexError.
    """
    headers = ["level", "f_xz [mm]", "f_xy [mm]", "f_res [mm]", "GCI_xz", "GCI_xy", "GCI_res"]
    rows: list[list[str]] = []
    n_gci = len(rec.gci_history)

    for i, (level, metrics) in enumerate(zip(rec.levels, rec.point_metrics_history)):
        f_xz, f_xy, f_res = metrics
        gci_index = i - 2
        if 0 <= gci_index < n_gci:
            gci = rec.gci_history[gci_index]
            gci_xz = _fmt_gci_pct(gci["xz"].GCI_f_m)
            gci_xy = _fmt_gci_pct(gci["xy"].GCI_f_m)
            gci_res = _fmt_gci_pct(gci["res"].GCI_f_m)
        else:
            gci_xz = gci_xy = gci_res = "--"
        # 10 decimal places (not 5) -- a diagnostic widening: two levels
        # that print as bit-for-bit identical at 5 decimals can still
        # differ starting at the 8th/9th digit, and that residual
        # difference is exactly what feeds RichardsonGCI.p -- rounding
        # it away here would hide the very thing gci_detail_table()'s
        # raw delta_m_c/delta_f_m columns are meant to let you inspect.
        rows.append([
            level, f"{f_xz:.10f}", f"{f_xy:.10f}", f"{f_res:.10f}",
            gci_xz, gci_xy, gci_res,
        ])

    return _table(headers, rows)


_PLANE_INDEX = {"xz": 0, "xy": 1, "res": 2}


def gci_detail_table(rec: "ConvergenceRecord") -> str:
    """
    Full RichardsonGCI detail, one row per (level-transition, plane)
    pair, in the same chronological order levels_history_table() walks.
    Surfaces everything RichardsonGCI computes that levels_history_table()
    has no room for -- r (refinement ratio), p (OBSERVED ORDER OF
    CONVERGENCE), delta_m_c/delta_f_m (RAW, signed differences in mm --
    f_medium-f_coarse and f_fine-f_medium, read straight off
    rec.point_metrics_history, not derived from the relative e_m_c/e_f_m),
    e_m_c/e_f_m (signed RELATIVE error, medium-vs-coarse and
    fine-vs-medium), GCI_m_c (the one levels_history_table() omits),
    and f_h0 (the Richardson-extrapolated estimate of the converged
    value itself) -- rather than duplicating GCI_f_m%, which
    levels_history_table() already tables per level.

    delta_m_c/delta_f_m are printed in scientific notation specifically
    to make a diagnostic question easy to answer at a glance: is the
    actual change between two mesh levels (this column) meaningfully
    larger than floating-point/solver noise, or is p being computed
    from two numbers that are both already indistinguishable from zero
    at that precision? p is a ratio of logs of these two raw deltas
    (see RichardsonGCI.p's own formula) -- when both deltas are tiny
    and of comparable magnitude to solver round-off, p becomes numerically
    unstable and can land anywhere, including well above the FEM
    element's true theoretical order (~2 for this platform's 2-node
    beam elements) without that meaning genuine super-convergence. A p
    noticeably far from ~2 is worth treating as suspect until
    delta_m_c/delta_f_m here confirm the underlying signal was actually
    large enough to trust.

    A _DummyGCI entry (non-uniform refinement ratio -- see
    convergence_solver.py's own RichardsonGCI/_DummyGCI docstrings)
    only ever carries GCI_m_c/GCI_f_m/converged; r/p/e_m_c/e_f_m/f_h0
    print as 'n/a' via getattr(..., float("nan")) rather than raising
    AttributeError. delta_m_c/delta_f_m are still computed either way --
    they come from point_metrics_history, not from the GCI object.
    """
    headers = ["transition", "plane", "r", "p",
               "delta_m_c [mm]", "delta_f_m [mm]",
               "e_m_c", "e_f_m", "GCI_m_c", "GCI_f_m", "f_h0 [mm]", "converged"]
    rows: list[list[str]] = []

    for i, gci_dict in enumerate(rec.gci_history):
        lvl_c, lvl_m, lvl_f = rec.levels[i], rec.levels[i + 1], rec.levels[i + 2]
        transition = f"{lvl_c}->{lvl_m}->{lvl_f}"
        m_c = rec.point_metrics_history[i]
        m_m = rec.point_metrics_history[i + 1]
        m_f = rec.point_metrics_history[i + 2]
        for plane in ("xz", "xy", "res"):
            idx = _PLANE_INDEX[plane]
            delta_m_c = m_m[idx] - m_c[idx]
            delta_f_m = m_f[idx] - m_m[idx]

            gci = gci_dict[plane]
            r = getattr(gci, "r", float("nan"))
            p = getattr(gci, "p", float("nan"))
            e_m_c = getattr(gci, "e_m_c", float("nan"))
            e_f_m = getattr(gci, "e_f_m", float("nan"))
            f_h0 = getattr(gci, "f_h0", float("nan"))
            rows.append([
                transition, plane,
                _fmt_num(r, ".6f"), _fmt_num(p, ".6f"),
                f"{delta_m_c:.3e}", f"{delta_f_m:.3e}",
                _fmt_gci_pct(e_m_c), _fmt_gci_pct(e_f_m),
                _fmt_gci_pct(gci.GCI_m_c), _fmt_gci_pct(gci.GCI_f_m),
                _fmt_num(f_h0, ".10f"), str(gci.converged),
            ])

    return _table(headers, rows)


def interval_block(rec: "ConvergenceRecord") -> str:
    """
    One interval's full block: header line, final-mesh line, the
    per-level metric/GCI_f_m history table, the full Richardson-detail
    table (r, p, e_m_c, e_f_m, GCI_m_c, f_h0 per transition per plane),
    and -- once at least one GCI has been computed -- a one-line
    Richardson-extrapolated final estimate for the resultant
    displacement (f_h0 on the "res" plane, from the LAST/finest
    transition available, with its own observed order p).
    """
    if rec.converged:
        status = "CONVERGED"
    else:
        status = f"NOT CONVERGED (stopped after {len(rec.levels)} level(s))"

    lines = [
        f"  Interval '{rec.label}'  [{rec.x_lo:.3f}, {rec.x_hi:.3f}] mm  -- {status}",
    ]
    if rec.x_final:
        lines.append(f"    final mesh: {len(rec.x_final)} node(s)")
    else:
        lines.append("    final mesh: (none -- no level was solved)")
    lines.append("")
    lines.append(levels_history_table(rec))

    if rec.gci_history:
        lines.append("")
        lines.append("  Richardson GCI detail (per transition, per plane):")
        lines.append(gci_detail_table(rec))

        last_res = rec.gci_history[-1]["res"]
        p_res = getattr(last_res, "p", float("nan"))
        f_h0_res = getattr(last_res, "f_h0", float("nan"))
        lines.append("")
        lines.append(
            f"  Richardson-extrapolated v_res (p={_fmt_num(p_res, '.6f')}): "
            f"{_fmt_num(f_h0_res, '.10f')} mm"
        )

    return "\n".join(lines)


def shaft_convergence_block(result: "MeshRefinementResult") -> str:
    """
    Full convergence block for one shaft: a one-line summary followed
    by one interval_block() per entry in result.per_load (insertion
    order), then the union of all_extra_nodes ready for
    Mesh1D(..., extra_mandatory=...).

    An empty result.per_load (e.g. run_convergence() was called with
    regions={} on a shaft, or the shaft had no gear/distributed-load
    interval to begin with) reports "0/0 interval(s)" rather than
    printing nothing -- fail-loud/visible, same convention as every
    other block module's "(none)"/"(missing from: ...)" lines.
    """
    n_total = len(result.per_load)
    n_converged = sum(1 for rec in result.per_load.values() if rec.converged)

    lines = [
        f"Shaft '{result.shaft_name}' -- mesh convergence: "
        f"{n_converged}/{n_total} interval(s) converged",
        "",
    ]

    for rec in result.per_load.values():
        lines.append(interval_block(rec))
        lines.append("")

    extra_nodes = result.all_extra_nodes
    if extra_nodes:
        nodes_str = ", ".join(f"{x:.3f}" for x in extra_nodes)
        lines.append(f"  Extra mandatory nodes (union, for Mesh1D): [{nodes_str}]")
    else:
        lines.append("  Extra mandatory nodes (union, for Mesh1D): (none)")

    return "\n".join(lines).rstrip("\n") + "\n"