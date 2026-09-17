"""
axisforge/fixtures/studies/shafts/convergence_studies/outputs/convergence_report.py

CONTENT ONLY -- same discipline as fem_studies/outputs/resolution_report.py
and fem_studies/outputs/comparison_report.py: every function here builds
and returns a text block, none of them opens a file. The one module
that actually writes is fixtures/studies/text_report.py
(write_studies_report()), which imports shaft_convergence_block() from
here the same way it already imports shaft_result_block() and
shaft_comparison_block() from their own sibling packages.

## CHANGED (earlier pass): this module was still hardcoded to exactly 3
## displacement-plane metrics -- levels_history_table() did
## `f_xz, f_xy, f_res = metrics` (positional unpacking, assumed a
## 3-tuple) and gci_detail_table()/interval_block() indexed
## rec.gci_history[i]["xz"]/["xy"]/["res"] by literal plane-name keys.
## That predates metric_spec.py's STUDY_VARIABLES/MetricSpec split:
## rec.point_metrics_history is now `list[dict[str, float]]` keyed by
## spec.name (e.g. "v_xz"/"v_xy"/"v_res" for a displacement run,
## "M_xz"/"M_xy"/"M_res" for a moment run -- see
## convergence_results.py's own metric_names() helper), and
## rec.gci_history is `list[dict[str, RichardsonGCI | _DummyGCI]]`,
## same keys. The old unpacking silently assigned f_xz/f_xy/f_res to
## the dict's KEYS (as strings) instead of raising, which is exactly
## why this crashed with "Unknown format code 'f' for object of type
## 'str'" only once a non-displacement metrics list (M_xz/M_xy/M_res)
## was actually exercised.
##
## Fix: every function here now reads the tracked metric NAMES off the
## record itself (_metric_names(), below) instead of assuming which
## three exist -- so this module works unchanged for
## default_displacement_metrics(), default_moment_metrics(), or any
## future metrics list, including one that doesn't have exactly 3
## entries. The "plane" column (gci_detail_table) is renamed "metric"
## accordingly, and interval_block()'s closing Richardson-extrapolated
## line is now printed ONCE PER TRACKED METRIC (was hardcoded to only
## ever print "v_res") instead of assuming a "_res" plane exists at all.

## CHANGED (this pass, confirmed with erg 2026-09-17): interval_block()
## no longer special-cases `rec.applicable is False` -- that field is
## gone from ConvergenceRecord (see convergence_results.py's own
## CHANGED note): every interval now went through a real solve attempt,
## so there is no "SKIPPED (no GCI attempted)" case left to render.
## shaft_convergence_block()'s skipped-interval count/suffix is removed
## for the same reason. A load-free interval that still fails to
## converge (near/through-zero M destabilising the relative-error GCI)
## now simply prints as an ordinary "NOT CONVERGED (stopped after N
## level(s))" block, same as any other non-convergent interval -- no
## special wording, because from the report's point of view it isn't a
## special case anymore.

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
    per-level history table (metric value at every grade tried, one
    column-group per TRACKED metric -- see _metric_names() -- plus
    GCI_f_m percentage once 3+ levels exist -- "--" before that, "n/a"
    for a _DummyGCI, i.e. a non-uniform refinement ratio, a degenerate
    observed order, or a metric landing exactly on 0.0, see
    convergence_solver.py's own RichardsonGCI/_DummyGCI docstrings), a
    full Richardson-detail table (r, p, e_m_c, e_f_m, GCI_m_c, GCI_f_m,
    f_h0 -- one row per level-transition per tracked metric), and one
    closing Richardson-extrapolated final-estimate line PER tracked
    metric (f_h0, with its observed order p).
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
itself.

Precision: metric values print at 10 decimals, percentages at 6
decimals, and gci_detail_table() has delta_m_c/delta_f_m -- the RAW
signed differences between levels, in scientific notation -- widened
deliberately as a diagnostic: p is computed directly from the ratio of
logs of exactly those two raw deltas (see RichardsonGCI.p's own
formula), so when they are both tiny and comparable to
floating-point/solver noise, p becomes numerically unstable and can
land far from the true order without that meaning genuine
super-convergence. This wider precision is what lets a reader confirm,
from the report alone, whether a suspicious p is real signal or noise.
This matters even more now that load-free/near-zero-M intervals are no
longer filtered out before reaching this report -- delta_m_c/delta_f_m
are the columns to check first when a moment interval reports NOT
CONVERGED with a suspicious-looking p.

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
    GCI_f_m -- non-uniform refinement ratio, degenerate order, or a
    metric landing exactly on 0.0 -- or a field _DummyGCI does not have
    at all) prints as 'n/a', never as the literal string 'nan'.

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


def _metric_names(rec: "ConvergenceRecord") -> list[str]:
    """
    Ordered list of tracked metric names for this record -- e.g.
    ["v_xz", "v_xy", "v_res"] for a displacement run, ["M_xz", "M_xy",
    "M_res"] for a moment run, or whatever future MetricSpec list a
    caller passes. MeshConvergenceStudy runs the exact same self._metrics
    list at every grade level, so every entry in point_metrics_history
    shares the same key set -- reading it off the FIRST level is
    enough, dict insertion order (Python 3.7+) keeps it stable and
    matches the order the caller's `metrics` list was built in.

    Falls back to gci_history's keys if point_metrics_history is
    somehow empty (interval never solved a single level -- e.g. it
    raised before the first grade completed). Returns [] only if
    neither has anything, which the callers below already handle by
    producing an empty table/no lines, same as any other "(none)" case
    in this module.
    """
    if rec.point_metrics_history:
        return list(rec.point_metrics_history[0].keys())
    if rec.gci_history:
        return list(rec.gci_history[0].keys())
    return []


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

def levels_history_table(rec: "ConvergenceRecord") -> str:
    """
    Per-level metric history for one interval: every TRACKED metric's
    value at every grade tried (one value column per name from
    _metric_names()), plus that metric's GCI_f_m (fine-vs-medium, see
    this module's own top docstring for why not GCI_m_c) once 3+
    levels exist.

    rec.gci_history[i] is the GCI dict computed from the triple of
    levels ending at rec.levels[i + 2] (Richardson needs 3 consecutive
    levels, and MeshConvergenceStudy._converge_one_load() only appends
    to gci_history once len(history) >= 3) -- so a level's GCI columns
    stay "--" for its first two entries, never a KeyError/IndexError.
    """
    names = _metric_names(rec)
    headers = (
        ["level"]
        + [f"{name} [mm]" for name in names]
        + [f"GCI_{name}" for name in names]
    )
    rows: list[list[str]] = []
    n_gci = len(rec.gci_history)

    for i, (level, metrics) in enumerate(zip(rec.levels, rec.point_metrics_history)):
        # 10 decimal places -- a diagnostic widening: two levels that
        # print as bit-for-bit identical at 5 decimals can still differ
        # starting at the 8th/9th digit, and that residual difference
        # is exactly what feeds RichardsonGCI.p -- rounding it away
        # here would hide the very thing gci_detail_table()'s raw
        # delta_m_c/delta_f_m columns are meant to let you inspect.
        value_cells = [f"{metrics[name]:.10f}" for name in names]

        gci_index = i - 2
        if 0 <= gci_index < n_gci:
            gci = rec.gci_history[gci_index]
            gci_cells = [_fmt_gci_pct(gci[name].GCI_f_m) for name in names]
        else:
            gci_cells = ["--"] * len(names)

        rows.append([level] + value_cells + gci_cells)

    return _table(headers, rows)


def gci_detail_table(rec: "ConvergenceRecord") -> str:
    """
    Full RichardsonGCI detail, one row per (level-transition, tracked
    metric) pair, in the same chronological order levels_history_table()
    walks. Surfaces everything RichardsonGCI computes that
    levels_history_table() has no room for -- r (refinement ratio), p
    (OBSERVED ORDER OF CONVERGENCE), delta_m_c/delta_f_m (RAW, signed
    differences in mm -- read straight off rec.point_metrics_history,
    not derived from the relative e_m_c/e_f_m), e_m_c/e_f_m (signed
    RELATIVE error, medium-vs-coarse and fine-vs-medium), GCI_m_c (the
    one levels_history_table() omits), and f_h0 (the
    Richardson-extrapolated estimate of the converged value itself) --
    rather than duplicating GCI_f_m%, which levels_history_table()
    already tables per level.

    The "metric" column (was "plane" before this module became
    metric-name-generic) holds whatever _metric_names() returns for
    this record -- "v_xz"/"v_xy"/"v_res", "M_xz"/"M_xy"/"M_res", or any
    future MetricSpec.name.

    delta_m_c/delta_f_m are printed in scientific notation specifically
    to make a diagnostic question easy to answer at a glance: is the
    actual change between two mesh levels (this column) meaningfully
    larger than floating-point/solver noise, or is p being computed
    from two numbers that are both already indistinguishable from zero
    at that precision? p is a ratio of logs of these two raw deltas
    (see RichardsonGCI.p's own formula) -- when both deltas are tiny
    and of comparable magnitude to solver round-off, p becomes
    numerically unstable and can land anywhere without that meaning
    genuine super-convergence. A p noticeably far from the FEM
    element's theoretical order is worth treating as suspect until
    delta_m_c/delta_f_m here confirm the underlying signal was actually
    large enough to trust. This is also the column to check first for a
    moment interval that keeps reporting NOT CONVERGED near a zero
    crossing: a small delta with a large e_m_c/e_f_m (relative error
    computed against a near-zero f_coarse/f_medium) is exactly the
    signature of that limitation, not a real meshing problem.

    A _DummyGCI entry (non-uniform refinement ratio, a degenerate
    observed order -- e.g. p ~= 0 or a non-monotonic level-to-level
    change -- or a metric landing exactly on 0.0; see
    convergence_solver.py's own RichardsonGCI/_DummyGCI docstrings)
    only ever carries GCI_m_c/GCI_f_m/converged; r/p/e_m_c/e_f_m/f_h0
    print as 'n/a' via getattr(..., float("nan")) rather than raising
    AttributeError. delta_m_c/delta_f_m are still computed either way
    -- they come from point_metrics_history, not from the GCI object.
    """
    names = _metric_names(rec)
    headers = ["transition", "metric", "r", "p",
               "delta_m_c [mm]", "delta_f_m [mm]",
               "e_m_c", "e_f_m", "GCI_m_c", "GCI_f_m", "f_h0 [mm]", "converged"]
    rows: list[list[str]] = []

    for i, gci_dict in enumerate(rec.gci_history):
        lvl_c, lvl_m, lvl_f = rec.levels[i], rec.levels[i + 1], rec.levels[i + 2]
        transition = f"{lvl_c}->{lvl_m}->{lvl_f}"
        m_c = rec.point_metrics_history[i]
        m_m = rec.point_metrics_history[i + 1]
        m_f = rec.point_metrics_history[i + 2]
        for name in names:
            delta_m_c = m_m[name] - m_c[name]
            delta_f_m = m_f[name] - m_m[name]

            gci = gci_dict[name]
            r = getattr(gci, "r", float("nan"))
            p = getattr(gci, "p", float("nan"))
            e_m_c = getattr(gci, "e_m_c", float("nan"))
            e_f_m = getattr(gci, "e_f_m", float("nan"))
            f_h0 = getattr(gci, "f_h0", float("nan"))
            rows.append([
                transition, name,
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
    table (r, p, e_m_c, e_f_m, GCI_m_c, f_h0 per transition per tracked
    metric), and -- once at least one GCI has been computed -- one
    Richardson-extrapolated final-estimate line PER tracked metric
    (f_h0, from the LAST/finest transition available, with its own
    observed order p). Previously this always assumed a "_res" plane
    and printed exactly one such line; it now prints one per name in
    _metric_names(), so a moment run reports M_xz/M_xy/M_res estimates
    instead of silently reporting nothing (or crashing on a "res" key
    that a differently-named metrics list never had).

    ## CHANGED (this pass, confirmed with erg 2026-09-17): the
    ## `if not rec.applicable: return ...` short-circuit that used to
    ## sit at the top of this function is REMOVED -- ConvergenceRecord
    ## no longer has an `applicable`/`skip_reason` field (see
    ## convergence_results.py's own CHANGED note). Every record reaching
    ## this function went through a real solve attempt, so it always
    ## renders the full CONVERGED/NOT CONVERGED block below.
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

    if rec.point_x:
        # MomentConvergenceStudy-only -- the ACTUAL physical x [mm]
        # each tracked point sits at, fixed from grade_0 and reused
        # unchanged at every grade (see MetricSpec.point_x's own
        # docstring in metric_spec.py). Added per erg's request
        # (2026-09-17): "quero o x do ponto que esta a ser avaliado" --
        # makes the fixed-point-across-grades claim checkable directly
        # from the report instead of taken on trust.
        pts_str = ", ".join(
            f"{pt} @ x={x:.3f}mm" for pt, x in sorted(rec.point_x.items())
        )
        lines.append(f"    tracked points (fixed at grade_0): {pts_str}")

    lines.append("")
    lines.append(levels_history_table(rec))

    if rec.gci_history:
        lines.append("")
        lines.append("  Richardson GCI detail (per transition, per metric):")
        lines.append(gci_detail_table(rec))

        last = rec.gci_history[-1]
        lines.append("")
        for name in _metric_names(rec):
            gci = last[name]
            p = getattr(gci, "p", float("nan"))
            f_h0 = getattr(gci, "f_h0", float("nan"))
            lines.append(
                f"  Richardson-extrapolated {name} (p={_fmt_num(p, '.6f')}): "
                f"{_fmt_num(f_h0, '.10f')} mm"
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

    ## CHANGED (this pass, confirmed with erg 2026-09-17): the
    ## skipped-interval count (n_skipped/skipped_suffix, keyed off
    ## rec.applicable) is REMOVED -- there is no more "skipped, no load"
    ## outcome (see convergence_results.py's/convergence_solver.py's own
    ## CHANGED notes). The summary line is back to a plain
    ## "N/M interval(s) converged".
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