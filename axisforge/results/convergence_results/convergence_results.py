"""
axisforge/results/fem_results/convergence_results.py

Result SHAPES for the mesh convergence study -- moved here from
axisforge/mesh/shaft/mesh_generation/mesh_convergence_study.py, same
split already applied to shaft_fem: RichardsonGCI/MeshConvergenceStudy
DO calculation (they stay in mesh_generation/, they are the solver side
of this study, the same way RigidBearingFEMSolver stays in solvers/),
while ConvergenceRecord and MeshRefinementResult are pure DATA -- what
a convergence run produced, not how it was produced. This mirrors
ShaftResults living in this same results/fem_results/ package while
RigidBearingFEMSolver stays in solvers/.

## CHANGED (earlier pass): point_metrics_history and gci_history widened
## from a fixed 3-component shape (f_xz/f_xy/f_res) to an OPEN metric
## name -> value / metric name -> GCI mapping. This is the data-side
## half of the per-variable convergence split discussed alongside
## metric_spec.py (axisforge/solvers/mesh/) -- displacement and bending
## moment (and, later, whatever sigma turns out to mean once its source
## is settled) each get tracked as their own named metric rather than
## v being the only thing this shape could ever hold.
##
## gci_history's type hint barely changes -- it was ALREADY
## dict[str, "RichardsonGCI | _DummyGCI"], just documented with the
## three literal keys {"xz","xy","res"} that mesh_convergence_study.py
## happened to always pass. The dataclass itself never assumed those
## specific keys; only the docstring and (elsewhere) convergence_report.py's
## rendering code did. So this file's actual change is smaller than it
## looks: point_metrics_history's tuple -> dict, a new peak_shifted
## field, and docstring updates to stop implying a fixed 3-key set.

## CHANGED (earlier pass, confirmed with erg 2026-09-17): `applicable` and
## `skip_reason` are REMOVED. They existed only to represent the "this
## interval was refused before solving because it has no load" outcome
## that MomentConvergenceStudy used to produce (see convergence_solver.py's
## own MomentConvergenceStudy._interval_has_load(), since deleted). Now
## that every interval always gets a real submodel solve and a real GCI
## attempt (converged=True/False is the only outcome), there is no
## "not evaluated" state left to represent -- keeping the fields around
## unused would just be a second way to ask the same question
## (`applicable` vs `converged`) that report code would have to
## remember never actually goes False anymore. See convergence_solver.py's
## own top-of-file CHANGED note for the reasoning behind removing the
## skip itself.

## CHANGED (this pass, confirmed with erg 2026-09-17): `n_points` and
## `point_x` are REMOVED from ConvergenceRecord. Both were ONLY ever
## set by MomentConvergenceStudy._converge_one_load() (convergence_solver.py),
## which is now deleted entirely -- see that module's own CHANGED note:
## "antes tinha posto a criar pontos de analise e assim e agora isso vai
## ser retirado porque esta abordagem do global gostei mesmo muito". A
## record built by MeshConvergenceStudy never populated either
## field (n_points stayed None, point_x stayed {}), and with the
## per-interval moment study gone, nothing populates them at all
## anymore -- keeping two permanently-dead fields around would just
## invite a future reader to wonder why they're always empty. Report
## code (convergence_report.py) no longer reads either field -- see
## that module's own matching CHANGED note.

ConvergenceRecord.gci_history still holds RichardsonGCI/_DummyGCI
instances directly, keyed by metric name (not a fixed {"xz", "xy",
"res"} set) -- those two classes were NOT converted into a results-side
dataclass (GCIResult) in this pass; that would have meant touching
RichardsonGCI's own call sites in convergence_solver.py, which is
explicitly staying untouched right now beyond the exception-guarding
fix. So this module keeps a TYPE_CHECKING-only reference to them for
the type hint, and otherwise never constructs or reads their internals
-- it only stores whatever convergence_solver.py's own _compute_gci()
hands it, for whichever metric name that value belongs to.

print_report() was REMOVED from MeshRefinementResult on the move -- a
results dataclass printing itself to stdout is the same data/
presentation mixing every other report in this platform has been
pulled apart into a *_report.py content-block module (see
resolution_report.py/comparison_report.py's own "CONTENT ONLY"
docstrings). That formatting logic is not lost, just not moved yet --
whoever builds fem_studies/outputs/convergence_report.py next should
treat MeshRefinementResult.print_report()'s old body as the starting
point for that module's content-block functions, not reintroduce a
print() method here. That future report module now also needs to
iterate an OPEN metric-name set (ConvergenceRecord.metric_names()
below) instead of hardcoding "xz"/"xy"/"res" columns -- flagging here
since it's the natural next file to touch once this lands.

shaft_name was ADDED to MeshRefinementResult (it did not exist before
this move) -- a mesh convergence study runs per ShaftSystem, and
without a name field there was no key for a fixtures-side library to
store results under; every other results shape moving through a
Library in this platform (ShaftResults included) carries its own name
for exactly this reason. Empty string default keeps existing call
sites that construct MeshRefinementResult() with no arguments working
unchanged; whoever wires this into a study (a future
convergence_study.py fixture, mirroring comparison_study.py) is
responsible for setting it before handing the result to a library.

Dependency (mesh/ only, for the TYPE_CHECKING gci_history hint --
no computation, no core modification):
  axisforge.solvers.mesh.convergence_solver
      RichardsonGCI, _DummyGCI -- referenced by type only; this module
      never imports them at runtime, never constructs one, and never
      reads a field off one that isn't already a dict key
      (rec.gci_history[i][metric_name].GCI_f_m etc. is done by report/print
      code, not here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.solvers.mesh.convergence_solver import (
        RichardsonGCI,
        _DummyGCI,
    )

__all__ = ["ConvergenceRecord", "MeshRefinementResult"]


@dataclass
class ConvergenceRecord:
    """
    One interval's (one gear face width, one bearing extent, one
    distributed load span, ...) convergence history -- every grade
    level tried, the point metric at each level, the GCI computed once
    3+ levels exist, and whether it converged before max_levels ran out.

    ## CHANGED: point_metrics_history/gci_history keyed by metric NAME
    ## (e.g. "v_xz", "v_xy", "v_res") rather than a fixed (f_xz, f_xy,
    ## f_res) triple -- which metric names actually appear depends
    ## entirely on which MetricSpec list MeshConvergenceStudy was
    ## constructed with (see axisforge/solvers/mesh/metric_spec.py).
    ## A record built from the default displacement-only MetricSpec set
    ## has the exact same three keys ("v_xz"/"v_xy"/"v_res" instead of
    ## the old bare "xz"/"xy"/"res") and the exact same values as before
    ## this change -- only the key strings differ, not the numbers.

    ## CHANGED (earlier pass, confirmed with erg 2026-09-17): `applicable`
    ## and `skip_reason` fields REMOVED -- see this module's own
    ## top-of-file CHANGED note. Every record now reflects a real solve
    ## attempt; `converged` is the only outcome field report/print code
    ## needs to check.

    ## CHANGED (this pass, confirmed with erg 2026-09-17): `n_points` and
    ## `point_x` fields REMOVED -- they were only ever populated by the
    ## now-deleted MomentConvergenceStudy. A record built by
    ## MeshConvergenceStudy (the only study left that produces
    ## per-interval ConvergenceRecords) never used either field.
    """

    label: str
    x_lo: float
    x_hi: float

    levels: list[str] = field(default_factory=list)

    point_metrics_history: list[dict[str, float]] = field(default_factory=list)
    """Per level: {metric_name: aggregated_value}, e.g.
    {"v_xz": ..., "v_xy": ..., "v_res": ...} for the default displacement
    metric set. Was: tuple[float, float, float] fixed to
    (f_xz, f_xy, f_res)."""

    gci_history: list[dict[str, "RichardsonGCI | _DummyGCI"]] = field(default_factory=list)
    """Per level transition (once len(levels) >= 3): {metric_name: ...},
    each value a RichardsonGCI (successful fit) or _DummyGCI (non-uniform
    refinement ratio, a degenerate observed order, or a metric landing
    exactly on 0.0 -- see convergence_solver.py's own RichardsonGCI/
    _compute_gci()). Type hint is unchanged from before this pass -- it
    was already an open dict[str, ...]; only the key set it's populated
    with widened."""

    peak_shifted: dict[str, list[bool]] = field(default_factory=dict)
    """Per metric name that uses a peak-tracking evaluation strategy: one
    bool per transition in gci_history, True if that metric's sampled
    extremum moved more than its strategy's stable_tol_mm between the
    two grades feeding that transition. Absent, or not meaningfully
    populated, for metrics sampled at a fixed geometric point (today's
    displacement metrics) -- FixedPointStrategy never repositions, so
    drift is not a concept that applies to them.

    Exists because RichardsonGCI's order-of-convergence math assumes the
    same physical location is being compared across grades; a metric
    whose peak is still migrating can report GCI_f_m < threshold (i.e.
    "converged") while actually only meaning "the value AT A DRIFTING
    LOCATION stopped changing much," which is a materially weaker claim.
    A future convergence_report.py should render this (e.g. flag the row)
    rather than let that distinction disappear into a bare True/False
    converged column."""

    converged: bool = False
    x_final: list[float] = field(default_factory=list)

    def metric_names(self) -> list[str]:
        """Metric names present in this record, in first-seen order --
        convenience for report/print code that needs to iterate an open
        key set instead of a fixed ("xz", "xy", "res")."""
        if not self.point_metrics_history:
            return []
        return list(self.point_metrics_history[0].keys())


@dataclass
class MeshRefinementResult:
    """
    Aggregated study result for ONE shaft -- one ConvergenceRecord per
    interval (gear face width, bearing extent, distributed load span,
    or any custom region passed to MeshConvergenceStudy.run()).
    """

    shaft_name: str = ""
    per_load: dict[str, ConvergenceRecord] = field(default_factory=dict)

    @property
    def all_extra_nodes(self) -> list[float]:
        """Union of all x_final sets across every interval, sorted --
        ready to pass to Mesh1D(..., extra_mandatory=this).

        ## NOTE (unchanged behaviour, flagged not fixed): this still
        ## includes x_final from intervals where rec.converged is False
        ## -- MeshConvergenceStudy._converge_one_load()'s own fallback
        ## ("if history: rec.x_final = ...") sets x_final to the last
        ## grade attempted even on failure, and this property doesn't
        ## distinguish that from a genuine converged mesh. That was
        ## already true before this pass; worth deciding deliberately
        ## once other metrics are in play, since "v converged, but its
        ## last-attempted nodes got folded into extra_mandatory anyway"
        ## is exactly the kind of silent averaging-away this whole
        ## change is meant to stop doing."""
        nodes: set[float] = set()
        for rec in self.per_load.values():
            nodes.update(rec.x_final)
        return sorted(nodes)