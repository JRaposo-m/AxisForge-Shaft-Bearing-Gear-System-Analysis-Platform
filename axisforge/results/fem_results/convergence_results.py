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

ConvergenceRecord.gci_history still holds RichardsonGCI/_DummyGCI
instances directly (a dict of {"xz": ..., "xy": ..., "res": ...} per
level transition) -- those two classes were NOT converted into a
results-side dataclass (GCIResult) in this pass; that would have meant
touching RichardsonGCI's own call sites in mesh_convergence_study.py,
which is explicitly staying untouched right now. So this module keeps
a TYPE_CHECKING-only reference to them for the type hint, and otherwise
never constructs or reads their internals -- it only stores whatever
mesh_convergence_study.py's own _compute_gci() hands it.

print_report() was REMOVED from MeshRefinementResult on the move -- a
results dataclass printing itself to stdout is the same data/
presentation mixing every other report in this platform has been
pulled apart into a *_report.py content-block module (see
resolution_report.py/comparison_report.py's own "CONTENT ONLY"
docstrings). That formatting logic is not lost, just not moved yet --
whoever builds fem_studies/outputs/convergence_report.py next should
treat MeshRefinementResult.print_report()'s old body as the starting
point for that module's content-block functions, not reintroduce a
print() method here.

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
  axisforge.mesh.shaft.mesh_generation.mesh_convergence_study
      RichardsonGCI, _DummyGCI -- referenced by type only; this module
      never imports them at runtime, never constructs one, and never
      reads a field off one that isn't already a dict key
      (rec.gci_history[i]["xz"].GCI_f_m etc. is done by report/print
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
    """

    label: str
    x_lo: float
    x_hi: float

    levels: list[str] = field(default_factory=list)

    point_metrics_history: list[tuple[float, float, float]] = field(default_factory=list)
    """Per level: (f_xz, f_xy, f_res)"""

    gci_history: list[dict[str, "RichardsonGCI | _DummyGCI"]] = field(default_factory=list)
    """Per level transition (once len(levels) >= 3): {'xz': ..., 'xy': ..., 'res': ...},
    each value a RichardsonGCI (successful fit) or _DummyGCI (non-uniform
    refinement ratio -- see mesh_convergence_study.py's own _compute_gci())."""

    converged: bool = False
    x_final: list[float] = field(default_factory=list)


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
        ready to pass to Mesh1D(..., extra_mandatory=this)."""
        nodes: set[float] = set()
        for rec in self.per_load.values():
            nodes.update(rec.x_final)
        return sorted(nodes)