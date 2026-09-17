"""
axisforge/results/fem_results/submodel_results.py

Pure-data, POST-postprocessing result for one submodel solve (one grade,
one interval). Built by SubmodelConvergencePostProcessing.process()
(sub_models/postprocessing.py) FROM a SubmodelSolution
(sub_models/lagrange_multipliers.py's raw solver output) -- never built
directly by SubmodelSolver.solve() itself. Mirrors the split already
used for the full-shaft pipeline: RigidSupportFEMSolver (solver) ->
Timoshenko/EulerBernoulliPostProcessing (postprocessing) -> ShaftResults
(results/); this is that same split one level down, for a single
submodel grade instead of a whole shaft.

MeshConvergenceStudy (convergence_solver.py) is the consumer: instead of
extracting metric values itself from a raw SubmodelSolution, it now
calls SubmodelConvergencePostProcessing.process(solution, metrics, ...)
and reads the returned SubmodelResult.metric_values -- the convergence
solver never touches d_xz/d_xy/elements directly anymore, only this
already-postprocessed shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SubmodelResult"]


@dataclass
class SubmodelResult:
    """
    One grade's postprocessed metric values for one interval.

    x_lo, x_hi, grade : identity, carried over from the SubmodelSolution
        this was built from.
    x_nodes : node positions of the submodel mesh at this grade -- kept
        (not just metric_values) because RichardsonGCI needs node COUNT
        per grade (x_nodes_coarse/medium/fine) to compute the refinement
        ratio r, not just the metric value itself.
    metric_values : {metric_name: aggregated_value}, e.g.
        {"v_xz": ..., "v_xy": ..., "v_res": ..., "M_xz": ..., ...} --
        whichever MetricSpecs were passed to process(). Open key set,
        same reasoning as ConvergenceRecord.point_metrics_history in
        convergence_results.py.
    """
    x_lo: float
    x_hi: float
    grade: str
    x_nodes: list[float] = field(default_factory=list)
    metric_values: dict[str, float] = field(default_factory=dict)