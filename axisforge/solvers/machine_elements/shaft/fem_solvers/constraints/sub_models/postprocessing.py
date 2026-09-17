"""
axisforge/solvers/machine_elements/shaft/fem_solvers/sub_models/postprocessing.py

Postprocessing for the convergence study: SubmodelSolver
(lagrange_multipliers.py, same directory) DOES the submodel solve and
returns a raw SubmodelSolution -- this module turns that into a
SubmodelResult (results/fem_results/submodel_results.py), which is all
MeshConvergenceStudy (convergence_solver.py) consumes -- it never reads
a SubmodelSolution's raw arrays directly.

## CHANGED (this pass, confirmed with erg 2026-09-17): this module is
## simplified down to ONLY the "field" extraction path (per-node array
## via spec.field_fn, sampled at spec.eval_strategy.points(...), reduced
## with spec.aggregate) -- i.e. what v_xz/v_xy/v_res have always used.
## Everything that only ever served the OTHER two MetricSpec.kind
## values, "moment_centroid" and "moment_point", is REMOVED:
##   - the kind dispatch itself (process() no longer branches on
##     spec.kind at all -- there is only one extraction path left, so
##     there is nothing left to dispatch between)
##   - _postproc_for() / the TimoshenkoPostProcessing/
##     EulerBernoulliPostProcessing dispatch-by-theory machinery (that
##     dispatch existed ONLY to pick the right bending_moment()
##     implementation for moment recovery -- "field" extraction reads
##     solution.d_xz/d_xy directly and never needed it)
##   - original_element_midpoints() / moment_at_original_centroids() /
##     _element_nodal_displacements() / _owning_element() (the fixed
##     original-element-centroid M recovery machinery MomentConvergenceStudy
##     depended on)
##   - the moment_list()/moment_cache closure inside process()
##
## Reasoning: MomentConvergenceStudy (convergence_solver.py) and its
## metric builders (default_moment_metrics()/build_moment_point_metrics(),
## metric_spec.py) are deleted -- per erg: "antes tinha posto a criar
## pontos de analise e assim e agora isso vai ser retirado porque esta
## abordagem do global gostei mesmo muito". M convergence now lives
## entirely in the whole-shaft global study
## (global_convergence_study.py), which reads ShaftResults.M/M_xz/M_xy
## directly (already computed by the production TimoshenkoPostProcessing/
## EulerBernoulliPostProcessing path at each grade's global re-solve) --
## it never goes through SubmodelConvergencePostProcessing or this
## module's own moment-at-centroid recovery at all. So nothing in the
## codebase constructs a MetricSpec with kind="moment_centroid"/
## "moment_point" anymore, and metric_spec.py's `kind`/`plane`/
## `point_index`/`point_x` fields are removed to match (see that
## module's own CHANGED note) -- this file's `process()` no longer reads
## spec.kind at all, so there is no risk of an AttributeError from that
## removal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from axisforge.results.fem_results.submodel_results import SubmodelResult

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.sub_models.lagrange_multipliers import (
        SubmodelSolution,
    )
    from axisforge.solvers.mesh.metric_spec import MetricSpec


class SubmodelConvergencePostProcessing:
    """
    Reads a SubmodelSolution (one per grade, from SubmodelSolver.solve())
    and a list[MetricSpec], and produces the SubmodelResult
    MeshConvergenceStudy consumes. Every MetricSpec is now a "field"
    metric (see this module's own top-of-file CHANGED note): sample
    spec.field_fn(solution) at spec.eval_strategy.points(...), reduce
    with spec.aggregate. Today's v_xz/v_xy/v_res are the only metrics
    built anywhere in the codebase (default_displacement_metrics(),
    metric_spec.py).
    """

    # ------------------------------------------------------------------
    # Entry point -- SubmodelSolution (+ MetricSpecs) -> SubmodelResult
    # ------------------------------------------------------------------

    def process(
        self,
        solution: "SubmodelSolution",
        metrics: list["MetricSpec"],
        shaft_system,
        grade0_x_nodes: list[float],
        prev_solution: "SubmodelSolution | None",
    ) -> SubmodelResult:
        """
        Builds the SubmodelResult for one grade. `prev_solution` is the
        previous grade's SubmodelSolution (or None on grade_0) -- passed
        through to each spec's eval_strategy.points() unchanged from how
        MeshConvergenceStudy._converge_one_load() already threads the
        "latest_result" argument. `grade0_x_nodes` is accepted for
        interface compatibility with that caller but unused here now --
        it was only ever needed by the removed moment-centroid recovery
        (original_element_midpoints() fixed its centroid list from it).
        """
        values: dict[str, float] = {}

        for spec in metrics:
            x_evals = spec.eval_strategy.points(
                shaft_system, solution.x_lo, solution.x_hi,
                spec.field_fn, prev_solution,
            )
            sampled = []
            for x in x_evals:
                i = self._find_node_index(solution.x_nodes, x)
                sampled.append(spec.field_fn(solution)[i])
            values[spec.name] = spec.aggregate(np.asarray(sampled))

        return SubmodelResult(
            x_lo=solution.x_lo, x_hi=solution.x_hi, grade=solution.grade,
            x_nodes=solution.x_nodes, metric_values=values,
        )

    @staticmethod
    def _find_node_index(x_nodes, x):
        from axisforge.mesh.shaft.element_type.elem import Elem
        return Elem.find_node_index(x_nodes, x)