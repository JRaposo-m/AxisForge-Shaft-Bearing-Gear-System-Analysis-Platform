"""axisforge/fixtures/studies/bearing/load_distribution/single_row/rolling_bearing_solver.py

Study-stage orchestrator for ISO/TS 16281 single-row load distribution.
Groups bearings by the solver dispatch.resolve_solver_cls() assigns them
(point-contact -> ISO16281BallSolver, line-contact -> ISO16281RollerSolver),
runs each group, and merges results into {label: [LoadDistributionResult]}.

Single-row only -- multi-row (THRUST) bearings are a separate concern, not
handled by this file. Lives in fixtures/studies/, not axisforge/solvers/:
it reads the shaft FEM registry (RigidBearingFEMResultsLibrary) and writes
the bearing results registry (BearingResultsLibrary) -- a solver may never
hold either, only a study can.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.fixtures.studies.shafts.results_library import RigidBearingFEMResultsLibrary
from axisforge.fixtures.studies.bearings.load_distribution.no_lubrication.results_library import BearingResultsLibrary
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.dispatch import resolve_solver_cls

if TYPE_CHECKING:
    from axisforge.results.bearings.load_distribution.single_row.ball_bearing_results import BallLoadDistributionResult
    from axisforge.results.bearings.load_distribution.single_row.roller_bearing_results import RollerLoadDistributionResult
    LoadDistributionResult = BallLoadDistributionResult | RollerLoadDistributionResult

from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.ball_bearing.solver import ISO16281BallSolver
from axisforge.results.bearings.load_distribution.single_row.ball_bearing_results import BallBearingResult
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.ball_bearing import postprocessing as _ball_pp
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.roller_bearing.solver import ISO16281RollerSolver
from axisforge.results.bearings.load_distribution.single_row.roller_bearing_results import RollerBearingResult
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.roller_bearing import postprocessing as _roller_pp
from axisforge.config import SOLVER_TOLERANCE


@dataclass(frozen=True)
class _PostprocAdapter:
    result_cls: type
    derel_cls: type
    stiffness_fn: Callable


def _ball_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9):
    return _ball_pp.bearing_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=eps)


def _roller_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9):
    return _roller_pp.bearing_stiffness(bearing, result, Fr_xz, Fr_xy, eps=eps)


_POSTPROC: dict[type, _PostprocAdapter] = {
    ISO16281BallSolver: _PostprocAdapter(
        BallBearingResult, _ball_pp.DynamicEquivalentRollingElementLoad, _ball_stiffness),
    ISO16281RollerSolver: _PostprocAdapter(
        RollerBearingResult, _roller_pp.LaminaDynamicEquivalentLoad, _roller_stiffness),
}


class RollingBearingSolver:
    """Dispatches single-row bearings to ISO16281BallSolver/ISO16281RollerSolver
    by CAPABILITY + REQUIRED_ATTRS, and postprocesses (capacity, dynamic
    equivalent load, stiffness) into a BearingResultsLibrary.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol       = tol
        self.psi_input = psi_input

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              shaft_library: RigidBearingFEMResultsLibrary,
              psi_override: dict[str, float] | None = None,
              results: BearingResultsLibrary | None = None,
              ) -> dict[str, list["LoadDistributionResult"]]:
        """Solve every bearing in `bearings`, grouped by dispatched solver class.

        results, if given, is populated via add_load_distribution_results()
        per group -- the caller's BearingResultsLibrary, not owned here.
        """
        shaft_results = shaft_library.get(shaft_system.name)

        groups: dict[type, dict[str, Bearing]] = {}
        for label, b in bearings.items():
            groups.setdefault(resolve_solver_cls(b, label=label), {})[label] = b

        merged: dict[str, list["LoadDistributionResult"]] = {}
        for solver_cls, group in groups.items():
            solver = solver_cls(tol=self.tol, psi_input=self.psi_input)
            group_override = (
                {lbl: psi_override[lbl] for lbl in group if lbl in psi_override}
                if psi_override else None
            )
            group_results = solver.solve(shaft_system, group, shaft_results, psi_override=group_override)

            if results is not None:
                results.add_load_distribution_results(solver_cls, group_results, group)

            for label, result in group_results.items():
                merged[label] = result.rows

        return {label: merged[label] for label in bearings}

    def postprocess_and_record(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              shaft_library: RigidBearingFEMResultsLibrary,
              catalog: dict[str, dict],
              load_distribution: dict[str, list["LoadDistributionResult"]] | None = None,
              results: BearingResultsLibrary | None = None,
              psi_override: dict[str, float] | None = None,
              ) -> BearingResultsLibrary:
        """Solve (if load_distribution not already given) then record capacity,
        dynamic equivalent load and stiffness for every bearing, per `catalog`.

        catalog: {label: {"capacity": {...kwargs...}, "dynamic_equivalent_load": {...kwargs...}}},
        both sub-keys optional per label.
        """
        results = results or BearingResultsLibrary()

        if load_distribution is None:
            load_distribution = self.solve(shaft_system, bearings, shaft_library,
                                           psi_override=psi_override, results=results)
        else:
            for label, b in bearings.items():
                results.set_load_distribution(label, load_distribution[label], bearing_type=b.bearing_type)

        shaft_results = shaft_library.get(shaft_system.name)
        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        for label, b in bearings.items():
            row_results = load_distribution[label]
            solver_cls  = resolve_solver_cls(b, label=label)
            adapter     = _POSTPROC.get(solver_cls)
            if adapter is None:
                raise NotImplementedError(
                    f"Sem postprocessing registado para {solver_cls.__name__} (bearing: '{label}')"
                )

            entry = catalog.get(label, {})

            cap_kwargs = dict(entry.get("capacity", {}))
            if cap_kwargs:
                results.set_capacity(label, b.family.per_element_dynamic_capacity(b, **cap_kwargs))

            wrapped = adapter.result_cls(rows=row_results)

            derel_kwargs = entry.get("dynamic_equivalent_load")
            if derel_kwargs is not None:
                results.set_dynamic_equivalent_load(
                    label, adapter.derel_cls.from_bearing_result(b, wrapped, **derel_kwargs))

            node = node_by_label[label]
            results.set_stiffness(label, adapter.stiffness_fn(b, wrapped, node.Fr_xz, node.Fr_xy, node.Fa))

        return results