"""Orquestrador ISO/TS 16281. Agrupa bearings pelo solver que
dispatch.resolve_solver_cls() lhes atribui e junta os resultados em
{label: [LoadDistributionResult, ...]}, na ordem de `bearings`."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import SimpleFEMResultsLibrary
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import BearingResultsLibrary, warn_if_floating_loaded
from axisforge.solvers.machine_elements.bearings.ISO_16281.dispatch import (
    resolve_solver_cls, resolve_solver_cls_for_attrs, SolverDispatchError,
)

if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import BallLoadDistributionResult
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import RollerLoadDistributionResult
    LoadDistributionResult = BallLoadDistributionResult | RollerLoadDistributionResult

from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_solver import ISO16281BallSolver
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import BallBearingResult
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing import ball_bearing_postprocessing as _ball_pp
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing import (
    ball_bearing_multirow_solver as _ball_multirow_solver,  # noqa: F401 -- regista ISO16281BallSolver.MULTIROW_SOLVER
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_solver import ISO16281RollerSolver
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import RollerBearingResult
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing import roller_bearing_postprocessing as _roller_pp
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing import (
    roller_bearing_multirow_solver as _roller_multirow_solver,  # noqa: F401 -- regista ISO16281RollerSolver.MULTIROW_SOLVER
)
from axisforge.config import SOLVER_TOLERANCE


def _is_multirow(bearing: Bearing) -> bool:
    rows = getattr(bearing, "rows", None)
    return bool(rows) and len(rows) >= 2


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
    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol = tol
        self.psi_input = psi_input

    def _row_solver_cls_for(self, bearing: Bearing, label: str) -> type:
        """Solver single-row cujo contrato as rows desta bearing satisfazem
        -- usado para dispatch normal E para o adapter de postprocessing de
        uma bearing multi-fila (o adapter e' do tipo de contacto, nao do
        solver multi-fila)."""
        if _is_multirow(bearing):
            return resolve_solver_cls_for_attrs(set(bearing.rows[0]), label=f"{label}[row0]")
        return resolve_solver_cls(bearing, label=label)

    def _multirow_solver_cls_for(self, bearing: Bearing, label: str) -> type:
        row_cls = self._row_solver_cls_for(bearing, label)
        if row_cls.MULTIROW_SOLVER is None:
            raise SolverDispatchError(
                f"'{label}': rows batem em {row_cls.__name__} mas ele nao tem MULTIROW_SOLVER"
            )
        return row_cls.MULTIROW_SOLVER

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              psi_override: dict[str, float] | None = None,
              results: BearingResultsLibrary | None = None,
              ) -> dict[str, list["LoadDistributionResult"]]:
        single_row, multi_row = {}, {}
        for label, b in bearings.items():
            (multi_row if _is_multirow(b) else single_row)[label] = b

        merged: dict[str, list["LoadDistributionResult"]] = {}

        groups: dict[type, dict[str, Bearing]] = {}
        for label, b in single_row.items():
            groups.setdefault(resolve_solver_cls(b, label=label), {})[label] = b

        for solver_cls, group in groups.items():
            solver = solver_cls(tol=self.tol, psi_input=self.psi_input)
            group_override = (
                {lbl: psi_override[lbl] for lbl in group if lbl in psi_override}
                if psi_override else None
            )
            local_lib = solver.solve(shaft_system, group, library, psi_override=group_override)

            if results is not None:
                results.add_load_distribution_library(solver_cls, local_lib, group)

            for label in group:
                merged[label] = local_lib.get(label).rows

        if multi_row:
            shaft_results = library.get(shaft_system.name)
            node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

            for label, b in multi_row.items():
                row_cls   = self._row_solver_cls_for(b, label)
                mr_solver = self._multirow_solver_cls_for(b, label)(tol=self.tol)

                node = node_by_label[label]
                Fr_xz, Fr_xy, Fa = node.Fr_xz, node.Fr_xy, node.Fa
                phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))
                warn_if_floating_loaded(b, label, Fa)

                if self.psi_input and psi_override is not None and label in psi_override:
                    psi = float(psi_override[label])
                else:
                    psi = node.psi_xz * np.cos(phi_Fr) + node.psi_xy * np.sin(phi_Fr)

                solve_kwargs = dict(Fr_xz=Fr_xz, Fr_xy=Fr_xy, psi=psi, label=label)
                if row_cls is ISO16281BallSolver:
                    solve_kwargs["Fa"] = Fa

                mr_result = mr_solver.solve_bearing(b, **solve_kwargs)
                merged[label] = mr_result.rows

                if results is not None:
                    results.set_load_distribution(label, mr_result.rows, bearing_type=b.bearing_type)
                    results.set_extra(label, "multirow_result", mr_result)

        return {label: merged[label] for label in bearings}

    def postprocess_and_record(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              catalog: dict[str, dict],
              load_distribution: dict[str, list["LoadDistributionResult"]] | None = None,
              results: BearingResultsLibrary | None = None,
              psi_override: dict[str, float] | None = None,
              ) -> BearingResultsLibrary:
        results = results or BearingResultsLibrary()

        if load_distribution is None:
            load_distribution = self.solve(shaft_system, bearings, library,
                                           psi_override=psi_override, results=results)
        else:
            for label, b in bearings.items():
                results.set_load_distribution(label, load_distribution[label], bearing_type=b.bearing_type)

        shaft_results = library.get(shaft_system.name)
        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        for label, b in bearings.items():
            row_results = load_distribution[label]
            row_cls = self._row_solver_cls_for(b, label)
            adapter = _POSTPROC.get(row_cls)
            if adapter is None:
                raise NotImplementedError(
                    f"Sem postprocessing registado para {row_cls.__name__} (bearing: '{label}')"
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