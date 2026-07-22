"""
axisforge/mesh/oneD/shaft/mesh_generation/mesh_convergence_study.py

Programa de ESTUDO — corre uma análise sequencial de refinamento de malha
para cada distributed_radial_load do ShaftSystem, e devolve as posições x
em que convergiu M(x)/sigma_b(x). Não faz parte do pipeline de produção:
corre-se uma vez, guarda-se o resultado, e o Mesh1D de produção passa a
usar diretamente `MeshRefinementResult.all_extra_nodes` como extra_nodes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader


@dataclass
class ConvergenceRecord:
    label: str
    x_lo: float
    x_hi: float
    levels: list[list[float]] = field(default_factory=list)     # candidatos por nível
    metric_history: list[float] = field(default_factory=list)   # valor comparado por nível
    error_history: list[float] = field(default_factory=list)    # erro relativo entre níveis
    converged: bool = False
    x_final: list[float] = field(default_factory=list)


@dataclass
class MeshRefinementResult:
    per_load: dict[str, ConvergenceRecord] = field(default_factory=dict)

    @property
    def all_extra_nodes(self) -> list[float]:
        nodes: set[float] = set()
        for rec in self.per_load.values():
            nodes.update(rec.x_final)
        return sorted(nodes)

    def print_report(self, unit_label: str = "mm") -> None:
        """Relatório organizado por load: níveis, erro, nodes finais."""
        print("=" * 70)
        print("  MESH CONVERGENCE STUDY — relatório")
        print("=" * 70)
        for label, rec in self.per_load.items():
            status = "✓ convergiu" if rec.converged else "✗ não convergiu (usou nível mais fino)"
            print(f"\n[{label}]  x_lo={rec.x_lo:.3f} {unit_label}  x_hi={rec.x_hi:.3f} {unit_label}  {status}")
            for lvl, (pts, metric) in enumerate(zip(rec.levels, rec.metric_history)):
                err = rec.error_history[lvl - 1] if lvl > 0 else float("nan")
                pts_str = ", ".join(f"{p:.3f}" for p in pts)
                print(f"  nível {lvl}: metric={metric:.6g}  err={err:.4%}  nodes=[{pts_str}]")
            print(f"  -> nodes finais ({len(rec.x_final)}): "
                  f"[{', '.join(f'{p:.3f}' for p in rec.x_final)}]")

        print("\n" + "-" * 70)
        all_nodes = self.all_extra_nodes
        print(f"TOTAL de extra_nodes únicos ({len(all_nodes)}):")
        print(f"  {all_nodes}")
        print("\n>>> Copia esta linha para o teu código de produção:")
        print(f"EXTRA_NODES_MM = {all_nodes!r}")
        print("=" * 70)

class MeshConvergenceStudy:
    def __init__(
        self,        
        solver: "SimpleFEMSolver",
        tol: float = 1e-4,
        max_levels: int = 4,

        metric: str = "sigma_b",   # "sigma_b" | "M_max" | "l2_M"
    ):
        self._solver = solver
        self._theory = solver._builder.theory
        self._tol = tol
        self._max_levels = max_levels
        self._metric = metric

    def run(self, shaft_system) -> MeshRefinementResult:
        result = MeshRefinementResult()
        for ld in shaft_system.distributed_radial_loads:
            result.per_load[ld.label] = self._converge_one_load(shaft_system, ld)
        return result

    # ------------------------------------------------------------------
    def _converge_one_load(self, shaft_system, ld) -> ConvergenceRecord:
        rec = ConvergenceRecord(label=ld.label, x_lo=ld.x_lo, x_hi=ld.x_hi)
        candidates = [ld.x_lo, ld.x_hi]
        prev_metric = None

        for level in range(self._max_levels):
            candidates = self._bisect(candidates)
            metric_value = self._evaluate(shaft_system, candidates)

            rec.levels.append(list(candidates))
            rec.metric_history.append(metric_value)

            if prev_metric is not None:
                err = abs(metric_value - prev_metric) / max(abs(metric_value), 1e-12)
                rec.error_history.append(err)
                if err < self._tol:
                    rec.converged = True
                    rec.x_final = list(candidates)
                    return rec

            prev_metric = metric_value

        rec.x_final = list(candidates)  # não convergiu -> fica o nível mais fino tentado
        return rec

    # ------------------------------------------------------------------
    def _bisect(self, points: list[float]) -> list[float]:
        new_points = sorted(points)
        out = list(new_points)
        for a, b in zip(new_points, new_points[1:]):
            out.append((a + b) / 2.0)
        return sorted(set(out))

    def _evaluate(self, shaft_system, candidates: list[float]) -> float:
        mesh = Mesh1D(shaft_system)
        mesh.add_mandatory_positions(candidates)          # correct API
        solver = SimpleFEMSolver(theory=self._theory)
        solver.solve(shaft_system, mesh=mesh)
        reader = ShaftResultsReader(solver, shaft_system)
        results = reader.read()

        if self._metric == "M_max":
            return float(np.max(results.M))
        if self._metric == "sigma_b":
            return float(np.max(results.sigma_b))
        if self._metric == "l2_M":
            return float(np.linalg.norm(results.M))
        raise ValueError(f"unknown metric: {self._metric}")