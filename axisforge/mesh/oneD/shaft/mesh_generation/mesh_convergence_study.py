"""
axisforge/mesh/oneD/shaft/mesh_generation/mesh_convergence_study.py

Mesh convergence study for distributed radial loads.

Strategy  : SubmodelSolver with successive grades (grade_0, grade_1, ...)
Criterion : Richardson extrapolation + Grid Convergence Index (GCI).

Convergence metric : resultant transverse displacement
                     v = sqrt(v_xz^2 + v_xy^2) per node.

References
----------
Roache, P.J. (1998). Verification and Validation in Computational Science and Engineering.
Richardson, L.F. (1911). Phil. Trans. R. Soc. London A, 210, 307-357.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.submodel_solver import SubmodelSolver, SubmodelResult


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass
class ConvergenceRecord:
    """Full convergence history for a single distributed_radial_load."""

    label: str
    x_lo: float
    x_hi: float

    levels: list[str] = field(default_factory=list)
    """Grade strings evaluated per level (e.g. 'grade_0', 'grade_1', ...)."""

    point_metrics_history: list[list[float]] = field(default_factory=list)
    """Resultant deflection v = sqrt(v_xz^2 + v_xy^2) per node per level."""

    gci_history: list[list[float]] = field(default_factory=list)
    """Per-interval GCI values per level (available from level 2 onwards)."""

    converged: bool = False
    x_final: list[float] = field(default_factory=list)


@dataclass
class MeshRefinementResult:
    """Aggregated study result — one ConvergenceRecord per load."""

    per_load: dict[str, ConvergenceRecord] = field(default_factory=dict)

    @property
    def all_extra_nodes(self) -> list[float]:
        """Union of all x_final sets, sorted — ready to pass to Mesh1D."""
        nodes: set[float] = set()
        for rec in self.per_load.values():
            nodes.update(rec.x_final)
        return sorted(nodes)

    def print_report(self, unit_label: str = "mm") -> None:
        """Print a per-load report: levels, GCI per interval, and final nodes."""
        print("=" * 70)
        print("  MESH CONVERGENCE STUDY — report  [Richardson GCI]")
        print("=" * 70)
        for label, rec in self.per_load.items():
            status = "converged" if rec.converged else "did not converge (finest level used)"
            print(f"\n[{label}]  x_lo={rec.x_lo:.3f} {unit_label}  "
                  f"x_hi={rec.x_hi:.3f} {unit_label}  {status}")
            for lvl, grade in enumerate(rec.levels):
                print(f"  level {lvl}: {grade}")
                if lvl < len(rec.gci_history):
                    gcis = rec.gci_history[lvl]
                    gci_str = ", ".join(f"{g:.4%}" for g in gcis)
                    print(f"           GCI  =[{gci_str}]")
            print(f"  -> final nodes ({len(rec.x_final)}): "
                  f"[{', '.join(f'{p:.3f}' for p in rec.x_final)}]")

        print("\n" + "-" * 70)
        all_nodes = self.all_extra_nodes
        print(f"TOTAL unique extra_nodes ({len(all_nodes)}):")
        print(f"  {all_nodes}")
        print("\n>>> Copy this line into your production code:")
        print(f"EXTRA_NODES_MM = {all_nodes!r}")
        print("=" * 70)


# ===========================================================================
# Richardson GCI
# ===========================================================================

class RichardsonGCI:
    """
    Grid Convergence Index based on Richardson extrapolation.

    Requires a minimum of 3 consecutive refinement levels to compute
    the observed order of convergence p and the GCI per interval.

    Metric: resultant transverse displacement v = sqrt(v_xz^2 + v_xy^2) per node.

    Parameters
    ----------
    gci_threshold : fractional error threshold for convergence (default 0.01 = 1%)
    safety_factor : Fs — 1.25 if p is verified in asymptotic range, 3.0 otherwise
    r             : refinement ratio between levels (2.0 for bisection)
    min_p         : lower clamp on observed order — guards against coarse-level noise
    max_p         : upper clamp on observed order — guards against super-convergence artefacts
    """

    def __init__(self,
                 f_coarse: float,
                 f_medium: float,
                 f_fine: float,
                 x_nodes_coarse: list[float],
                 x_nodes_medium: list[float],
                 x_nodes_fine: list[float],
                 gci_threshold: float = 0.01,
                 safety_factor: float = 1.25):

        self.x_nodes_coarse = x_nodes_coarse
        self.x_nodes_medium = x_nodes_medium
        self.x_nodes_fine   = x_nodes_fine
        self.gci_threshold  = gci_threshold
        self.safety_factor  = safety_factor
        self.f_coarse       = f_coarse
        self.f_medium       = f_medium
        self.f_fine         = f_coarse

        self.r_m_c          = (len(x_nodes_medium) - 1) / (len(x_nodes_coarse) - 1)
        self.r_f_m          = (len(x_nodes_fine) - 1) / (len(x_nodes_medium) - 1)

        if self.r_f_m == self.r_m_c:
            self.r = self.r_f_m
        # else:
            # depois aqui preciso de ver o que fazer, provavelmente um raise ValueError


        # here the function is capable of obtaining the convergence of
        # the interval by taking into consideration an value f the user must define


        self.p = np.log((self.f_fine - self.f_medium)/(
            self.f_medium - self.f_coarse)) / np.log(self.r)

        self.e_m_c = (self.f_medium - self.f_coarse) / self.f_coarse
        self.e_f_m = (self.f_fine - self.f_medium) / self.f_medium

        self.GCI_m_c = self.safety_factor * abs(self.e_m_c) / (
            (self.r_m_c**self.p) - 1)
        self.GCI_f_m = self.safety_factor * abs(self.e_f_m) / (
            (self.r_f_m**self.p) - 1)

    def compute_gci(
        self,
        pts_fine:    list[float],
        vals_fine:   list[float],
        pts_med:     list[float],
        vals_med:    list[float],
        pts_coarse:  list[float],
        vals_coarse: list[float],
    ) -> list[float]:
        """
        Compute per-interval GCI from three consecutive refinement levels.

        Levels must satisfy: coarse ⊂ medium ⊂ fine (bisection construction).
        GCI is evaluated at points shared across all three levels,
        then mapped to per-interval values on the fine grid.
        Intervals without shared ancestor points receive GCI = inf.

        Returns
        -------
        gci_per_interval : list of length len(pts_fine) - 1
        """
        ...

    def all_converged(self, gci_per_interval: list[float]) -> bool:
        """True if ALL intervals satisfy GCI < gci_threshold."""
        ...

    def unconverged_intervals(self, gci_per_interval: list[float]) -> list[int]:
        """Indices of intervals where GCI >= gci_threshold."""
        ...

    def _gci_at_point(self, f1: float, f2: float, f3: float) -> float:
        """
        GCI at a single point from three refinement levels (fine, medium, coarse).

        Returns 0.0 if no change between levels (converged by inspection).
        Returns inf if convergence is oscillatory (not in asymptotic range).
        """
        ...

    @staticmethod
    def _build_map(pts: list[float], vals: list[float]) -> dict[float, float]:
        """Map rounded x-positions to metric values for O(1) lookup."""
        ...


# ===========================================================================
# Orchestrator
# ===========================================================================

class MeshConvergenceStudy:
    """
    Runs the mesh refinement study for all distributed_radial_loads
    in a ShaftSystem.

    Strategy  : SubmodelSolver with successive grades (grade_0, grade_1, ...)
    Criterion : RichardsonGCI on resultant deflection v = sqrt(v_xz^2 + v_xy^2).
                ALL intervals must satisfy GCI < gci_threshold.

    The global_solver must already be solved before calling run().

    Parameters
    ----------
    global_solver : solved SimpleFEMSolver (solution used as BCs for submodels)
    gci_threshold : fractional GCI threshold (default 0.01 = 1%)
    safety_factor : GCI safety factor Fs (3.0 general, 1.25 if p is verified)
    max_levels    : maximum grade levels before giving up
    """

    _MIN_LEVELS_FOR_GCI = 3  # Richardson requires 3 evaluations minimum

    def __init__(
        self,
        global_solver: SimpleFEMSolver,
        gci_threshold: float = 0.01,
        safety_factor: float = 3.0,
        max_levels: int = 8,
    ): ...

    def run(self, shaft_system) -> MeshRefinementResult:
        """
        Run the study for all distributed_radial_loads in shaft_system.
        global_solver must already be solved.
        """
        ...

    def _converge_one_load(
        self,
        shaft_system,
        ld,
    ) -> ConvergenceRecord:
        """
        Runs SubmodelSolver for grade_0, grade_1, grade_2, ...
        until RichardsonGCI converges or max_levels is reached.

        Levels 0-1 : collect results (bootstrap — Richardson needs 3 levels).
        Level 2+   : compute GCI, check convergence.
        """
        ...

    def _extract_metric(
        self,
        result: SubmodelResult,
    ) -> list[float]:
        """
        Extract resultant transverse displacement per node from SubmodelResult.

        v_i = sqrt(v_xz_i^2 + v_xy_i^2)  for each node i in the submodel.

        Returns
        -------
        list of length len(result.x_nodes)
        """
        ...