"""
axisforge/solvers/mesh/convergence_solver.py

Mesh convergence study for distributed radial loads.

Strategy  : SubmodelSolver with successive grades (grade_0, grade_1, ...)
Criterion : Richardson extrapolation + Grid Convergence Index (GCI).

Convergence metric : resultant transverse displacement
                     v = sqrt(v_xz^2 + v_xy^2) per node.

Moved here from axisforge/mesh/shaft/mesh_generation/mesh_convergence_study.py
-- this is the solver side of the study, so it lives in solvers/ (under a
new mesh/ sub-package) rather than mesh/shaft/mesh_generation/, the same
way RigidBearingFEMSolver lives in solvers/ rather than results/.

RichardsonGCI/_DummyGCI/MeshConvergenceStudy stay here -- they DO
calculation, they are this study's solver side, same reasoning
RigidBearingFEMSolver stays in solvers/ rather than results/.
ConvergenceRecord and MeshRefinementResult moved OUT to
axisforge/results/fem_results/convergence_results.py -- pure data,
same split already applied to ShaftResults; see that module's own
docstring for the full reasoning (including why MeshRefinementResult's
old print_report() was dropped on the move, and why shaft_name was
added to it).

References
----------
Roache, P.J. (1998). Verification and Validation in Computational Science and Engineering.
Richardson, L.F. (1911). Phil. Trans. R. Soc. London A, 210, 307-357.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing import RigidBearingFEMSolver
from axisforge.solvers.machine_elements.shaft.fem_solvers.sub_models import SubmodelSolver, SubmodelResult
from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.core.loads import DistributedRadialLoad, LoadPlane
from axisforge.config import MIN_FACE_WIDTH_FOR_CONVERGENCE_MM, SOLVER_TOLERANCE
from axisforge.results.fem_results.convergence_results import (
    ConvergenceRecord,
    MeshRefinementResult,
)


# ===========================================================================
# GCI placeholder for the non-uniform-refinement-ratio case
# ===========================================================================

@dataclass
class _DummyGCI:
    GCI_f_m:   float = float("nan")
    GCI_m_c:   float = float("nan")
    converged: bool  = False


# ===========================================================================
# Richardson GCI
# ===========================================================================

class RichardsonGCI:
    """
    Grid Convergence Index based on Richardson extrapolation.

    Requires a minimum of 3 consecutive refinement levels to compute
    the observed order of convergence p and the GCI per interval.

    Metric: resultant transverse displacement v = sqrt(v_xz^2 + v_xy^2) on the centroid
    of the application in question

    If it is an distributed load there will automatically be a node in the centroid
    and in the center if it is a gear force. If it is a bearing there will also be a
    node in the center.

    The point is that there must be an analysis in importante points
        - if there is an machine element in the interval the convergence should be
        for the mid point or the medium displacement
        - if there's only a distributed load it should be in the centroid
        - if it is a complex case it should be the media and the points of interest

    Parameters
    ----------
    f_coarse        : metric value at coarse level
    f_medium        : metric value at medium level
    f_fine          : metric value at fine level
    x_nodes_coarse  : node positions at coarse level
    x_nodes_medium  : node positions at medium level
    x_nodes_fine    : node positions at fine level
    gci_threshold   : fractional error threshold for convergence (default 0.01 = 1%)
    safety_factor   : Fs — 1.25 if p is verified in asymptotic range, 3.0 otherwise
    min_p           : lower clamp on observed order — guards against coarse-level noise
    max_p           : upper clamp on observed order — guards against super-convergence artefacts
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
        self.f_fine         = f_fine

        self.r_m_c = (len(x_nodes_medium) - 1) / (len(x_nodes_coarse) - 1)
        self.r_f_m = (len(x_nodes_fine)   - 1) / (len(x_nodes_medium) - 1)

        if self.r_m_c <= 1.0:
            raise ValueError(
                f"Refinement ratio r_m_c must be > 1. Got {self.r_m_c:.4f}. "
                f"Medium mesh must be finer than coarse mesh."
            )
        if self.r_f_m <= 1.0:
            raise ValueError(
                f"Refinement ratio r_f_m must be > 1. Got {self.r_f_m:.4f}. "
                f"Fine mesh must be finer than medium mesh."
            )
        if self.r_f_m != self.r_m_c:
            raise ValueError(
                f"Non-uniform refinement ratio: r_f_m={self.r_f_m:.4f} != "
                f"r_m_c={self.r_m_c:.4f}. "
                f"Grades must be generated by uniform bisection."
            )

        self.r = self.r_f_m

        # observed order of convergence
        self.p = float(
            np.log((self.f_coarse - self.f_medium) /
                   (self.f_medium - self.f_fine)) / np.log(self.r)
        )


        self.e_m_c = (self.f_medium - self.f_coarse) / self.f_coarse
        self.e_f_m = (self.f_fine   - self.f_medium) / self.f_medium

        self.f_h0 = self.f_coarse + (self.f_coarse - self.f_medium) / (self.r**self.p - 1)

        self.GCI_m_c = (self.safety_factor * abs(self.e_m_c) / (self.r_m_c**self.p - 1))
        self.GCI_f_m = (self.safety_factor * abs(self.e_f_m) / (self.r_f_m**self.p - 1))

        self.converged_m_c = self.GCI_m_c < self.gci_threshold
        self.converged_f_m = self.GCI_f_m < self.gci_threshold
        self.converged     = self.converged_f_m and self.converged_m_c


# ===========================================================================
# Orchestrator
# ===========================================================================

class MeshConvergenceStudy:
    """
    Runs the mesh refinement study for all the intervals selected
    in a ShaftSystem.

    Must be the orchestrator of this whole process for the convergence study
    calling this activates all the submodeling and convergence studies

    Strategy  : SubmodelSolver with successive grades (grade_0, grade_1, ...)
    Criterion : RichardsonGCI on resultant deflection v = sqrt(v_xz^2 + v_xy^2).
                ALL intervals must satisfy GCI < gci_threshold.

    The global_solver must already be solved before calling run().

    Parameters
    ----------
    global_solver : solved RigidBearingFEMSolver (solution used as BCs for submodels)
    gci_threshold : fractional GCI threshold (default 0.01 = 1%)
    safety_factor : GCI safety factor Fs (3.0 general, 1.25 if p is verified)
    max_levels    : maximum grade levels before giving up
    """

    _MIN_LEVELS_FOR_GCI = 3  # Richardson requires 3 evaluations minimum

    def __init__(
        self,
        global_solver: RigidBearingFEMSolver,
        gci_threshold: float = 0.01,
        safety_factor: float = 1.25,
        max_levels: int = 8,
    ):
        self._global_solver = global_solver
        self._gci_threshold = gci_threshold
        self._safety_factor = safety_factor
        self._max_levels    = max_levels

    def run(self,
            shaft_system,
            intervals: list[tuple[float, float, str]]) -> MeshRefinementResult:
        """
        Run the convergence study for the specified intervals.

        Parameters
        ----------
        shaft_system : full ShaftSystem
        intervals    : list of (x_lo, x_hi, label) — any zone of interest,
                       not limited to distributed radial loads.
                       e.g. bearing zones, gear face widths, custom regions.
        """
        if self._global_solver.d_total_xz is None:
            raise RuntimeError(
                "MeshConvergenceStudy.run() requires a solved RigidBearingFEMSolver. "
                "Call global_solver.solve(shaft_system) first."
            )

        result = MeshRefinementResult(shaft_name=getattr(shaft_system, "name", ""))
        for x_lo, x_hi, label in intervals:
            result.per_load[label] = self._converge_one_load(
                shaft_system, x_lo, x_hi, label
            )
        return result

    def _converge_one_load(self, shaft_system, x_lo, x_hi, label) -> ConvergenceRecord:
        x_evals = self._eval_points_for_interval(shaft_system, x_lo, x_hi)
        rec     = ConvergenceRecord(label=label, x_lo=x_lo, x_hi=x_hi)
        solver  = SubmodelSolver(x_lo=x_lo, x_hi=x_hi)
        history: list[tuple[str, SubmodelResult, float, float, float]] = []

        for level in range(self._max_levels):
            grade  = f"grade_{level}"
            result = solver.solve(self._global_solver, shaft_system, grade)
            f_xz, f_xy, f_res = self._extract_metric_multi(result, x_evals)

            rec.levels.append(grade)
            rec.point_metrics_history.append((f_xz, f_xy, f_res))
            history.append((grade, result, f_xz, f_xy, f_res))

            if len(history) >= self._MIN_LEVELS_FOR_GCI:
                _, res_c, f_xz_c, f_xy_c, f_res_c = history[-3]
                _, res_m, f_xz_m, f_xy_m, f_res_m = history[-2]
                _, res_f, f_xz_f, f_xy_f, f_res_f = history[-1]

                gci_xz = self._compute_gci(f_xz_c, f_xz_m, f_xz_f,
                                            res_c.x_nodes, res_m.x_nodes, res_f.x_nodes)
                gci_xy = self._compute_gci(f_xy_c, f_xy_m, f_xy_f,
                                            res_c.x_nodes, res_m.x_nodes, res_f.x_nodes)
                gci_res = self._compute_gci(f_res_c, f_res_m, f_res_f,
                                            res_c.x_nodes, res_m.x_nodes, res_f.x_nodes)

                rec.gci_history.append({
                    "xz":  gci_xz,
                    "xy":  gci_xy,
                    "res": gci_res,
                })

                if gci_xz.converged and gci_xy.converged and gci_res.converged:
                    rec.converged = True
                    rec.x_final   = list(res_f.x_nodes)
                    return rec

        if history:
            rec.x_final = list(history[-1][1].x_nodes)
        return rec


    def _compute_gci(self, f_c, f_m, f_f, x_c, x_m, x_f) -> RichardsonGCI:
        try:
            return RichardsonGCI(
                f_coarse=f_c, f_medium=f_m, f_fine=f_f,
                x_nodes_coarse=x_c, x_nodes_medium=x_m, x_nodes_fine=x_f,
                gci_threshold=self._gci_threshold,
                safety_factor=self._safety_factor,
            )
        except ValueError:
            # non-uniform refinement ratio — return a dummy converged=False
            return _DummyGCI()

    def _extract_metric_multi(
        self,
        result: SubmodelResult,
        x_evals: list[float],
    ) -> tuple[float, float, float]:
        """
        Mean resultant and per-plane transverse displacement over all x_evals.

        Returns
        -------
        f_xz  : mean |v_xz| over x_evals
        f_xy  : mean |v_xy| over x_evals
        f_res : mean sqrt(v_xz² + v_xy²) over x_evals
        """
        v_xz_list: list[float] = []
        v_xy_list: list[float] = []

        for x in x_evals:
            i = Elem.find_node_index(result.x_nodes, x)
            v_xz_list.append(float(result.d_xz[3 * i + 1]))
            v_xy_list.append(float(result.d_xy[3 * i + 1]))

        f_xz = float(np.mean(np.abs(v_xz_list)))
        f_xy = float(np.mean(np.abs(v_xy_list)))
        f_res = float(np.mean(
            np.sqrt(np.array(v_xz_list)**2 + np.array(v_xy_list)**2)
        ))

        return f_xz, f_xy, f_res

    # ===========================================================================
    # Helpers adicionados a MeshConvergenceStudy
    # ===========================================================================

    def _eval_points_for_interval(
        self,
        shaft_system,
        x_lo: float,
        x_hi: float,
    ) -> list[float]:
        """
        Determine physically meaningful evaluation points within [x_lo, x_hi].

        Raises
        ------
        ValueError
            If any element or load that overlaps [x_lo, x_hi] extends beyond it —
            the interval must fully contain every element it is meant to represent.
            Caller must redefine the interval to cover the full physical extent.
        """
        points: list[float] = []

        # 1. bearings
        for b in shaft_system.bearings:
            p = b.position
            if x_lo - SOLVER_TOLERANCE <= p <= x_hi + SOLVER_TOLERANCE:
                lo_b, hi_b = shaft_system.bearing_extent(b)
                if lo_b < x_lo - SOLVER_TOLERANCE or hi_b > x_hi + SOLVER_TOLERANCE:
                    raise ValueError(
                        f"Interval [{x_lo:.4f}, {x_hi:.4f}] does not fully contain "
                        f"bearing '{b.label or b.designation}' "
                        f"(extent [{lo_b:.4f}, {hi_b:.4f}] mm). "
                        f"Redefine interval to [{min(x_lo, lo_b):.4f}, {max(x_hi, hi_b):.4f}]."
                    )
                points.append(p)

        # 2. gears
        for ge in shaft_system.gears:
            p = ge.position
            if x_lo - SOLVER_TOLERANCE <= p <= x_hi + SOLVER_TOLERANCE:
                lo_g, hi_g = shaft_system.gear_extent(ge)
                if lo_g < x_lo - SOLVER_TOLERANCE or hi_g > x_hi + SOLVER_TOLERANCE:
                    raise ValueError(
                        f"Interval [{x_lo:.4f}, {x_hi:.4f}] does not fully contain "
                        f"gear '{ge.label}' "
                        f"(extent [{lo_g:.4f}, {hi_g:.4f}] mm). "
                        f"Redefine interval to [{min(x_lo, lo_g):.4f}, {max(x_hi, hi_g):.4f}]."
                    )
                points.append(p)

        # 3. distributed radial loads — any overlap requires full containment
        for ld in shaft_system.distributed_radial_loads:
            overlaps = ld.x_lo < x_hi - SOLVER_TOLERANCE and ld.x_hi > x_lo + SOLVER_TOLERANCE
            if not overlaps:
                continue
            if ld.x_lo < x_lo - SOLVER_TOLERANCE or ld.x_hi > x_hi + SOLVER_TOLERANCE:
                raise ValueError(
                    f"Interval [{x_lo:.4f}, {x_hi:.4f}] does not fully contain "
                    f"DistributedRadialLoad '{ld.label}' "
                    f"(span [{ld.x_lo:.4f}, {ld.x_hi:.4f}] mm). "
                    f"Redefine interval to [{min(x_lo, ld.x_lo):.4f}, {max(x_hi, ld.x_hi):.4f}]."
                )
            cx = ld.centroid(LoadPlane.XY)
            points.append(cx)

        if not points:
            points.append((x_lo + x_hi) / 2.0)

        points.sort()
        deduped: list[float] = [points[0]]
        for p in points[1:]:
            if p - deduped[-1] > 1e-4:
                deduped.append(p)

        return deduped


    VALID_REGIONS = frozenset({"gears", "external_distributed", "bearings"})

    @staticmethod
    def intervals_from_shaft_system(
        shaft_system,
        regions: "set[str] | None" = None,
    ) -> tuple[list[tuple[float, float, str]], list[str]]:
        """
        Parameters
        ----------
        regions : set[str] | None
            Which sources to scan for intervals -- any subset of
            {"gears", "external_distributed", "bearings"}. None (default)
            scans all three, unchanged from this method's original
            behaviour, so any existing direct caller keeps working
            exactly as before.

            "bearings" stays available here even though no fixtures-side
            capability exposes it today: the interval it produces still
            only represents a RIGID point reaction (bearing.position),
            not the real load distribution across rolling elements --
            that only becomes physically meaningful once a roller-bearing
            solver exists that resolves the per-roller distribution and
            feeds THAT into the FEM. Until then, "bearings" is reachable
            only by calling this method directly with regions=
            {"bearings", ...} explicitly -- see
            fixtures/studies/shafts/convergence_studies/convergence_study.py's
            own run_convergence(), which never passes it.

        Raises
        ------
        ValueError
            If `regions` contains anything outside VALID_REGIONS.
        """
        if regions is None:
            regions = set(MeshConvergenceStudy.VALID_REGIONS)
        else:
            unknown = regions - MeshConvergenceStudy.VALID_REGIONS
            if unknown:
                raise ValueError(
                    f"intervals_from_shaft_system: unknown region(s) {sorted(unknown)} "
                    f"-- expected a subset of {sorted(MeshConvergenceStudy.VALID_REGIONS)}."
                )

        intervals: list[tuple[float, float, str]] = []
        skipped:   list[str] = []
        seen: set[tuple[float, float]] = set()

        def _add(x_lo: float, x_hi: float, label: str) -> None:
            key = (round(x_lo, 4), round(x_hi, 4))
            if key not in seen:
                seen.add(key)
                intervals.append((x_lo, x_hi, label))

        if "gears" in regions:
            for ge in shaft_system.gears:
                lo, hi = shaft_system.gear_extent(ge)
                label  = ge.label or f"gear@{ge.position:.1f}"
                if (hi - lo) < MIN_FACE_WIDTH_FOR_CONVERGENCE_MM:
                    skipped.append(
                        f"Gear '{label}' @ {ge.position:.4f} mm — no face width defined "
                        f"(b < {MIN_FACE_WIDTH_FOR_CONVERGENCE_MM} mm). "
                        f"Set gear.b to include it in the convergence study."
                    )
                    continue
                _add(lo, hi, label)

        if "external_distributed" in regions:
            for ld in shaft_system.distributed_radial_loads:
                label = ld.label or f"dist@[{ld.x_lo:.1f},{ld.x_hi:.1f}]"
                _add(ld.x_lo, ld.x_hi, label)

        if "bearings" in regions:
            for b in shaft_system.bearings:
                lo, hi = shaft_system.bearing_extent(b)
                label  = getattr(b, "label", "") or getattr(b, "designation", "") or f"bearing@{b.position:.1f}"
                if (hi - lo) < MIN_FACE_WIDTH_FOR_CONVERGENCE_MM:
                    skipped.append(
                        f"Bearing '{label}' @ {b.position:.4f} mm — no width defined "
                        f"(b < {MIN_FACE_WIDTH_FOR_CONVERGENCE_MM} mm). "
                        f"Set bearing.b to include it in the convergence study."
                    )
                    continue
                _add(lo, hi, label)

        return intervals, skipped