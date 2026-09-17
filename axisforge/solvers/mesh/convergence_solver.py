"""
axisforge/solvers/mesh/convergence_solver.py   [DRAFT]

Mesh convergence study for distributed radial loads.

Strategy  : SubmodelSolver with successive grades (grade_0, grade_1, ...)
Criterion : Richardson extrapolation + Grid Convergence Index (GCI),
            evaluated INDEPENDENTLY per tracked metric (see metric_spec.py).

## CHANGED (this pass), per solvers/README.md's design contract ("No
## solver imports another solver... they meet only at the dispatcher
## and at the result containers"):
##   - MeshConvergenceStudy no longer takes a RigidSupportFEMSolver.
##     It takes `shaft_results: ShaftResults` + `settings: BeamModelSettings`
##     instead -- both threaded straight through to SubmodelSolver.solve(),
##     which itself now takes the same two (see lagrange_multipliers.py).
##     RigidSupportFEMSolver is no longer imported here at all.
##   - _converge_one_load() no longer extracts metric values itself
##     (the old `spec.eval_strategy.points()` + `spec.field_fn()` inline
##     loop). Extraction is delegated to
##     SubmodelConvergencePostProcessing.process() (sub_models/postprocessing.py),
##     which returns a SubmodelResult (results/fem_results/submodel_results.py)
##     -- this module reads SubmodelResult.metric_values/.x_nodes only.

## DONE (was open): FixedPointStrategy.points() in metric_spec.py is no
## longer a stub -- both it and this module's own _eval_points_for_interval()
## now delegate to one shared free function, metric_spec.eval_points_for_interval(),
## so there's exactly one implementation instead of two copies that could
## silently drift apart. See _eval_points_for_interval() below.

## CHANGED (this pass, confirmed with erg 2026-09-17): two follow-up
## fixes to the M-in-load-free-interval problem, decided together --
##   1. RichardsonGCI's relative-error math is UNCHANGED (still
##      e = delta / f_local) -- but it no longer leaks a raw
##      ZeroDivisionError when f_coarse/f_medium is exactly 0.0, or when
##      a level has only 1 node. Both paths now raise ValueError
##      explicitly, same as every other degenerate case already handled,
##      so _compute_gci()'s single except clause is the only place that
##      ever needs to catch a convergence-math failure. _compute_gci()'s
##      except clause is also widened from `except ValueError` to
##      `except (ValueError, ZeroDivisionError, ArithmeticError)` as
##      defense in depth.
##   2. (superseded by the removal below -- kept here for history) an
##      earlier pass stopped MomentConvergenceStudy from refusing
##      load-free intervals up front, instead of skipping them.

## CHANGED (this pass, confirmed with erg 2026-09-17): MomentConvergenceStudy
## -- the per-interval, ratio-selected multi-point (pt0/pt1/pt2) moment
## tracker -- is REMOVED from this module entirely. Per erg: "antes
## tinha posto a criar pontos de analise e assim e agora isso vai ser
## retirado porque esta abordagem do global gostei mesmo muito". The
## whole reason that class existed -- a single fixed evaluation point
## per interval (DisplacementConvergenceStudy's approach) being too
## coarse a proxy for a derived, piecewise-constant field like M over a
## WIDE interval -- doesn't apply anymore: the global bearing-to-bearing
## study (axisforge/fixtures/studies/shafts/convergence_studies/
## global_convergence_study.py) tracks M as an aggregate (max/mean/rms/
## max_minus_mean) over the WHOLE refined domain, not a handful of
## fixed points chosen by a size-ratio heuristic, so there is no longer
## a "which points, how many" question to answer at the per-interval
## level at all.
##
## What stays: the per-interval submodel study (v_xz/v_xy/v_res, one
## fixed point per interval) is UNCHANGED in behaviour -- it is still
## the right tool for "does v converge, per feature" (see
## check_resolution_fem_convergence_total.py), and it never depended on
## anything MomentConvergenceStudy-specific. RichardsonGCI/_DummyGCI/
## _compute_gci are also UNCHANGED and still metric-agnostic -- they
## are reused as-is by global_convergence_study.py.
##
## RENAMED (this pass, confirmed with erg 2026-09-17): the class itself
## goes back to being called `MeshConvergenceStudy` (no more
## `DisplacementConvergenceStudy` name + a `MeshConvergenceStudy = ...`
## back-compat alias underneath it). Per erg: "isto desapareceu por
## causa do global correto, entao nao faz sentido manter o nome como
## displacement... passava para um geral como o MeshConvergenceStudy
## apenas" -- the "Displacement" qualifier only ever made sense as a
## contrast against MomentConvergenceStudy (a sibling class tracking a
## different variable); now that MomentConvergenceStudy is gone and M
## convergence lives entirely in the global study instead, there is no
## longer a family of per-interval study classes to disambiguate
## between -- just the one, so it gets the one general name. It still
## only ever tracks whatever MetricSpec list it's given (v by default,
## via default_displacement_metrics()) -- nothing about its actual
## behaviour changed, only the name.
##
## What else moved because of this: metric_spec.py's "moment_point"/
## "moment_centroid" MetricSpec kinds, build_moment_point_metrics(),
## default_moment_metrics(), the dead field_M_xz/xy/res field functions,
## and PeakTrackingStrategy (only ever wired for a moment-tracking
## strategy that no longer exists) are removed there too -- see that
## module's own CHANGED note. convergence_study.py's study_kind="moment"
## branch and its min_p/max_p/components kwargs are removed -- the
## dispatcher now only knows "displacement" and "global". ConvergenceRecord's
## n_points/point_x fields (only ever set by MomentConvergenceStudy) are
## removed from convergence_results.py, and convergence_report.py's
## "tracked points (fixed at grade_0)" rendering is removed to match.

References
----------
Roache, P.J. (1998). Verification and Validation in Computational Science and Engineering.
Richardson, L.F. (1911). Phil. Trans. R. Soc. London A, 210, 307-357.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.sub_models.lagrange_multipliers import (
    SubmodelSolver,
    SubmodelSolution,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.sub_models.postprocessing import (
    SubmodelConvergencePostProcessing,
)
from axisforge.config import MIN_FACE_WIDTH_FOR_CONVERGENCE_MM
from axisforge.results.fem_results.convergence_results import (
    ConvergenceRecord,
    MeshRefinementResult,
)
from axisforge.results.fem_results.submodel_results import SubmodelResult
from axisforge.solvers.mesh.metric_spec import (
    MetricSpec,
    default_displacement_metrics,
    eval_points_for_interval,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
    from axisforge.results.fem_results.shaft_results import ShaftResults


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
    Metric-agnostic -- takes f_coarse/f_medium/f_fine as plain floats.

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
                 safety_factor: float = 1.25,
                 min_p: float | None = None,
                 max_p: float | None = None):

        self.x_nodes_coarse = x_nodes_coarse
        self.x_nodes_medium = x_nodes_medium
        self.x_nodes_fine   = x_nodes_fine
        self.gci_threshold  = gci_threshold
        self.safety_factor  = safety_factor
        self.f_coarse       = f_coarse
        self.f_medium       = f_medium
        self.f_fine         = f_fine

        # Explicit guard before dividing, instead of letting a level
        # with only 1 node raise a raw ZeroDivisionError. Raise
        # ValueError here, let _compute_gci's fallback handle it like
        # any other "can't build a real GCI from this" case.
        n_c = len(x_nodes_coarse) - 1
        n_m = len(x_nodes_medium) - 1
        n_f = len(x_nodes_fine) - 1
        if n_c <= 0 or n_m <= 0 or n_f <= 0:
            raise ValueError(
                f"Cannot compute refinement ratio: level(s) with fewer "
                f"than 2 nodes (n_c={n_c}, n_m={n_m}, n_f={n_f})."
            )
        self.r_m_c = n_m / n_c
        self.r_f_m = n_f / n_m

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

        # "trivially converged" short-circuit. A tracked metric can be
        # essentially FLAT across grades -- e.g. M_xy for a shaft whose
        # load is entirely in the XZ plane. If the level-to-level change
        # is already negligible relative to the metric's own scale
        # (RELATIVE_FLAT_TOL) AND in absolute terms (ABS_FLAT_TOL, for a
        # metric whose true value is itself ~0), treat this transition
        # as converged outright -- GCI_m_c=GCI_f_m=0.0, converged=True --
        # instead of raising. p/e_m_c/e_f_m/f_h0 stay their NaN defaults
        # (getattr(..., float("nan")) upstream in gci_detail_table()).
        scale = max(abs(self.f_coarse), abs(self.f_medium), abs(self.f_fine), 1.0)
        RELATIVE_FLAT_TOL = 1e-8
        ABS_FLAT_TOL = 1e-9
        d_m_c = abs(self.f_medium - self.f_coarse)
        d_f_m = abs(self.f_fine - self.f_medium)
        flat_tol = max(ABS_FLAT_TOL, RELATIVE_FLAT_TOL * scale)
        if d_m_c <= flat_tol and d_f_m <= flat_tol:
            self.p = float("nan")
            self.e_m_c = 0.0
            self.e_f_m = 0.0
            self.f_h0 = self.f_fine
            self.GCI_m_c = 0.0
            self.GCI_f_m = 0.0
            self.converged_m_c = True
            self.converged_f_m = True
            self.converged = True
            return

        # observed order of convergence
        # guard the ratio BEFORE log, not after -- a sign change between
        # (f_coarse-f_medium) and (f_medium-f_fine) (oscillatory
        # convergence) makes this ratio negative, and np.log() of a
        # negative number silently returns nan instead of raising. Raise
        # explicitly here so the existing except -> _DummyGCI() fallback
        # in _compute_gci handles this exactly like the other degenerate
        # cases.
        if self.f_medium == self.f_fine:
            raise ValueError(
                "f_medium == f_fine -- cannot compute observed order p "
                "(division by zero in the convergence ratio) for this "
                "metric/interval."
            )
        ratio = (self.f_coarse - self.f_medium) / (self.f_medium - self.f_fine)
        if ratio <= 0.0:
            raise ValueError(
                f"Non-monotonic convergence (f_coarse-f_medium and "
                f"f_medium-f_fine have opposite sign, ratio={ratio:.6g} <= 0) "
                f"-- cannot compute a real observed order p for this "
                f"metric/interval."
            )
        self.p = float(np.log(ratio) / np.log(self.r))

        if min_p is not None:
            self.p = max(self.p, min_p)
        if max_p is not None:
            self.p = min(self.p, max_p)

        # guard against the degenerate p ~= 0 case (r**p - 1 == 0) --
        # both the f_h0 extrapolation and the GCI formulas divide by
        # r**p - 1.
        if abs(self.r**self.p - 1.0) < 1e-12:
            raise ValueError(
                f"Degenerate observed order p={self.p:.6g} (r**p - 1 ~= 0) -- "
                f"cannot extrapolate or compute GCI for this metric/interval."
            )

        # Explicit guard before dividing by f_coarse/f_medium -- a
        # metric that lands exactly on 0.0 at a level (M sampled at a
        # sign-change node) raises ValueError here instead of a raw
        # ZeroDivisionError, falling back to _DummyGCI the same way.
        if self.f_coarse == 0.0:
            raise ValueError(
                "f_coarse == 0.0 -- cannot compute relative error e_m_c "
                "(division by zero) for this metric/interval."
            )
        self.e_m_c = (self.f_medium - self.f_coarse) / self.f_coarse

        if self.f_medium == 0.0:
            raise ValueError(
                "f_medium == 0.0 -- cannot compute relative error e_f_m "
                "(division by zero) for this metric/interval."
            )
        self.e_f_m = (self.f_fine - self.f_medium) / self.f_medium

        self.f_h0 = self.f_coarse + (self.f_coarse - self.f_medium) / (self.r**self.p - 1)

        self.GCI_m_c = (self.safety_factor * abs(self.e_m_c) / (self.r_m_c**self.p - 1))
        self.GCI_f_m = (self.safety_factor * abs(self.e_f_m) / (self.r_f_m**self.p - 1))

        self.converged_m_c = self.GCI_m_c < self.gci_threshold
        self.converged_f_m = self.GCI_f_m < self.gci_threshold
        self.converged     = self.converged_f_m and self.converged_m_c


def _compute_gci(f_c, f_m, f_f, x_c, x_m, x_f, spec: MetricSpec) -> "RichardsonGCI | _DummyGCI":
    """
    Metric-agnostic GCI computation with fallback -- shared by
    MeshConvergenceStudy (this module) and by
    global_convergence_study.run_global_convergence() (imports this
    function directly). `spec` only needs to duck-type gci_threshold/
    safety_factor/min_p/max_p; this function never checks its concrete
    type.

    except clause is widened from `except ValueError` to
    `except (ValueError, ZeroDivisionError, ArithmeticError)` as
    defense in depth -- RichardsonGCI itself converts every
    division-by-zero path it knows about into an explicit ValueError,
    but this still catches a future degenerate case that raises
    ZeroDivisionError/ArithmeticError directly.
    """
    try:
        return RichardsonGCI(
            f_coarse=f_c, f_medium=f_m, f_fine=f_f,
            x_nodes_coarse=x_c, x_nodes_medium=x_m, x_nodes_fine=x_f,
            gci_threshold=spec.gci_threshold,
            safety_factor=spec.safety_factor,
            min_p=spec.min_p,
            max_p=spec.max_p,
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        # non-uniform refinement ratio, degenerate/non-monotonic order,
        # a metric landing exactly on 0.0, a level with 1 node, etc.
        # (see RichardsonGCI's own guards) -- dummy, converged=False
        return _DummyGCI()


# ===========================================================================
# Orchestrator -- DISPLACEMENT (per-interval, local submodels)
# ===========================================================================

class MeshConvergenceStudy:
    """
    Runs the mesh refinement study for all the intervals selected
    in a ShaftSystem, tracking whichever metrics its `metrics` list
    says to (v_xz/v_xy/v_res by default, via
    default_displacement_metrics()), one fixed evaluation point per
    interval.

    ## RENAMED (2026-09-17): this class used to be split into
    ## DisplacementConvergenceStudy + MomentConvergenceStudy (two
    ## sibling per-interval studies, one per tracked variable). Now
    ## that MomentConvergenceStudy is removed -- M convergence lives
    ## entirely in the global bearing-to-bearing study instead (see
    ## global_convergence_study.py) -- there is only ever one
    ## per-interval study class again, so it goes back to its original,
    ## general name rather than keeping a "Displacement" qualifier that
    ## no longer contrasts against anything.

    Strategy  : SubmodelSolver with successive grades (grade_0, grade_1, ...)
    Criterion : RichardsonGCI, run independently per tracked metric.
                ALL intervals must satisfy GCI < threshold for EVERY
                requested metric. See metric_spec.py.

    Parameters
    ----------
    shaft_results : ShaftResults
        Already-solved global result (RigidSupportFEMSolver + a reader
        that populates ShaftResults) -- used as prescribed BCs for every
        submodel. This solver never solves the global model itself and
        never imports RigidSupportFEMSolver (see solvers/README.md's
        design contract).
    settings : BeamModelSettings
        Forwarded to SubmodelSolver.solve() for every submodel element
        build -- must match whatever theory shaft_results was solved
        with.
    metrics : list[MetricSpec] | None
        Which variables to track, and how. Defaults to
        default_displacement_metrics() -- v_xz/v_xy/v_res.
    gci_threshold, safety_factor : fallback defaults for MetricSpecs
        built without their own (each MetricSpec carries its own; these
        have no effect on a `metrics` list that already sets them).
    max_levels : maximum grade levels before giving up.
    """

    _MIN_LEVELS_FOR_GCI = 3  # Richardson requires 3 evaluations minimum

    def __init__(
        self,
        shaft_results: "ShaftResults",
        settings: "BeamModelSettings",
        metrics: list[MetricSpec] | None = None,
        gci_threshold: float = 0.01,
        safety_factor: float = 1.25,
        max_levels: int = 8,
    ):
        self._shaft_results = shaft_results
        self._settings       = settings
        self._metrics        = metrics if metrics is not None else default_displacement_metrics()
        self._gci_threshold = gci_threshold
        self._safety_factor = safety_factor
        self._max_levels    = max_levels
        self._postproc = SubmodelConvergencePostProcessing()

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
        if self._shaft_results.d_total_xz is None or len(self._shaft_results.d_total_xz) == 0:
            raise RuntimeError(
                "MeshConvergenceStudy.run() requires an already-solved ShaftResults "
                "(shaft_results.d_total_xz is empty/None). Solve the global model "
                "first (RigidSupportFEMSolver + a results reader) and pass the "
                "resulting ShaftResults in -- this study never solves the global "
                "model itself."
            )

        result = MeshRefinementResult(shaft_name=getattr(shaft_system, "name", ""))
        for x_lo, x_hi, label in intervals:
            result.per_load[label] = self._converge_one_load(
                shaft_system, x_lo, x_hi, label
            )
        return result

    def _converge_one_load(self, shaft_system, x_lo, x_hi, label) -> ConvergenceRecord:
        rec    = ConvergenceRecord(label=label, x_lo=x_lo, x_hi=x_hi)
        solver = SubmodelSolver(x_lo=x_lo, x_hi=x_hi)

        # history[grade_index] = (grade_name, SubmodelSolution, SubmodelResult)
        history: list[tuple[str, SubmodelSolution, SubmodelResult]] = []
        grade0_x_nodes: list[float] | None = None

        for level in range(self._max_levels):
            grade = f"grade_{level}"
            prev_solution = history[-1][1] if history else None

            solution = solver.solve(self._shaft_results, shaft_system, self._settings, grade)
            if level == 0:
                grade0_x_nodes = list(solution.x_nodes)

            submodel_result = self._postproc.process(
                solution, self._metrics, shaft_system, grade0_x_nodes, prev_solution,
            )

            rec.levels.append(grade)
            rec.point_metrics_history.append(submodel_result.metric_values)
            history.append((grade, solution, submodel_result))

            if len(history) >= self._MIN_LEVELS_FOR_GCI:
                _, _, res_c = history[-3]
                _, _, res_m = history[-2]
                _, _, res_f = history[-1]

                gci_this_transition: dict[str, RichardsonGCI] = {}
                for spec in self._metrics:
                    gci_this_transition[spec.name] = self._compute_gci(
                        res_c.metric_values[spec.name],
                        res_m.metric_values[spec.name],
                        res_f.metric_values[spec.name],
                        res_c.x_nodes, res_m.x_nodes, res_f.x_nodes,
                        spec,
                    )
                rec.gci_history.append(gci_this_transition)

                if all(g.converged for g in gci_this_transition.values()):
                    rec.converged = True
                    rec.x_final   = list(res_f.x_nodes)
                    return rec

        if history:
            rec.x_final = list(history[-1][2].x_nodes)
        return rec

    def _compute_gci(self, f_c, f_m, f_f, x_c, x_m, x_f, spec: MetricSpec) -> "RichardsonGCI | _DummyGCI":
        """One-line delegator to the module-level _compute_gci() above."""
        return _compute_gci(f_c, f_m, f_f, x_c, x_m, x_f, spec)

    # ===========================================================================
    # Helpers
    # ===========================================================================

    def _eval_points_for_interval(
        self,
        shaft_system,
        x_lo: float,
        x_hi: float,
    ) -> list[float]:
        """
        One-line delegator to metric_spec.eval_points_for_interval() --
        was a full standalone copy of the same body. FixedPointStrategy
        (metric_spec.py) calls that same shared function directly, so
        this method and FixedPointStrategy.points() are guaranteed to
        stay identical.
        """
        return eval_points_for_interval(shaft_system, x_lo, x_hi)

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
            scans all three.

            "bearings" stays available here even though no fixtures-side
            capability exposes it today: the interval it produces still
            only represents a RIGID point reaction (bearing.position),
            not the real load distribution across rolling elements.

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