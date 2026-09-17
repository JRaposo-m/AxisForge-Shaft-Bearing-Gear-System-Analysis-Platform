"""
axisforge/fixtures/studies/shafts/convergence_studies/global_convergence_study.py

Global (whole-shaft, or a restricted span of it) mesh convergence --
the sibling of convergence_study.py's per-interval studies. Those
refine a small LOCAL subdomain (one gear face width, one distributed-
load span, ...) in isolation, bounded by prescribed BCs read off an
already-solved global model. This module instead re-solves the FULL
RigidSupportFEMSolver from scratch at every refinement level, and asks
one question per shaft: "does the shaft's overall response (not a
single interval's) stop changing as the mesh gets finer?" -- confirmed
with erg (2026-09-17): "quero que faça a analise global e nao por
intervalo".

## CHANGED (this pass, confirmed with erg 2026-09-17): new `domain`
## parameter on run_global_convergence()/GlobalMetricSpec's evaluation --
## "global" no longer means "the whole shaft, always". Three modes:
##   "full"               -- unchanged from before this pass: refine and
##       evaluate over the shaft's ENTIRE mandatory-node extent.
##   "bearing_to_bearing" -- refine and evaluate ONLY within
##       [min(bearing positions), max(bearing positions)] -- the main
##       span the shaft is actually supported over. Any overhang beyond
##       the outermost bearing (a sprocket, a coupling, ...) is still
##       SOLVED (it's part of the same FEM model -- its loads still
##       affect reactions and the moment/deflection field everywhere)
##       but is NOT refined further and NOT sampled for the tracked
##       criterion. Requires >= 2 bearings on the shaft.
##   (x_lo, x_hi)          -- explicit override, same contract as the
##       two named modes: refine + evaluate only inside this span, solve
##       the whole shaft regardless.
## Per erg's own framing: "num outro ficheiro o global de bearing a
## bearing" -- the point of "bearing to bearing" is to keep the
## discretisation-convergence QUESTION scoped to the span whose
## response actually matters structurally (between the supports),
## while a consola/overhang outside that span, often carrying a single
## concentrated load with an already-simple (locally linear/constant)
## M/V diagram, doesn't need its own refinement to answer that
## question -- same reasoning MomentConvergenceStudy's docstring gives
## for why a load-free span is a weak GCI candidate, generalised: scope
## the domain to where refinement is actually informative, rather than
## chasing convergence everywhere.

Why this belongs here, not in solvers/mesh/convergence_solver.py
-------------------------------------------------------------------
convergence_solver.py's own top-of-file note is explicit: "RigidSupportFEMSolver
is no longer imported here at all" -- that module's design contract
(solvers/README.md: "No solver imports another solver... they meet
only at the dispatcher and at the result containers") is why the
per-interval studies there take an already-solved ShaftResults in,
rather than solving anything themselves beyond the SubmodelSolver they
own. A global study has no such luxury -- it genuinely needs to solve
the FULL model, from scratch, at every refinement level. That places it
at the same layer as fem_simple.py's solve_system() and
convergence_study.py's run_convergence() -- fixtures/studies/, the
"dispatcher" layer that is allowed to import RigidSupportFEMSolver
directly -- not inside convergence_solver.py itself.

What IS reused, unchanged, from convergence_solver.py: RichardsonGCI /
_compute_gci / _DummyGCI. Those are pure GCI math -- three floats in,
a converged verdict out -- and never imported RigidSupportFEMSolver in
the first place, so reusing them here does not violate that module's
own contract. GlobalMetricSpec (below) is duck-type compatible with
_compute_gci's `spec` parameter (same attribute names: gci_threshold,
safety_factor, min_p, max_p) -- _compute_gci never checks the concrete
type, only reads those four attributes.

Why comparing at FIXED nodes needs no interpolation
-----------------------------------------------------
Mesh1D._mandatory_positions() always re-includes every geometric
mandatory position (sections/bearings/gears/loads) PLUS whatever is
passed as extra_mandatory, then dedupes by MESH_MIN_NODE_DIST_MM.
Grader.get_grade("grade_N") bisects elementwise, N times, starting
from _base_nodes() (which already filters to [x_lo, x_hi] -- see
mesh_grade.py) -- it only ever ADDS midpoints within that span, it
never removes a node. So if grade_0's WITHIN-DOMAIN node set is X0,
grade_N's WITHIN-DOMAIN node set is X0 union (N generations of
midpoints) -- X0 is an exact subset of every grade_N mesh, same float
values, REGARDLESS of domain mode. That means every field value AT AN
X0 POSITION can be read directly out of any finer grade's ShaftResults
with a tolerant lookup (_locate_fixed_nodes below) -- no interpolation.

## IMPORTANT (this pass): X0 is now `grader.get_grade("grade_0")` --
## i.e. the base nodes WITHIN [x_lo, x_hi] only -- computed ONCE, up
## front, from the domain-scoped Grader, rather than captured from
## grade_0's own ShaftResults.x_nodes as before. Before this pass,
## domain was always "full" and the Grader's own [x_lo, x_hi] equalled
## the whole shaft's extent, so `grader.get_grade("grade_0")` and
## `result.x_nodes` at level 0 were identical -- capturing either
## worked. With domain="bearing_to_bearing" (or an explicit span),
## `result.x_nodes` at ANY level still contains every mandatory node
## for the WHOLE shaft (sections, gears, the other bearing's own
## extent nodes, overhang loads, ...) -- only Grader's own bisection
## output is scoped to [x_lo, x_hi]. Using result.x_nodes as X0 would
## have silently pulled in out-of-domain mandatory positions as
## "tracked" nodes; using grader.get_grade("grade_0") directly does not.

What gets fed into RichardsonGCI
-----------------------------------
For each tracked field (e.g. "v" or "M", read straight off ShaftResults
-- see ShaftResults' own docstring for what v/v_xz/v_xy/M/M_xz/M_xy
already are, no re-derivation needed here), at each grade:
    values_at_x0 = field[indices of X0 within this grade's x_nodes]
    f_grade      = _evaluate_metric(values_at_x0, x0, spec)   # criterion-dependent
f_grade is exactly the kind of scalar RichardsonGCI already expects
(a solution ESTIMATE that should settle to a fixed value as h -> 0).
x_nodes_coarse/medium/fine handed to RichardsonGCI for the
REFINEMENT-RATIO computation are the grade's own REAL WITHIN-DOMAIN
mesh (grader.get_grade(grade), not the full result.x_nodes) -- the
ratio must reflect how much the REFINED region's element count grew,
which is exactly what the Grader for that domain produces; using the
full-shaft result.x_nodes count instead would understate the ratio
whenever domain != "full" (the untouched overhang region would dilute
it). A global Grader doubles the domain's element count every level,
so r ~= 2 at every transition, same as before this pass.

Result shape -- reusing convergence_report.py unmodified
-------------------------------------------------------------
One ConvergenceRecord per shaft (label="global", x_lo/x_hi = the
STUDY's domain -- the whole shaft for domain="full", the bearing span
for domain="bearing_to_bearing"), wrapped in MeshRefinementResult.per_load
= {"global": rec}. convergence_report.py (and text_report.py, which
calls it) already knows how to render this -- a single-entry dict
costs nothing there and needs zero changes to that module.

Dependency (mesh + solvers + fixtures, read-only access -- no core
modification):
  axisforge.mesh.shaft.mesh_generation.mesh_1D
      Mesh1D -- builds the shaft's full mandatory-position mesh (used
      as the Grader's base node list regardless of domain mode -- the
      Grader itself does the [x_lo, x_hi] filtering).
  axisforge.mesh.shaft.mesh_generation.mesh_grade
      Grader -- bisects within [x_lo, x_hi], which this module sets
      per `domain`.
  axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support
      RigidSupportFEMSolver -- the full-shaft solve, once per grade.
  axisforge.solvers.machine_elements.shaft.static_solvers.results_reader
      ShaftResultsReader -- turns each grade's solve into a ShaftResults.
  axisforge.solvers.mesh.convergence_solver
      RichardsonGCI (via _compute_gci), _DummyGCI -- GCI math only.
  axisforge.results.fem_results.convergence_results
      ConvergenceRecord, MeshRefinementResult -- same result SHAPES
      the per-interval studies already populate.
  axisforge.fixtures.studies.shafts.convergence_studies.convergence_library
      ConvergenceResultsLibrary -- same library type run_convergence()
      already stores into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np

from axisforge.config import SOLVER_TOLERANCE
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.shaft.mesh_generation.mesh_grade import Grader
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import (
    RigidSupportFEMSolver,
)
from axisforge.solvers.machine_elements.shaft.static_solvers.results_reader import (
    ShaftResultsReader,
)
from axisforge.solvers.mesh.convergence_solver import _compute_gci
from axisforge.results.fem_results.convergence_results import (
    ConvergenceRecord,
    MeshRefinementResult,
)
from axisforge.fixtures.studies.shafts.convergence_studies.convergence_library import (
    ConvergenceResultsLibrary,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
        ShaftSystem,
    )
    from axisforge.fixtures.construction.construction_capabilities import (
        ConstructionCapabilities,
    )
    from axisforge.results.fem_results.shaft_results import ShaftResults

__all__ = [
    "DomainSpec",
    "GlobalMetricSpec",
    "default_global_metrics",
    "run_global_convergence",
]

_CRITERIA = {"max", "mean", "rms", "max_minus_mean", "node"}
_DOMAIN_MODES = {"full", "bearing_to_bearing"}

DomainSpec = Union[str, "tuple[float, float]"]


# ===========================================================================
# Metric spec -- deliberately NOT axisforge.solvers.mesh.metric_spec.MetricSpec
# ===========================================================================

@dataclass
class GlobalMetricSpec:
    """
    One globally-tracked field. Unlike metric_spec.MetricSpec (built
    around SubmodelResult's reshaped (-1, 3) DOF-triplet layout, and a
    per-point/per-interval evaluation strategy), this reads a field
    DIRECTLY off ShaftResults by attribute name -- ShaftResults already
    stores v/v_xz/v_xy and M/M_xz/M_xy as flat, node-aligned arrays
    (see results_reader.py), so no extraction machinery is needed here.

    Parameters
    ----------
    name      : report key, e.g. "v_res", "M_xz".
    attr      : ShaftResults attribute to read, e.g. "v" (resultant
                displacement), "v_xz", "M" (resultant moment), "M_xy".
                Must be a node-aligned 1D array attribute.
    criterion : how the field (sampled at the fixed, WITHIN-DOMAIN
                node set -- see this module's own top docstring on
                `domain`) is reduced to the one scalar RichardsonGCI
                tracks:
                  "max"            -> max(abs(values))
                  "mean"           -> mean(values)  (signed, NOT abs --
                                      a field that's genuinely centred
                                      near zero should show that; abs()
                                      would hide a real sign-averaging
                                      effect that could itself be
                                      diagnostic)
                  "rms"            -> sqrt(mean(values**2))
                  "max_minus_mean" -> abs(max(abs(values)) - mean(abs(values)))
                                      -- a peakedness/spread indicator:
                                      can still be changing meaningfully
                                      even once "mean" has settled, if a
                                      single point is still sharpening.
                  "node"           -> values AT node_x only, i.e. this
                                      spec tracks one physical point
                                      (must fall within the study's
                                      `domain`), evaluated on the
                                      GLOBALLY refined mesh. Requires
                                      node_x to be set.
    node_x    : physical x [mm], required (and only meaningful) when
                criterion == "node". Must coincide with a grade_0
                mandatory node position WITHIN the study's domain --
                pick one from Mesh1D(shaft_system).x_nodes (filtered to
                the domain) if unsure; anything else raises at run time.
    gci_threshold, safety_factor, min_p, max_p :
                forwarded verbatim to RichardsonGCI (same fields
                metric_spec.MetricSpec carries -- duck-type
                compatible with _compute_gci's `spec` parameter).
    """
    name: str
    attr: str
    criterion: str = "max"
    node_x: float | None = None
    gci_threshold: float = 0.01
    safety_factor: float = 1.25
    min_p: float | None = None
    max_p: float | None = None

    def __post_init__(self) -> None:
        if self.criterion not in _CRITERIA:
            raise ValueError(
                f"GlobalMetricSpec {self.name!r}: unknown criterion "
                f"{self.criterion!r} -- expected one of {sorted(_CRITERIA)}."
            )
        if self.criterion == "node" and self.node_x is None:
            raise ValueError(
                f"GlobalMetricSpec {self.name!r}: criterion='node' requires "
                f"node_x to be set."
            )
        if self.criterion != "node" and self.node_x is not None:
            raise ValueError(
                f"GlobalMetricSpec {self.name!r}: node_x is only meaningful "
                f"with criterion='node' (got criterion={self.criterion!r})."
            )


_ATTR_FOR = {
    ("v", "xz"): "v_xz", ("v", "xy"): "v_xy", ("v", "res"): "v",
    ("M", "xz"): "M_xz", ("M", "xy"): "M_xy", ("M", "res"): "M",
}


def default_global_metrics(
    components: tuple[str, ...] = ("res",),
    criterion: str = "max",
    v_gci_threshold: float = 0.01,
    v_safety_factor: float = 1.25,
    M_gci_threshold: float = 0.02,
    M_safety_factor: float = 3.0,
) -> list[GlobalMetricSpec]:
    """
    Default global metric set: v and M over the study's domain, one
    GlobalMetricSpec per requested (variable, component) pair, all
    sharing the same `criterion`. components=("res",) by default,
    mirroring the same "resultant only unless told otherwise"
    convention default_moment_metrics()/build_moment_point_metrics()
    already use. Pass e.g. components=("xz", "xy", "res") for all three
    planes on both v and M.

    criterion="node" is NOT usable through this convenience builder --
    it needs a per-metric node_x, which this function has no way to
    supply generically for both v and M at once. Build GlobalMetricSpec
    instances directly (with explicit node_x) for that case.

    v keeps the tighter displacement threshold/safety factor (0.01/1.25,
    same as MeshConvergenceStudy's own default), M keeps the looser
    moment one (0.02/3.0 -- the same figures the old, now-removed
    MomentConvergenceStudy used to default to).
    """
    if criterion == "node":
        raise ValueError(
            "default_global_metrics: criterion='node' is not supported "
            "here -- it needs an explicit node_x per metric. Build "
            "GlobalMetricSpec(..., criterion='node', node_x=...) instances "
            "directly instead."
        )
    unknown = set(components) - {"xz", "xy", "res"}
    if unknown:
        raise ValueError(
            f"default_global_metrics: unknown component(s) {sorted(unknown)} "
            f"-- expected a subset of {{'xz', 'xy', 'res'}}."
        )
    specs: list[GlobalMetricSpec] = []
    for comp in components:
        specs.append(GlobalMetricSpec(
            name=f"v_{comp}", attr=_ATTR_FOR[("v", comp)], criterion=criterion,
            gci_threshold=v_gci_threshold, safety_factor=v_safety_factor,
        ))
    for comp in components:
        specs.append(GlobalMetricSpec(
            name=f"M_{comp}", attr=_ATTR_FOR[("M", comp)], criterion=criterion,
            gci_threshold=M_gci_threshold, safety_factor=M_safety_factor,
        ))
    return specs


def _locate_index_of(x: float, fixed_nodes: list[float]) -> int:
    """Tolerant index lookup of a single position within fixed_nodes --
    used by criterion='node'. Raises ValueError (not silently picking
    the nearest node) if nothing within SOLVER_TOLERANCE matches."""
    for i, xn in enumerate(fixed_nodes):
        if abs(xn - x) <= SOLVER_TOLERANCE:
            return i
    raise ValueError(
        f"GlobalMetricSpec: node_x={x:.6f} mm does not coincide with any "
        f"grade_0 mandatory node position WITHIN the study's domain. Pick "
        f"a value from the domain-filtered Mesh1D(shaft_system).x_nodes "
        f"(section boundary, bearing, gear, load, or distributed-load "
        f"edge/centroid)."
    )


def _evaluate_metric(values: np.ndarray, fixed_nodes: list[float], spec: GlobalMetricSpec) -> float:
    """
    Reduces `values` (the tracked field, sampled at `fixed_nodes` --
    same order, same length) to the one scalar RichardsonGCI tracks for
    this spec, per spec.criterion. See GlobalMetricSpec's own docstring
    for what each criterion means.
    """
    if spec.criterion == "max":
        return float(np.max(np.abs(values)))
    if spec.criterion == "mean":
        return float(np.mean(values))
    if spec.criterion == "rms":
        return float(np.sqrt(np.mean(np.square(values))))
    if spec.criterion == "max_minus_mean":
        return float(abs(np.max(np.abs(values)) - np.mean(np.abs(values))))
    if spec.criterion == "node":
        idx = _locate_index_of(spec.node_x, fixed_nodes)
        return float(values[idx])
    raise ValueError(f"Unknown criterion {spec.criterion!r} for metric {spec.name!r}.")


def _locate_fixed_nodes(fixed_nodes: list[float], x_nodes: list[float]) -> list[int]:
    """
    Index of every `fixed_nodes` position within `x_nodes`, tolerant to
    SOLVER_TOLERANCE. Both lists are sorted ascending and fixed_nodes is
    a genuine subsequence of x_nodes by construction (see this module's
    own top docstring) -- a single linear two-pointer sweep is enough.

    Raises
    ------
    RuntimeError
        If a fixed-node position is not found in x_nodes -- would mean
        the "bisection never drops a node" assumption this module
        relies on has been violated.
    """
    idx: list[int] = []
    j = 0
    n = len(x_nodes)
    for x in fixed_nodes:
        while j < n and x_nodes[j] < x - SOLVER_TOLERANCE:
            j += 1
        if j >= n or abs(x_nodes[j] - x) > SOLVER_TOLERANCE:
            raise RuntimeError(
                f"GlobalConvergenceStudy: fixed node x={x:.6f} mm (from grade_0) "
                f"was not found in a finer grade's mesh -- bisection should always "
                f"preserve prior nodes; this indicates MESH_MIN_NODE_DIST_MM merged "
                f"it into a neighbour, or the mesh was rebuilt with a different "
                f"geometry between grades."
            )
        idx.append(j)
        j += 1
    return idx


def _resolve_domain(ss: "ShaftSystem", x_base: list[float], domain: DomainSpec) -> tuple[float, float]:
    """
    Turns `domain` into a concrete (x_lo, x_hi) span for this shaft.
    See this module's own top docstring for what each mode means.
    """
    if domain == "full":
        return x_base[0], x_base[-1]

    if domain == "bearing_to_bearing":
        positions = sorted(b.position for b in ss.bearings)
        if len(positions) < 2:
            raise RuntimeError(
                f"GlobalConvergenceStudy: shaft '{ss.name}' has fewer than "
                f"2 bearings -- domain='bearing_to_bearing' needs at least "
                f"2 to define a span. Use domain='full' or an explicit "
                f"(x_lo, x_hi) instead."
            )
        return positions[0], positions[-1]

    if isinstance(domain, tuple) and len(domain) == 2:
        x_lo, x_hi = float(domain[0]), float(domain[1])
        if x_lo >= x_hi:
            raise ValueError(
                f"GlobalConvergenceStudy: explicit domain=({x_lo}, {x_hi}) "
                f"must have x_lo < x_hi."
            )
        return x_lo, x_hi

    raise ValueError(
        f"GlobalConvergenceStudy: unknown domain {domain!r} -- expected "
        f"one of {sorted(_DOMAIN_MODES)} or an explicit (x_lo, x_hi) tuple."
    )


# ===========================================================================
# Per-shaft global refinement loop
# ===========================================================================

def _run_one_shaft_global(
    ss: "ShaftSystem",
    settings: BeamModelSettings,
    metrics: list[GlobalMetricSpec],
    max_levels: int,
    distribute_gear_labels: set[str] | None,
    kGA_override: float | None,
    domain: DomainSpec,
) -> ConvergenceRecord:
    x_base = Mesh1D(ss).x_nodes
    if len(x_base) < 2:
        raise RuntimeError(
            f"GlobalConvergenceStudy: shaft '{ss.name}' has fewer than 2 "
            f"mandatory node positions -- cannot run a global refinement "
            f"study on it."
        )

    x_lo, x_hi = _resolve_domain(ss, x_base, domain)
    grader = Grader(x_lo=x_lo, x_hi=x_hi, x_nodes=x_base)

    # ## CHANGED (this pass): fixed_nodes is now computed directly from
    # the domain-scoped Grader, up front -- see this module's own top
    # docstring "IMPORTANT" note for why capturing it from a solved
    # grade_0's result.x_nodes (the previous approach) would silently
    # pull in out-of-domain mandatory positions once domain != "full".
    fixed_nodes = grader.get_grade("grade_0")
    if len(fixed_nodes) < 2:
        raise RuntimeError(
            f"GlobalConvergenceStudy: shaft '{ss.name}', domain=({x_lo:.3f}, "
            f"{x_hi:.3f}) mm contains fewer than 2 mandatory node positions "
            f"-- cannot refine/track convergence within it."
        )

    rec = ConvergenceRecord(label="global", x_lo=x_lo, x_hi=x_hi)
    history: list[tuple[str, list[float]]] = []
    result: "ShaftResults | None" = None

    for level in range(max_levels):
        grade = f"grade_{level}"
        grade_nodes = grader.get_grade(grade)  # within-domain nodes only, this grade

        solver = RigidSupportFEMSolver(
            settings,
            distribute_gear_labels=distribute_gear_labels,
            kGA_override=kGA_override,
        )
        # extra_mandatory only injects the WITHIN-DOMAIN refinement --
        # Mesh1D still includes every other mandatory position for the
        # whole shaft regardless (sections, both bearings, gears, loads,
        # any out-of-domain overhang) -- the model solved is always the
        # FULL shaft; only what gets bisected further is scoped.
        solver.solve(ss, extra_mandatory=grade_nodes)
        result = ShaftResultsReader(solver, ss).read()

        idx = _locate_fixed_nodes(fixed_nodes, result.x_nodes)

        metric_values: dict[str, float] = {}
        for spec in metrics:
            field = np.asarray(getattr(result, spec.attr))
            sampled = field[idx]
            metric_values[spec.name] = _evaluate_metric(sampled, fixed_nodes, spec)

        rec.levels.append(grade)
        rec.point_metrics_history.append(metric_values)
        # store this grade's WITHIN-DOMAIN node list alongside the
        # result, for the ratio computation below -- NOT result.x_nodes
        # (see this module's own top docstring on why: the full-shaft
        # node count would dilute the ratio whenever domain != "full").
        history.append((grade, grade_nodes))

        if len(history) >= 3:
            _, x_c = history[-3]
            _, x_m = history[-2]
            _, x_f = history[-1]

            gci_this_transition: dict[str, object] = {}
            for spec in metrics:
                gci_this_transition[spec.name] = _compute_gci(
                    rec.point_metrics_history[-3][spec.name],
                    rec.point_metrics_history[-2][spec.name],
                    rec.point_metrics_history[-1][spec.name],
                    x_c, x_m, x_f,
                    spec,
                )
            rec.gci_history.append(gci_this_transition)

            if all(g.converged for g in gci_this_transition.values()):
                rec.converged = True
                rec.x_final = list(result.x_nodes)
                return rec

    if result is not None:
        rec.x_final = list(result.x_nodes)
    return rec


# ===========================================================================
# Public entry point -- also reachable via convergence_study.run_convergence(
#   study_kind="global")
# ===========================================================================

def run_global_convergence(
    system: "SpurHelicalGearSystem",
    construction: "ConstructionCapabilities",
    library: ConvergenceResultsLibrary | None = None,
    *,
    beam_theory: str = "timoshenko",
    shear_theory: str | None = "cowper",
    integration_method: str | None = "exact",
    metrics: list[GlobalMetricSpec] | None = None,
    domain: DomainSpec = "full",
    max_levels: int = 8,
    distribute_gear_labels: set[str] | None = None,
    kGA_override: float | None = None,
) -> ConvergenceResultsLibrary:
    """
    Run a GLOBAL mesh convergence study on every ShaftSystem in
    system.shafts. One fresh RigidSupportFEMSolver per grade per shaft
    (never reused -- solve() overwrites its instance's own attributes
    in place).

    Unlike the per-interval studies, there is no `regions` parameter
    and no per-interval submodel -- every grade solves the ENTIRE
    shaft, and convergence is judged on `metrics` evaluated per
    GlobalMetricSpec.criterion, over whatever span `domain` selects.
    There is also no already-solved ShaftResults to pass in: this
    function builds its own baseline AND every refined solve itself.

    Parameters
    ----------
    system, construction : same contract as convergence_study.run_convergence().
    library : ConvergenceResultsLibrary | None
        Reused if given; new one created if omitted.
    beam_theory, shear_theory, integration_method : assembled into one
        BeamModelSettings, applied to every shaft's every grade.
    metrics : list[GlobalMetricSpec] | None
        Which fields to track, and how. None -> default_global_metrics()
        (resultant v + resultant M, criterion="max").
    domain : "full" | "bearing_to_bearing" | (x_lo, x_hi)
        Which span of each shaft gets refined and sampled -- see this
        module's own top docstring for the full reasoning. "full"
        (default): the shaft's entire mandatory-node extent, unchanged
        from before this parameter existed. "bearing_to_bearing":
        [min bearing position, max bearing position] -- requires >= 2
        bearings on the shaft, raises RuntimeError otherwise. An
        explicit (x_lo, x_hi) tuple overrides both. The FEM model
        solved at every grade is ALWAYS the full shaft regardless of
        this setting -- only what gets refined/sampled is scoped.
    max_levels : maximum grade levels before giving up, per shaft.
    distribute_gear_labels, kGA_override : forwarded to every
        RigidSupportFEMSolver instance.

    Returns
    -------
    ConvergenceResultsLibrary
        One MeshRefinementResult per shaft, each with a single
        ConvergenceRecord keyed "global" in .per_load.
    """
    if not construction.has_capability("systems.parallel_axis_linear"):
        raise ValueError(
            "run_global_convergence: construction did not request "
            "'systems.parallel_axis_linear' -- the only Construction "
            "capability that resolves the system (gear-mesh loads + "
            "shaft positions) before returning it. Without it, `system` "
            "cannot be trusted to have been resolved, and every grade's "
            "global solve would run with zero gear-mesh loads."
        )
    if not (domain in _DOMAIN_MODES or (isinstance(domain, tuple) and len(domain) == 2)):
        raise ValueError(
            f"run_global_convergence: domain must be one of "
            f"{sorted(_DOMAIN_MODES)} or an explicit (x_lo, x_hi) tuple, "
            f"got {domain!r}."
        )

    settings = BeamModelSettings(
        beam_theory=beam_theory,
        shear_theory=shear_theory,
        integration_method=integration_method,
    )
    metrics = metrics if metrics is not None else default_global_metrics()
    library = library if library is not None else ConvergenceResultsLibrary()

    for ss in system.shafts:
        rec = _run_one_shaft_global(
            ss, settings, metrics, max_levels,
            distribute_gear_labels, kGA_override, domain,
        )
        result = MeshRefinementResult(shaft_name=ss.name, per_load={"global": rec})
        library.store(result)

    return library