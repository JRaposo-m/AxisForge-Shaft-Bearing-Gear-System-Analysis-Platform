"""
axisforge/solvers/mesh/metric_spec.py

Per-variable convergence metric definitions for MeshConvergenceStudy.

Why this module exists
-----------------------
The original convergence_solver.py hardcoded a single metric family:
resultant transverse displacement v = sqrt(v_xz^2 + v_xy^2), sampled at a
FIXED evaluation point per interval (load centroid / bearing position /
gear position -- see MeshConvergenceStudy._eval_points_for_interval()).

That is a fine choice for v: it is the FEM's primal unknown (solved
directly by the stiffness system), it converges at the element's
optimal order, and it is a smooth field over the interval -- a fixed
sample point is a defensible proxy for "did displacement converge".

## CHANGED (this pass, confirmed with erg 2026-09-17): every M-tracking
## code path that used to live in this module -- default_moment_metrics(),
## build_moment_point_metrics(), _validate_components(),
## _MOMENT_COMPONENT_PLANES, the "moment_centroid"/"moment_point"
## MetricSpec kinds, and PeakTrackingStrategy (only ever wired for those
## kinds, never actually used through field_fn/eval_strategy) -- is
## REMOVED. Per erg: "antes tinha posto a criar pontos de analise e
## assim e agora isso vai ser retirado porque esta abordagem do global
## gostei mesmo muito". M convergence is no longer tracked per-interval
## at all -- it lives entirely in the whole-shaft global study
## (axisforge/fixtures/studies/shafts/convergence_studies/
## global_convergence_study.py), which tracks M as a domain-wide
## aggregate (max/mean/rms/max_minus_mean/a specific node) over the
## refined mesh, not via MetricSpec/EvalPointStrategy at all -- see that
## module's own GlobalMetricSpec, a completely separate, unrelated
## dataclass. field_M_xz/field_M_xy/field_M_res are also removed -- they
## were ALREADY dead code before this pass (default_moment_metrics()'s
## own prior docstring said so explicitly: extraction went through
## SubmodelConvergencePostProcessing's fixed-centroid logic, kind=
## "moment_centroid"/"moment_point", never through field_fn at all), so
## removing them changes nothing that was actually reachable.
##
## What this means MetricSpec now tracks: v_xz/v_xy/v_res only (via
## default_displacement_metrics(), still the default `metrics=` for
## MeshConvergenceStudy). There is only one MetricSpec "shape" left, so
## `kind`/`plane`/`point_index`/`point_x` are REMOVED from the dataclass
## entirely -- confirmed safe against
## axisforge/solvers/machine_elements/shaft/fem_solvers/sub_models/
## postprocessing.py (SubmodelConvergencePostProcessing), which used to
## be the file reading spec.kind to decide how to extract each grade's
## value. That module is rewritten in the same pass (see its own
## top-of-file CHANGED note): process() no longer branches on spec.kind
## at all -- there is only the "field" extraction path left (sample
## spec.field_fn via spec.eval_strategy.points(...), reduce with
## spec.aggregate), so there is nothing left in this codebase that reads
## MetricSpec.kind/.plane/.point_index/.point_x anywhere.

It is NOT a fine choice for bending moment M or stress sigma -- kept
here as background for why M was ever a MetricSpec candidate, even
though that path is gone now:

  - M and sigma are DERIVED quantities (obtained by differentiating the
    displacement field, or by taking EI * curvature). Differentiation
    costs one order of accuracy relative to the primal field -- this is
    standard FEM a posteriori error theory (cf. Zienkiewicz & Zhu
    superconvergent patch recovery). So M/sigma converge slower than v,
    and judging them against the same GCI threshold that "worked" for v
    is not meaningful.

  - M and sigma are typically NOT flattest at the fixed geometric points
    (load centroid, bearing position) that work for v -- their extrema
    can sit elsewhere in the interval and can MIGRATE slightly in x as
    the mesh refines. This is exactly the failure mode the whole-shaft
    global study sidesteps by tracking an aggregate over the WHOLE
    refined domain instead of a handful of fixed/tracked points.

Design principle (explicit, per project discussion, still applies to
what remains)
-----------------------------------------------------
Do NOT build one generalized "evaluation point" abstraction that tries
to serve every variable identically. v gets its own MetricSpec: its own
field extractor, its own EvalPointStrategy, its own GCI threshold/safety
factor. A future per-interval metric (if one is ever added again) should
get the same treatment rather than fighting a shared abstraction --
FixedPointStrategy's fixed-point logic should stay free to diverge from
whatever a different variable's strategy needs.

Open question this module still does NOT resolve (flagged, not guessed):
  SIGMA SOURCE: is sigma nominal bending stress (M*c/I, a smooth field)
  or notch/concentration stress from postprocessing.py's Peterson/
  Shigley/Neuber Kt lookups at shoulders/keyways? A 1D beam mesh
  refining toward a geometric step does not converge toward a finite
  notch stress in the classical FEM sense -- Kt-based stress is a
  tabulated correction applied AT a section, not a mesh limit. No sigma
  MetricSpec is defined below until this is settled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Protocol, TYPE_CHECKING

import numpy as np

from axisforge.config import SOLVER_TOLERANCE
from axisforge.core.loads import LoadPlane

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.sub_models import (
        SubmodelResult,
    )


# ===========================================================================
# Field extraction -- per node, per metric
# ===========================================================================

class FieldFn(Protocol):
    """
    A field function reads one SubmodelResult and returns a per-node
    array ALIGNED WITH result.x_nodes (same length, same order) of the
    scalar quantity this metric tracks. Sign is preserved -- strategies
    and aggregators decide whether/where to take abs().
    """
    def __call__(self, result: "SubmodelResult") -> np.ndarray: ...


def field_v_xz(result: "SubmodelResult") -> np.ndarray:
    """Transverse displacement, xz-plane. result.d_xz[3*i+1] per node i."""
    return np.asarray(result.d_xz).reshape(-1, 3)[:, 1]


def field_v_xy(result: "SubmodelResult") -> np.ndarray:
    """Transverse displacement, xy-plane. result.d_xy[3*i+1] per node i."""
    return np.asarray(result.d_xy).reshape(-1, 3)[:, 1]


def field_v_res(result: "SubmodelResult") -> np.ndarray:
    """Resultant transverse displacement sqrt(v_xz^2 + v_xy^2) per node."""
    return np.sqrt(field_v_xz(result) ** 2 + field_v_xy(result) ** 2)


# ===========================================================================
# Evaluation-point strategy
# ===========================================================================

class EvalPointStrategy(ABC):
    """
    Decides WHERE (which x values) a metric is sampled at, for a given
    grade's SubmodelResult. Returns points sorted, deduped, and clipped
    to [x_lo, x_hi] -- same contract
    MeshConvergenceStudy._eval_points_for_interval() already follows.
    """

    @abstractmethod
    def points(
        self,
        shaft_system,
        x_lo: float,
        x_hi: float,
        field_fn: FieldFn,
        latest_result: "SubmodelResult | None",
    ) -> list[float]:
        ...


def eval_points_for_interval(shaft_system, x_lo: float, x_hi: float) -> list[float]:
    """
    Shared by FixedPointStrategy.points() (below) and
    MeshConvergenceStudy._eval_points_for_interval() (convergence_solver.py,
    a one-line delegator to this function) -- one copy of this logic
    instead of two that could drift.

    Determine physically meaningful evaluation points within [x_lo, x_hi]:
    bearing positions, gear positions, distributed-load centroids, or
    the interval midpoint as a fallback if none apply.

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


class FixedPointStrategy(EvalPointStrategy):
    """
    Grade-independent geometric points: load centroid / bearing position
    / gear position, or the interval midpoint as a fallback. Delegates to
    eval_points_for_interval() above -- v's (and, today, the only)
    strategy, because v is a smooth primal field and a fixed point is a
    defensible proxy for it.

    `field_fn` and `latest_result` are ignored -- points never depend on
    the solved field, only on shaft_system geometry.
    """

    def points(self, shaft_system, x_lo, x_hi, field_fn, latest_result) -> list[float]:
        return eval_points_for_interval(shaft_system, x_lo, x_hi)


# ===========================================================================
# MetricSpec -- what MeshConvergenceStudy actually iterates over
# ===========================================================================

def _default_aggregate(values: np.ndarray) -> float:
    """mean(abs(.)) over the sampled points -- matches
    _extract_metric_multi()'s historical behaviour
    (np.mean(np.abs(v_xz_list)))."""
    return float(np.mean(np.abs(values)))


@dataclass
class MetricSpec:
    """
    One convergence-tracked variable. MeshConvergenceStudy holds a
    list[MetricSpec] (default: the three displacement specs below) and
    runs RichardsonGCI independently per spec, per interval, per grade.

    Parameters
    ----------
    name          : key used in ConvergenceRecord.gci_history /
                    point_metrics_history (e.g. "v_xz", "v_res").
    field_fn      : per-node array extractor (see field_v_xz etc. above).
    eval_strategy : where to sample field_fn's output. FixedPointStrategy
                    is stateless, so sharing an instance is harmless.
    aggregate     : combines the sampled values into one float. Default
                    mean(abs(.)).
    gci_threshold, safety_factor, min_p, max_p :
                    per-metric RichardsonGCI parameters.

    ## CHANGED (this pass, confirmed with erg 2026-09-17): `kind`/
    ## `plane`/`point_index`/`point_x` are REMOVED -- confirmed safe
    ## against a fresh copy of postprocessing.py
    ## (SubmodelConvergencePostProcessing), which no longer branches on
    ## spec.kind at all (see that module's own CHANGED note): there is
    ## only ever one MetricSpec "shape" now, the "field" extraction
    ## (spec.field_fn sampled at spec.eval_strategy.points(...), reduced
    ## with spec.aggregate), so a `kind` discriminator has nothing left
    ## to discriminate between.
    """
    name: str
    field_fn: FieldFn | None = None
    eval_strategy: EvalPointStrategy | None = None
    aggregate: Callable[[np.ndarray], float] = _default_aggregate
    gci_threshold: float = 0.01
    safety_factor: float = 1.25
    min_p: float | None = None
    max_p: float | None = None


# ---------------------------------------------------------------------------
# Default metric set -- v_xz/v_xy/v_res. The only metric set this module
# builds now that default_moment_metrics()/build_moment_point_metrics()
# are removed.
# ---------------------------------------------------------------------------

def default_displacement_metrics() -> list[MetricSpec]:
    """Fresh instances every call -- FixedPointStrategy is stateless so
    sharing would be harmless, but keep the pattern consistent regardless."""
    return [
        MetricSpec(name="v_xz", field_fn=field_v_xz, eval_strategy=FixedPointStrategy()),
        MetricSpec(name="v_xy", field_fn=field_v_xy, eval_strategy=FixedPointStrategy()),
        MetricSpec(name="v_res", field_fn=field_v_res, eval_strategy=FixedPointStrategy()),
    ]


STUDY_VARIABLES = ("v_xz", "v_xy", "v_res")
"""Declarative list of variables this module can track. Narrowed (this
pass, confirmed with erg 2026-09-17) from ("v_xz","v_xy","v_res","M_xz",
"M_xy","M_res") -- M is no longer tracked here at all (see module
docstring); it lives in global_convergence_study.py's own, unrelated
GlobalMetricSpec/_ATTR_FOR instead. sigma deliberately still excluded
until its source (nominal vs Kt-corrected) is settled."""


# ===========================================================================
# NOTE (extraction path, confirmed against a fresh postprocessing.py)
# ===========================================================================
#
# MeshConvergenceStudy._converge_one_load() (convergence_solver.py) calls
# SubmodelConvergencePostProcessing.process(solution, self._metrics,
# shaft_system, grade0_x_nodes, prev_solution) and reads back
# SubmodelResult.metric_values[spec.name] / .x_nodes. postprocessing.py
# (sub_models/postprocessing.py) was reviewed and rewritten in this same
# pass to match: process() no longer branches on spec.kind at all (that
# field is gone -- see MetricSpec's own CHANGED note above) -- for every
# spec, it calls spec.eval_strategy.points(shaft_system, x_lo, x_hi,
# spec.field_fn, prev_solution) to get sample x's, then spec.aggregate(
# spec.field_fn(solution)[indices at those x's]). The old
# original_element_midpoints()/moment_at_original_centroids()/
# _postproc_for() moment-recovery machinery (TimoshenkoPostProcessing/
# EulerBernoulliPostProcessing dispatch by beam_theory) is removed from
# that file entirely -- it only ever served the now-deleted
# "moment_centroid"/"moment_point" MetricSpec kinds.
