"""
axisforge/fixtures/studies/shafts/convergence_studies/convergence_study.py

Studies stage: runs a mesh convergence study for every ShaftSystem in a
SpurHelicalGearSystem's system.shafts, publishing every
MeshRefinementResult into one ConvergenceResultsLibrary.

## CHANGED (this pass, confirmed with erg 2026-09-17): `study_kind="moment"`
## is REMOVED entirely, along with MomentConvergenceStudy and its
## MomentConvergenceStudy-only kwargs (min_p/max_p/components). Per erg:
## "antes tinha posto a criar pontos de analise e assim e agora isso vai
## ser retirado porque esta abordagem do global gostei mesmo muito" --
## see convergence_solver.py's own CHANGED note for the full reasoning.
## This dispatcher now only knows two study kinds:
##
##   "displacement" -> MeshConvergenceStudy: per-interval
##       submodels (gear face widths, distributed-load spans, per
##       `regions`), tracks v at one fixed point per interval.
##   "global"        -> run_global_convergence() (global_convergence_study.py):
##       NO submodels/intervals -- refines the WHOLE shaft mesh and
##       re-solves RigidSupportFEMSolver from scratch at every grade,
##       tracking a domain-wide criterion (max/mean/rms/max_minus_mean/
##       a specific node) instead of per-interval points. This is now
##       the only way M gets tracked at all in this module -- there is
##       no more per-interval moment study.
##
## `_DEFAULT_GCI_THRESHOLD`/`_DEFAULT_SAFETY_FACTOR` are simplified from
## per-study_kind dicts down to plain scalars (0.01/1.25), since
## "displacement" is the only study_kind left that resolves a fallback
## from them at all.

## CHANGED (earlier pass): `study_kind: "displacement" | "moment"` kwarg
## was introduced here (see history above for why "moment" is gone
## again). `study_kind` remains a CALL-TIME kwarg, not partial-pinned by
## study_capabilities.py -- same treatment as shear_theory/
## integration_method already get. study_capabilities.py needs NO
## changes for this: its "shaft_fem.convergence.<region>.timoshenko"
## capabilities still pin beam_theory + regions only, exactly as
## before; study_kind is supplied by the caller at each
## run_convergence() call.
##
## `metrics=`/`global_metrics=` are each only meaningful for their own
## study_kind. Passing the wrong one for the requested study_kind is a
## hard error rather than a silent no-op, so a caller doesn't get
## quietly downgraded to a different behaviour by force of habit.
##
## gci_threshold/safety_factor default to None and resolve to the
## displacement defaults (0.01/1.25) if the caller doesn't override.
## NOT accepted for study_kind="global" at all (each GlobalMetricSpec
## already carries its own -- see that class's own docstring in
## global_convergence_study.py); passing them there is a hard error
## too, for the same "don't let a caller silently think this had an
## effect" reasoning.

## CHANGED (earlier pass): new `study_kind="global"` value, dispatching
## into global_convergence_study.run_global_convergence() -- refines the
## WHOLE shaft mesh and tracks GCI on a domain-wide criterion (max /
## mean / rms / max_minus_mean / a specific node -- see
## GlobalMetricSpec.criterion in global_convergence_study.py) instead
## of per-interval submodels. Brought into THIS dispatcher (rather than
## staying a separate entry point) per erg's own framing: "podias
## colocar aqui que tipo de analise de convergencia... se e global ou
## nao". `regions` has no effect for study_kind="global" (there are no
## intervals -- the whole shaft is always the domain) -- left
## unguarded (not a hard error) since passing it is harmless, unlike
## `metrics`/`global_metrics` which would silently change semantics if
## misapplied; `regions` genuinely does nothing either way. New
## `global_metrics`/`kGA_override` kwargs are only meaningful for
## study_kind="global"; `kGA_override` is otherwise unused by this
## dispatcher (the baseline solve for displacement never took it
## either, before or after this pass -- unchanged).

Everything else below (the capability guard, the "one instance per
shaft" rule, intervals_from_shaft_system()'s region gating -- called
only for study_kind="displacement" now, it lives on MeshConvergenceStudy
-- the ShaftResultsReader flow for the displacement baseline) is
UNCHANGED from the previous version.

## RENAMED (2026-09-17): convergence_solver.py's per-interval study
## class is back to being called plain `MeshConvergenceStudy` -- the
## "DisplacementConvergenceStudy" name (plus a `MeshConvergenceStudy = ...`
## back-compat alias underneath it) is gone. See that module's own
## RENAMED note. This file already imported it as `MeshConvergenceStudy`
## (the alias) before this pass, so no import-site change was needed
## here, only the docstring mentions below.

Dependency (solvers + fixtures, read-only access -- no core modification):
  axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support
      RigidSupportFEMSolver -- builds each shaft's baseline global solve
      (displacement path only -- global_convergence_study.py builds its
      own, at every grade, not just once).
  axisforge.solvers.machine_elements.shaft.static_solvers.results_reader
      ShaftResultsReader -- turns that solve into the ShaftResults the
      displacement study actually consumes.
  axisforge.solvers.mesh.convergence_solver
      MeshConvergenceStudy -- does the actual refinement/GCI work for
      study_kind="displacement".
  axisforge.solvers.mesh.metric_spec
      MetricSpec -- only meaningful for study_kind="displacement".
  axisforge.fixtures.studies.shafts.convergence_studies.global_convergence_study
      GlobalMetricSpec, run_global_convergence -- do the actual
      refinement/GCI work for study_kind="global".
  axisforge.fixtures.studies.shafts.convergence_studies.convergence_library
      ConvergenceResultsLibrary -- the result SHAPE this module publishes into,
      shared by both study_kind values.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import (
    RigidSupportFEMSolver,
)
from axisforge.solvers.machine_elements.shaft.static_solvers.results_reader import (
    ShaftResultsReader,
)
from axisforge.solvers.mesh.convergence_solver import MeshConvergenceStudy
from axisforge.solvers.mesh.metric_spec import MetricSpec
from axisforge.fixtures.studies.shafts.convergence_studies.convergence_library import (
    ConvergenceResultsLibrary,
)
from axisforge.fixtures.studies.shafts.convergence_studies.global_convergence_study import (
    DomainSpec,
    GlobalMetricSpec,
    run_global_convergence,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.fixtures.construction.construction_capabilities import (
        ConstructionCapabilities,
    )

__all__ = ["run_convergence"]

_STUDY_KINDS = ("displacement", "global")

# Fallback defaults for a "displacement" study whose caller doesn't
# override gci_threshold/safety_factor -- mirrors
# MeshConvergenceStudy's own constructor defaults exactly
# (convergence_solver.py). "global" is NOT covered by these on purpose
# -- see the guard in run_convergence() below: gci_threshold/
# safety_factor are rejected outright for study_kind="global" rather
# than resolving to some invented default, because GlobalMetricSpec
# already carries its own per-metric thresholds and there is no single
# scalar that could override "all of them" without silently reshaping
# the caller's metrics list.
_DEFAULT_GCI_THRESHOLD = 0.01
_DEFAULT_SAFETY_FACTOR = 1.25


def run_convergence(
    system: "SpurHelicalGearSystem",
    construction: "ConstructionCapabilities",
    library: ConvergenceResultsLibrary | None = None,
    *,
    beam_theory: str = "timoshenko",
    shear_theory: str | None = "cowper",
    integration_method: str | None = "exact",
    regions: "set[str]" = frozenset({"gears", "external_distributed"}),
    distribute_gear_labels: set[str] | None = None,
    kGA_override: float | None = None,
    study_kind: str = "displacement",
    metrics: list[MetricSpec] | None = None,
    global_metrics: list[GlobalMetricSpec] | None = None,
    domain: DomainSpec = "full",
    gci_threshold: float | None = None,
    safety_factor: float | None = None,
    max_levels: int = 8,
) -> ConvergenceResultsLibrary:
    """
    Run a mesh convergence study on every ShaftSystem in system.shafts,
    storing each shaft's MeshRefinementResult in `library`. One
    dispatcher, two study kinds:

      "displacement" -> MeshConvergenceStudy: per-interval
          submodels (gear face widths, distributed-load spans, per
          `regions`), tracks v at one fixed point per interval.
      "global"        -> run_global_convergence() (global_convergence_study.py):
          NO submodels/intervals -- refines the WHOLE shaft mesh and
          re-solves RigidSupportFEMSolver from scratch at every grade,
          tracking a domain-wide criterion (max/mean/rms/
          max_minus_mean/a specific node -- see
          GlobalMetricSpec.criterion) instead of per-interval points.
          This is the only way to track M (moment) convergence in this
          module -- there is no more per-interval moment study.

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Already built AND resolved. Purely read here.
    construction : ConstructionCapabilities
        Checked via construction.has_capability("systems.parallel_axis_linear").
    library : ConvergenceResultsLibrary | None
        Reused if given; a new one created if omitted. Shared type
        across both study_kind values, so a library can be re-passed
        across calls to accumulate e.g. a "displacement" run and a
        "global" run for the same system side by side (distinct
        shaft_name keys don't collide; re-solving the same shaft under
        a different study_kind DOES overwrite its previous
        MeshRefinementResult -- one result per shaft per library, same
        "fresh solve replaces stale" rule as every other Library in
        this project).
    beam_theory : "timoshenko" | "euler_bernoulli"
        Pinned via study_capabilities.py's partial application per
        capability string -- a direct caller may still pass it
        explicitly. Applies to both study_kind values.
    shear_theory, integration_method : passed at call time, exactly
        like solve_system() -- built into one BeamModelSettings,
        applied uniformly across every shaft's solve(s), for whichever
        study_kind is requested.
    regions : set[str]
        Forwarded to MeshConvergenceStudy.intervals_from_shaft_system()
        for study_kind="displacement" only. Has NO EFFECT for
        study_kind="global" (there are no intervals -- the whole shaft
        is always the domain); passing a non-default value there is
        not an error, it is simply unused. Default
        {"gears", "external_distributed"} -- NOT "bearings"; see this
        module's original top docstring for why.
    distribute_gear_labels, kGA_override : RigidSupportFEMSolver
        constructor kwargs, applied uniformly across all shafts and
        both study_kind values (kGA_override reaches the baseline
        solve for "displacement", and every grade's solve for
        "global").
    study_kind : "displacement" | "global"
        Picks which study actually runs -- see the two-way summary
        above. Anything else raises ValueError.
    metrics : list[MetricSpec] | None
        Forwarded to MeshConvergenceStudy. None -> its own
        default_displacement_metrics(). Only valid when study_kind=
        "displacement" -- passing it for study_kind="global" raises
        ValueError.
    global_metrics : list[GlobalMetricSpec] | None
        Forwarded to run_global_convergence(). None -> its own
        default_global_metrics() (resultant v + resultant M,
        criterion="max"). Only valid when study_kind="global" --
        passing it for study_kind="displacement" raises ValueError.
        Build GlobalMetricSpec instances directly (criterion="mean"/
        "rms"/"max_minus_mean"/"node", the last with an explicit
        node_x) for anything beyond the plain "max" default.
    domain : "full" | "bearing_to_bearing" | (x_lo, x_hi)
        Only meaningful for study_kind="global" -- forwarded straight
        to run_global_convergence() (see that function's own docstring
        in global_convergence_study.py). Must be left at its default
        ("full") for study_kind="displacement" -- that study has its
        own domain concept (`regions` + per-interval submodels);
        passing a non-default `domain` there raises ValueError.
    gci_threshold, safety_factor : float | None
        For study_kind="displacement": None (the default) resolves to
        0.01/1.25. Pass explicitly to override. For study_kind=
        "global": must be left None -- each GlobalMetricSpec in
        `global_metrics` already carries its own; passing either here
        raises ValueError (see this module's own top-of-file CHANGED
        note for why there is no single sensible override to apply
        across a whole metrics list).
    max_levels : maximum grade levels before giving up, applied
        uniformly across all shafts, for whichever study_kind is
        requested.

    Returns
    -------
    ConvergenceResultsLibrary
    """
    if study_kind not in _STUDY_KINDS:
        raise ValueError(
            f"run_convergence: study_kind must be one of {_STUDY_KINDS}, "
            f"got {study_kind!r}"
        )

    if study_kind == "global":
        if metrics is not None:
            raise ValueError(
                "run_convergence: metrics= is only accepted with "
                "study_kind='displacement' -- pass global_metrics= for "
                "study_kind='global' instead."
            )
        if gci_threshold is not None or safety_factor is not None:
            raise ValueError(
                "run_convergence: gci_threshold=/safety_factor= are not "
                "accepted with study_kind='global' -- each GlobalMetricSpec "
                "in `global_metrics` already carries its own gci_threshold/"
                "safety_factor (see global_convergence_study.default_global_metrics()); "
                "there is no single scalar that could override an entire "
                "metrics list without silently reshaping it. Build "
                "GlobalMetricSpec instances (or call "
                "default_global_metrics(v_gci_threshold=..., M_gci_threshold=..., "
                "...)) with the thresholds you want instead."
            )
    else:  # study_kind == "displacement"
        if global_metrics is not None:
            raise ValueError(
                "run_convergence: global_metrics= is only accepted with "
                "study_kind='global'."
            )
        if domain != "full":
            raise ValueError(
                "run_convergence: domain= is only accepted with "
                "study_kind='global' -- study_kind='displacement' has its "
                "own domain concept (`regions` + per-interval submodels), "
                "so passing a non-default `domain` here is almost certainly "
                "a mistake rather than a no-op you intended."
            )

    if not construction.has_capability("systems.parallel_axis_linear"):
        raise ValueError(
            "run_convergence: construction did not request "
            "'systems.parallel_axis_linear' -- the only Construction "
            "capability that resolves the system (gear-mesh loads + "
            "shaft positions) before returning it. Without it, `system` "
            "cannot be trusted to have been resolved, and the solve(s) "
            "this study refines around would be built with zero "
            "gear-mesh loads."
        )

    library = library if library is not None else ConvergenceResultsLibrary()

    # ------------------------------------------------------------------
    # study_kind="global" -- no submodels, no baseline solve here at all;
    # run_global_convergence() builds and solves everything itself, once
    # per grade per shaft. Dispatch and return early.
    # ------------------------------------------------------------------
    if study_kind == "global":
        return run_global_convergence(
            system, construction, library,
            beam_theory=beam_theory,
            shear_theory=shear_theory,
            integration_method=integration_method,
            metrics=global_metrics,
            domain=domain,
            max_levels=max_levels,
            distribute_gear_labels=distribute_gear_labels,
            kGA_override=kGA_override,
        )

    # ------------------------------------------------------------------
    # study_kind="displacement" -- unchanged from the previous version:
    # one baseline global solve per shaft, then a per-interval submodel
    # study bounded by it.
    # ------------------------------------------------------------------
    resolved_gci = gci_threshold if gci_threshold is not None else _DEFAULT_GCI_THRESHOLD
    resolved_safety = safety_factor if safety_factor is not None else _DEFAULT_SAFETY_FACTOR

    settings = BeamModelSettings(
        beam_theory=beam_theory,
        shear_theory=shear_theory,
        integration_method=integration_method,
    )

    for ss in system.shafts:
        global_solver = RigidSupportFEMSolver(
            settings,
            distribute_gear_labels=distribute_gear_labels,
        )
        global_solver.solve(ss)

        reader = ShaftResultsReader(global_solver, ss)
        shaft_results = reader.read()

        study = MeshConvergenceStudy(
            shaft_results,
            settings,
            metrics=metrics,
            gci_threshold=resolved_gci,
            safety_factor=resolved_safety,
            max_levels=max_levels,
        )

        intervals, skipped = MeshConvergenceStudy.intervals_from_shaft_system(
            ss, regions=regions,
        )
        for reason in skipped:
            warnings.warn(
                f"run_convergence: shaft '{ss.name}': {reason}",
                stacklevel=2,
            )

        result = study.run(ss, intervals)
        library.store(result)

    return library