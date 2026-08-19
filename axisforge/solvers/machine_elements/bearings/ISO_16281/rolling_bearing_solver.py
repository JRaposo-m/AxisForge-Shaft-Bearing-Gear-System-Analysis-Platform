"""
axisforge/solvers/machine_elements/bearings/ISO_16281/rolling_bearing_solver.py

Orchestrator for per-bearing-type ISO/TS 16281 solves.

A shaft's bearing set can mix contact types -- e.g. a locating deep-groove
ball bearing plus a floating cylindrical roller bearing on the same shaft.
RollingBearingSolver groups bearings by BearingType, dispatches each group
to the registered per-type solver, and merges the results back into one
ordered {label: LoadDistributionResult} dict, in the caller's original
`bearings` order.

This is the only module in the package that knows about more than one
contact type. Each per-type solver (ISO16281BallSolver, ISO16281RollerSolver)
stays self-contained and never imports the other.

solve() vs postprocess_and_record()
------------------------------------
solve() only unifies the load-distribution solve. It never touches the
final BearingResultsLibrary in library.py.

postprocess_and_record() [placeholder name] runs solve() (or accepts an
already-solved dict), then computes capacity, dynamic equivalent load and
secant stiffness per bearing via the correct per-type postprocessing
module, and records everything into a BearingResultsLibrary. It's opt-in
and separate from solve() because capacity/dynamic-equivalent-load need
extra per-bearing inputs (catalogue Cr/Ca, rotating flags) that a bare
solve() has no reason to require.

Contact-type-specific postprocessing (Q_j, minimum_axial_load,
lamina_distribution, stress_riser_factor, thrust_* capacity variants) is
NOT re-exposed here -- call it on the concrete per-type module directly.
postprocess_and_record() only wires up what's uniform enough across
contact types to automate: capacity, dynamic equivalent load, stiffness.

Capacity dispatch -- core cleanup
-----------------------------------
This module used to hold a _PostprocAdapter.capacity_cls per BearingType
(RollingElementCapacity for ball, RollerElementCapacity for roller) and a
"method" key in catalog[label]["capacity"] (default "radial", or
"thrust_nonzero_alpha"/"thrust_90deg") to pick the right classmethod on it.
Both dataclasses are gone now -- capacity (Q_ci/Q_ce) is computed by
core/.../families/{ball,roller}/{radial,thrust}/, and every subtype's
BearingFamily already knows which of its own capacity formulas applies
(e.g. ThrustBallFamily.per_element_dynamic_capacity() auto-dispatches on
alpha_0 == 90deg internally, radial families never had a choice to begin
with). So postprocess_and_record() now just calls

    b.family.per_element_dynamic_capacity(b, **cap_kwargs)

directly -- no capacity_cls, no "method" key. cap_kwargs becomes whatever
that family's own method signature expects: {"Cr": ...} for a radial ball/
roller family, {"Ca": ...} for a thrust ball/roller family, optionally
{"i": ..., "lambda_v": ...} for roller families that expose them. See each
subtype's per_element_dynamic_capacity() docstring in core/ for its exact
kwargs. _PostprocAdapter therefore only carries derel_cls and stiffness_fn
now -- the two things that genuinely still differ per contact type and
have no core/ equivalent (dynamic equivalent load, secant stiffness).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    BearingResultsLibrary,
)

# Type hint only, never constructed here -- see library.py's own alias.
if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
        BallLoadDistributionResult,
    )
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import (
        RollerLoadDistributionResult,
    )

    LoadDistributionResult = BallLoadDistributionResult | RollerLoadDistributionResult
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_solver import (
    ISO16281BallSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing import (
    ball_bearing_postprocessing as _ball_pp,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_solver import (
    ISO16281RollerSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing import (
    roller_bearing_postprocessing as _roller_pp,
)
from axisforge.config import SOLVER_TOLERANCE


# ---------------------------------------------------------------------------
# Type dispatch table
# ---------------------------------------------------------------------------

_SOLVER_MAP: dict[BearingType, type] = {
    BearingType.DEEP_GROOVE_BALL:  ISO16281BallSolver,
    BearingType.ANGULAR_CONTACT:   ISO16281BallSolver,
    BearingType.CYLINDRICAL_ROLLER: ISO16281RollerSolver,
    # BearingType.SELF_ALIGNING_BALL / THRUST_BALL / THRUST_CYLINDRICAL_ROLLER /
    # THRUST_NEEDLE_ROLLER: core/ already has capacity support for these
    # (see families/ball/thrust, families/roller/thrust), but no internal
    # load-distribution SOLVER has been written for them yet -- Phase 2.
    # BearingType.TAPERED_ROLLER / SPHERICAL_ROLLER: needs an extra
    # coordinate transform ISO16281RollerSolver doesn't implement -- Phase 2.
}


# ---------------------------------------------------------------------------
# Postprocessing dispatch -- one adapter per contact-type family, absorbing
# the signature mismatch between ball's bearing_stiffness() (takes Fa) and
# roller's (doesn't -- radial roller bearings carry no axial load). Capacity
# is no longer part of this table -- see module docstring.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PostprocAdapter:
    derel_cls: type             # .from_distribution(bearing, result, ...)
    stiffness_fn: Callable      # (bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9)


def _ball_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9):
    return _ball_pp.bearing_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=eps)


def _roller_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9):
    return _roller_pp.bearing_stiffness(bearing, result, Fr_xz, Fr_xy, eps=eps)


_POSTPROC_MAP: dict[BearingType, _PostprocAdapter] = {
    BearingType.DEEP_GROOVE_BALL: _PostprocAdapter(
        derel_cls=_ball_pp.DynamicEquivalentRollingElementLoad,
        stiffness_fn=_ball_stiffness,
    ),
    BearingType.ANGULAR_CONTACT: _PostprocAdapter(
        derel_cls=_ball_pp.DynamicEquivalentRollingElementLoad,
        stiffness_fn=_ball_stiffness,
    ),
    BearingType.CYLINDRICAL_ROLLER: _PostprocAdapter(
        derel_cls=_roller_pp.LaminaDynamicEquivalentLoad,
        stiffness_fn=_roller_stiffness,
    ),
}


class RollingBearingSolver:
    """
    Orchestrates ISO/TS 16281 solves across a shaft's full bearing set,
    dispatching each bearing to the solver registered for its BearingType.

    Parameters
    ----------
    tol       float   residual tolerance, forwarded to every per-type solver
    psi_input bool    forwarded to every per-type solver -- see
                      ISO16281BallSolver's / ISO16281RollerSolver's docstring
                      for the psi_override plane convention this implies.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol       = tol
        self.psi_input = psi_input

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              psi_override: dict[str, float] | None = None,
              results: BearingResultsLibrary | None = None,
              ) -> dict[str, LoadDistributionResult]:
        """
        Solve every bearing in `bearings`, regardless of contact type.

        Parameters
        ----------
        shaft_system  : ShaftSystem -- fully resolved
        bearings      : {label: Bearing} -- may mix BearingType values
        library       : SimpleFEMResultsLibrary -- must contain results for
                        shaft_system.name
        psi_override  : {label: psi [rad]} -- forwarded as-is to whichever
                        per-type solver owns that label
        results       : optional BearingResultsLibrary. When given, each
                        per-type local library is handed to it wholesale
                        via add_load_distribution_library() -- one call per
                        BearingType group. Omitted by default.

        Returns
        -------
        {label: LoadDistributionResult}, ordered to match `bearings`.

        Raises
        ------
        NotImplementedError : if any bearing's BearingType has no solver
                               registered in _SOLVER_MAP yet.
        """
        groups: dict[BearingType, dict[str, Bearing]] = {}
        for label, b in bearings.items():
            groups.setdefault(b.bearing_type, {})[label] = b

        merged: dict[str, LoadDistributionResult] = {}
        for bearing_type, group in groups.items():
            solver_cls = _SOLVER_MAP.get(bearing_type)
            if solver_cls is None:
                labels = sorted(group)
                raise NotImplementedError(
                    f"No ISO/TS 16281 solver registered for BearingType."
                    f"{bearing_type.name} yet (bearings: {labels}). "
                    f"Available: {[t.name for t in _SOLVER_MAP]}"
                )

            solver = solver_cls(tol=self.tol, psi_input=self.psi_input)
            group_override = (
                {lbl: psi_override[lbl] for lbl in group if lbl in psi_override}
                if psi_override else None
            )
            local_lib = solver.solve(shaft_system, group, library,
                                     psi_override=group_override)

            if results is not None:
                results.add_load_distribution_library(bearing_type, local_lib)

            for label in group:
                merged[label] = local_lib.get(label)

        # Preserve the caller's original ordering rather than the grouping order.
        return {label: merged[label] for label in bearings}

    # ------------------------------------------------------------------
    # Full pipeline -- opt-in, separate from solve() (see module docstring)
    # ------------------------------------------------------------------

    def postprocess_and_record(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              catalog: dict[str, dict],
              load_distribution: dict[str, LoadDistributionResult] | None = None,
              results: BearingResultsLibrary | None = None,
              psi_override: dict[str, float] | None = None,
              ) -> BearingResultsLibrary:
        """
        [PLACEHOLDER NAME] Full pipeline: solve() (if not already done) ->
        capacity -> dynamic equivalent load -> stiffness -> recorded into a
        BearingResultsLibrary, tagged with bearing_type.

        Parameters
        ----------
        shaft_system, bearings, library, psi_override
                      : same as solve()
        catalog       : {label: {"capacity": {...}, "dynamic_equivalent_load": {...}}}
                      per-bearing kwargs for the two postprocessing calls:
                        catalog[label]["capacity"] -> forwarded to
                          b.family.per_element_dynamic_capacity(b, **kwargs) --
                          e.g. {"Cr": 29_600.0} for a radial ball/roller
                          family, {"Ca": 12_500.0} for a thrust family. The
                          family itself picks the right ISO/TS 16281 formula
                          (no "method" key anymore -- see module docstring).
                        catalog[label]["dynamic_equivalent_load"] -> forwarded
                          to adapter.derel_cls.from_distribution(bearing,
                          result, **kwargs), typically {"inner_rotating":
                          ..., "outer_rotating": ...}.
                      A label may omit either sub-dict to skip that step.
        load_distribution : reuse an already-computed solve() result instead
                      of solving again. If None, calls self.solve() itself.
        results       : BearingResultsLibrary to record into. A new one is
                      created if not given.

        Returns
        -------
        BearingResultsLibrary, populated for every label in `bearings` with
        at least load_distribution, plus whichever of capacity /
        dynamic_equivalent_load / stiffness `catalog` requested.
        """
        results = results or BearingResultsLibrary()

        if load_distribution is None:
            load_distribution = self.solve(shaft_system, bearings, library,
                                           psi_override=psi_override,
                                           results=results)
        else:
            for label, b in bearings.items():
                results.set_load_distribution(label, load_distribution[label],
                                              bearing_type=b.bearing_type)

        shaft_results = library.get(shaft_system.name)
        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        for label, b in bearings.items():
            result = load_distribution[label]

            # Fail fast, same guard as before the capacity migration: a
            # bearing type with no stiffness/derel adapter yet (e.g. the new
            # thrust/self-aligning families -- core/ supports their capacity
            # already, but no adapter is registered here yet) should not
            # silently record a partial bundle (capacity set, stiffness
            # missing). Check the adapter before doing anything.
            adapter = _POSTPROC_MAP.get(b.bearing_type)
            if adapter is None:
                raise NotImplementedError(
                    f"No postprocessing adapter registered for BearingType."
                    f"{b.bearing_type.name} yet (bearing: '{label}'). "
                    f"Available: {[t.name for t in _POSTPROC_MAP]}"
                )

            entry = catalog.get(label, {})

            cap_kwargs = dict(entry.get("capacity", {}))
            if cap_kwargs:
                results.set_capacity(label, b.family.per_element_dynamic_capacity(b, **cap_kwargs))

            derel_kwargs = entry.get("dynamic_equivalent_load")
            if derel_kwargs is not None:
                derel = adapter.derel_cls.from_distribution(b, result, **derel_kwargs)
                results.set_dynamic_equivalent_load(label, derel)

            node = node_by_label[label]
            stiff = adapter.stiffness_fn(b, result, node.Fr_xz, node.Fr_xy, node.Fa)
            results.set_stiffness(label, stiff)

        return results