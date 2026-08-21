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

Multi-row dispatch -- UPDATE, this turn
-----------------------------------------
solve() and postprocess_and_record() now detect a multi-row bearing
(getattr(bearing, "rows", None) with len >= 2 -- see _is_multirow() below)
and route it to ISO16281MultiRowBallSolver / multirow_dynamic_equivalent_
load() instead of silently mis-handling it through the single-row
_SOLVER_MAP/_POSTPROC_MAP path. Only point-contact multi-row is supported:
a multi-row bearing whose BearingType does not resolve to ISO16281BallSolver
in _SOLVER_MAP (i.e. not DEEP_GROOVE_BALL/ANGULAR_CONTACT today) still
raises NotImplementedError -- there is no multi-row roller/other solver yet.

This is a breaking change to solve()'s return shape, deliberately chosen
(Option A, confirmed by the user over the alternative of a separate
parallel slot): every value in the returned {label: ...} dict, and every
BearingResultBundle.load_distribution / .dynamic_equivalent_load, is now a
LIST -- length 1 for an ordinary single-row bearing, length i for a
multi-row bearing's i rows. A single-row result is never unwrapped back to
a bare object; callers index [0] explicitly. capacity is NOT a list -- see
library.py's own module docstring for why that stays a single (Q_ci, Q_ce)
pair regardless of row count.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, TYPE_CHECKING

import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    BearingResultsLibrary,
    warn_if_floating_loaded,
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
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_multirow_solver import (
    ISO16281MultiRowBallSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_multirow_postprocessing import (
    multirow_dynamic_equivalent_load,
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
# Multi-row detection -- UPDATE, this turn
#
# A bearing counts as multi-row when it exposes >=2 rows (the same
# threshold ISO16281MultiRowBallSolver.solve_bearing() itself enforces).
# Whether a multi-row bearing can actually be SOLVED is a separate question
# (see the BearingType check in solve()/postprocess_and_record() below) --
# this helper only answers "does this bearing have row structure at all",
# so it stays correct even after a second multi-row solver (e.g. roller)
# is eventually added.
# ---------------------------------------------------------------------------

def _is_multirow(bearing: Bearing) -> bool:
    rows = getattr(bearing, "rows", None)
    return bool(rows) and len(rows) >= 2


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
                      (single-row and multi-row alike)
    psi_input bool    forwarded to every per-type solver -- see
                      ISO16281BallSolver's / ISO16281RollerSolver's docstring
                      for the psi_override plane convention this implies.
                      Also honoured for multi-row bearings (see solve()).
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
              ) -> dict[str, list["LoadDistributionResult"]]:
        """
        Solve every bearing in `bearings`, regardless of contact type or
        row count.

        Parameters
        ----------
        shaft_system  : ShaftSystem -- fully resolved
        bearings      : {label: Bearing} -- may mix BearingType values and
                        single-row/multi-row bearings
        psi_override  : {label: psi [rad]} -- forwarded as-is to whichever
                        per-type solver owns that label; honoured for
                        multi-row bearings too (same plane convention as
                        ISO16281BallSolver -- see its class docstring).
        results       : optional BearingResultsLibrary. Single-row groups
                        are handed over wholesale via
                        add_load_distribution_library(), one call per
                        BearingType group. Multi-row bearings are recorded
                        one at a time via set_load_distribution(), plus the
                        full MultiRowBallLoadDistributionResult (row f_r/f_a
                        split, n_iter, residual, ok) stashed under
                        bundle.extra["multirow_result"] so a caller that
                        wants the convergence diagnostics doesn't have to
                        re-solve to get them.

        Returns
        -------
        {label: [LoadDistributionResult, ...]}, ordered to match `bearings`.
        Every value is a list -- length 1 for a single-row bearing, length i
        for a multi-row bearing's i rows (index-aligned with
        bearing.rows). This is Option A (confirmed): callers index [0]
        explicitly rather than the shape silently changing per bearing.

        Raises
        ------
        NotImplementedError : if any single-row bearing's BearingType has no
                               solver registered in _SOLVER_MAP yet, or if a
                               multi-row bearing's BearingType does not
                               resolve to ISO16281BallSolver (no multi-row
                               roller/other solver exists yet).
        """
        single_row: dict[str, Bearing] = {}
        multi_row: dict[str, Bearing] = {}
        for label, b in bearings.items():
            (multi_row if _is_multirow(b) else single_row)[label] = b

        merged: dict[str, list["LoadDistributionResult"]] = {}

        # ------------------------------------------------------------
        # Single-row: existing per-BearingType dispatch, unchanged.
        # ------------------------------------------------------------
        groups: dict[BearingType, dict[str, Bearing]] = {}
        for label, b in single_row.items():
            groups.setdefault(b.bearing_type, {})[label] = b

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
                merged[label] = [local_lib.get(label)]

        # ------------------------------------------------------------
        # Multi-row: route to ISO16281MultiRowBallSolver -- UPDATE, this turn.
        # Point-contact only (see _is_multirow()/module docstring): a
        # multi-row bearing whose BearingType does not resolve to
        # ISO16281BallSolver in _SOLVER_MAP raises, it is never silently
        # solved as if it were single-row.
        # ------------------------------------------------------------
        if multi_row:
            shaft_results  = library.get(shaft_system.name)
            node_by_label  = {n.label: n for n in shaft_results.bearing_nodes}
            mr_solver      = ISO16281MultiRowBallSolver(tol=self.tol)

            for label, b in multi_row.items():
                solver_cls = _SOLVER_MAP.get(b.bearing_type)
                if solver_cls is not ISO16281BallSolver:
                    n_rows = len(b.rows)
                    raise NotImplementedError(
                        f"No multi-row ISO/TS 16281 solver registered for "
                        f"BearingType.{b.bearing_type.name} yet (bearing: "
                        f"'{label}', {n_rows} rows). Multi-row dispatch "
                        f"currently only covers point-contact types that "
                        f"resolve to ISO16281BallSolver in single-row form "
                        f"(DEEP_GROOVE_BALL, ANGULAR_CONTACT)."
                    )

                node   = node_by_label[label]
                Fr_xz, Fr_xy, Fa = node.Fr_xz, node.Fr_xy, node.Fa
                phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

                warn_if_floating_loaded(b, label, Fa)

                # Same psi resolution as ISO16281BallSolver.solve() -- see
                # that class's docstring for the plane convention.
                if (self.psi_input
                        and psi_override is not None
                        and label in psi_override):
                    psi = float(psi_override[label])
                else:
                    psi = node.psi_xz * np.cos(phi_Fr) + node.psi_xy * np.sin(phi_Fr)

                mr_result = mr_solver.solve_bearing(
                    b, Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa, psi=psi, label=label,
                )
                merged[label] = mr_result.row_results

                if results is not None:
                    results.set_load_distribution(
                        label, mr_result.row_results, bearing_type=b.bearing_type,
                    )
                    results.set_extra(label, "multirow_result", mr_result)

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
              load_distribution: dict[str, list["LoadDistributionResult"]] | None = None,
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
                          Always a single (Q_ci, Q_ce) pair, even for a
                          multi-row bearing -- capacity is not row-indexed
                          here (see library.py's own module docstring).
                        catalog[label]["dynamic_equivalent_load"] -> forwarded
                          to adapter.derel_cls.from_distribution(bearing,
                          result, **kwargs) for a single-row bearing, or to
                          multirow_dynamic_equivalent_load(bearing, ...,
                          **kwargs) for a multi-row one -- typically
                          {"inner_rotating": ..., "outer_rotating": ...} in
                          both cases (same kwarg names). Always recorded as
                          a list -- length 1 or i.
                      A label may omit either sub-dict to skip that step.
        load_distribution : reuse an already-computed solve() result instead
                      of solving again. If None, calls self.solve() itself.
                      Same shape as solve()'s return: {label: list[...]}.
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
            row_results = load_distribution[label]   # list[LoadDistributionResult], len 1 or i

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
                if _is_multirow(b):
                    # Row-by-row, via the SAME orchestration-only function
                    # design_bearing_combination_comparison.py already calls
                    # directly -- multirow_dynamic_equivalent_load() only
                    # ever reads mr_result.row_results, so a bare
                    # SimpleNamespace carrying just that is a legitimate
                    # stand-in for the full MultiRowBallLoadDistributionResult
                    # here (no need to keep it around from solve() just for
                    # this call).
                    derel_list = multirow_dynamic_equivalent_load(
                        b, SimpleNamespace(row_results=row_results), **derel_kwargs,
                    )
                else:
                    derel_list = [
                        adapter.derel_cls.from_distribution(b, row_results[0], **derel_kwargs)
                    ]
                results.set_dynamic_equivalent_load(label, derel_list)

            # Stiffness always reads row 0 -- for a multi-row bearing every
            # row shares the same rigid-ring (delta_r, delta_a) at
            # convergence (the co-located-rows idealization --
            # MultiRowBallLoadDistributionResult.delta_r/.delta_a already
            # pick row 0 as the representative for the same reason), so
            # row_results[0] is correct for BOTH the single-row (len==1) and
            # multi-row (len==i) case without branching.
            node = node_by_label[label]
            stiff = adapter.stiffness_fn(b, row_results[0], node.Fr_xz, node.Fr_xy, node.Fa)
            results.set_stiffness(label, stiff)

        return results