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
{"lambda_v": ...} for a roller family that exposes it (i is read off
bearing.i internally by every family now, never passed here -- see
DeepGrooveBallFamily's / CylindricalRollerFamily's own
per_element_dynamic_capacity() docstrings in core/). _PostprocAdapter
therefore only carries result_cls, derel_cls and stiffness_fn now -- the
things that genuinely still differ per contact type and have no core/
equivalent (dynamic equivalent load, secant stiffness) plus, since this
turn, which container type wraps a list of rows for them.

Multi-row dispatch -- routing is purely structural
-----------------------------------------------------
solve() and postprocess_and_record() detect a multi-row bearing
(getattr(bearing, "rows", None) with len >= 2 -- see _is_multirow() below)
and route it to ISO16281MultiRowBallSolver instead of the single-row
_SOLVER_MAP path. _is_multirow() is the only gate, based on the bearing's
own `.rows`, not on a BearingType-keyed lookup. An earlier version of this
branch additionally checked `_SOLVER_MAP.get(b.bearing_type) is
ISO16281BallSolver` before allowing the multi-row solve to proceed -- that
check was removed because it was both redundant and actively wrong:
_SOLVER_MAP is populated for single-row dispatch only, and the one
multi-row family that exists today (MultiRowThrustBallFamily) declares
BEARING_TYPE = BearingType.THRUST_BALL, which was never in _SOLVER_MAP to
begin with (THRUST_BALL single-row solving is still Phase 2 -- see
_SOLVER_MAP's own comments), so the old check raised NotImplementedError
for the only real multi-row bearing that exists, unconditionally.

The actual guardrail against a non-point-contact `.rows` shape does not
need a type label at all: ISO16281MultiRowBallSolver.solve_bearing() calls
check_bearing_ready() on every row view against the same REQUIRED_ATTRS
ISO16281BallSolver itself requires (A, alpha_0, phi_j, Ri, cp, Dpw, Z) --
if a future multi-row family's rows don't carry point-contact geometry,
that call raises a RuntimeError naming exactly which attributes are
missing, which is both correct and more informative than a BearingType
check would be. There is still no multi-row roller/other solver --
ISO16281MultiRowBallSolver is the only one instantiated here -- so a
non-point-contact multi-row bearing fails loudly at that
check_bearing_ready() call rather than being silently mis-solved.

NOTE (adjacent, not fixed by this change): postprocess_and_record()'s own
_POSTPROC_MAP lookup (`adapter = _POSTPROC_MAP.get(b.bearing_type)`) is a
separate dict from _SOLVER_MAP and still has no BearingType.THRUST_BALL
entry. A multi-row THRUST_BALL bearing reaches solve() successfully, but
postprocess_and_record() on that same bearing still raises
NotImplementedError at the adapter lookup, since THRUST_BALL isn't
registered in _POSTPROC_MAP either. That is a distinct gap in a different
dict and is intentionally left alone here.

UPDATE, this turn -- BallBearingResult / RollerBearingResult unification
-----------------------------------------------------------------------------
Ball_Bearing/ and Roller_Bearing/ were separately reconstructed so that
every solve -- single-row or multi-row -- now hands back the SAME
container type (BallBearingResult / RollerBearingResult), whose `.rows` is
always a list (length 1 or i), rather than a bare per-row result on one
path and a separate ad-hoc multi-row class on the other. This orchestrator
had not been updated to match -- three call sites here still assumed the
old, pre-unification shapes, and are fixed by this pass:

  1. local_lib.get(label) -- BallLoadDistributionLibrary /
     RollerLoadDistributionLibrary now store a BallBearingResult /
     RollerBearingResult per label (see ball_bearing_solver.py's /
     roller_bearing_solver.py's own solve(), which wraps every
     solve_contact() output via .single() before recording it), not a
     bare per-row result. `merged[label] = [local_lib.get(label)]` used to
     be correct when .get() returned the bare row directly; it now wraps
     the CONTAINER in a length-1 list instead of the row itself. Fixed to
     `merged[label] = local_lib.get(label).rows` -- unwrapping the
     container to get back the actual list[LoadDistributionResult] the
     Option A contract promises.

  2. mr_result.row_results -- ISO16281MultiRowBallSolver.solve_bearing()
     used to return its own MultiRowBallLoadDistributionResult (defined in
     ball_bearing_multirow_solver.py), whose per-row list was named
     `.row_results`. That class is retired; solve_bearing() now returns a
     BallBearingResult via its `.multirow()` classmethod, whose per-row
     list is named `.rows` (same attribute name single-row results use).
     Every `.row_results` reference below is renamed to `.rows`.

  3. multirow_dynamic_equivalent_load() -- lived in
     ball_bearing_multirow_postprocessing.py, which is retired outright
     (the file no longer exists). Its job is now
     DynamicEquivalentRollingElementLoad.from_bearing_result() /
     LaminaDynamicEquivalentLoad.from_bearing_result() in
     ball_bearing_postprocessing.py / roller_bearing_postprocessing.py,
     which handle single-row (1 row) and multi-row (i rows) through the
     SAME method, driven by a BallBearingResult's/RollerBearingResult's own
     `.is_multirow`. postprocess_and_record()'s dynamic-equivalent-load
     step is simplified accordingly: it no longer branches on
     _is_multirow(b) to pick between two different functions -- it always
     wraps `row_results` (the list this orchestrator already has) into the
     right container type via the new `_PostprocAdapter.result_cls` field,
     and calls `adapter.derel_cls.from_bearing_result(b, wrapped, ...)`
     once, unconditionally. The wrapping is a plain container construction
     (`result_cls(rows=row_results)`), not `.single()`/`.multirow()` --
     row_results is already exactly the list either of those would have
     produced, so re-deriving it from a bare row would only re-litigate
     work already done.

NOTE (not fixed by this pass, flagged for later): this orchestrator still
passes the whole per-type `local_lib` (now BallBearingResult/
RollerBearingResult-valued) into
`results.add_load_distribution_library(bearing_type, local_lib)`, and
`mr_result` (now a BallBearingResult) into
`results.set_load_distribution(label, mr_result.rows, ...)` /
`results.set_extra(label, "multirow_result", mr_result)`. Whether
library.py's BearingResultsLibrary itself has been updated to expect
BallBearingResult/RollerBearingResult-shaped entries (vs. the older bare
per-row shape) was not checked in this pass -- library.py was not
re-examined here. If it still assumes the pre-unification shape, it is the
next thing to reconcile, not this file.
"""
from __future__ import annotations

from dataclasses import dataclass
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
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
    BallBearingResult,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing import (
    ball_bearing_postprocessing as _ball_pp,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_multirow_solver import (
    ISO16281MultiRowBallSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_solver import (
    ISO16281RollerSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import (
    RollerBearingResult,
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
    # NOTE: THRUST_BALL single-row solving is Phase 2 and deliberately not
    # here -- this table is single-row dispatch only. Multi-row THRUST_BALL
    # (MultiRowThrustBallFamily) is routed by _is_multirow() below, not by
    # this table -- see the module docstring's "Multi-row dispatch" section.
}


# ---------------------------------------------------------------------------
# Multi-row detection
#
# A bearing counts as multi-row when it exposes >=2 rows (the same
# threshold ISO16281MultiRowBallSolver.solve_bearing() itself enforces).
# Whether a multi-row bearing can actually be SOLVED is a separate question
# (see the module docstring's "Multi-row dispatch" section) -- this helper
# only answers "does this bearing have row structure at all", so it stays
# correct even after a second multi-row solver (e.g. roller) is eventually
# added.
# ---------------------------------------------------------------------------

def _is_multirow(bearing: Bearing) -> bool:
    rows = getattr(bearing, "rows", None)
    return bool(rows) and len(rows) >= 2


# ---------------------------------------------------------------------------
# Postprocessing dispatch -- one adapter per contact-type family, absorbing
# the signature mismatch between ball's bearing_stiffness() (takes Fa) and
# roller's (doesn't -- radial roller bearings carry no axial load). Capacity
# is not part of this table -- see module docstring.
#
# result_cls -- NEW, this turn: the BallBearingResult/RollerBearingResult
# container type used to wrap a bare list[LoadDistributionResult] before
# handing it to derel_cls.from_bearing_result() (see module docstring,
# "BallBearingResult / RollerBearingResult unification").
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PostprocAdapter:
    result_cls: type             # BallBearingResult / RollerBearingResult
    derel_cls: type              # .from_bearing_result(bearing, result, ...)
    stiffness_fn: Callable       # (bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9)


def _ball_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9):
    return _ball_pp.bearing_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=eps)


def _roller_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9):
    return _roller_pp.bearing_stiffness(bearing, result, Fr_xz, Fr_xy, eps=eps)


_POSTPROC_MAP: dict[BearingType, _PostprocAdapter] = {
    BearingType.DEEP_GROOVE_BALL: _PostprocAdapter(
        result_cls=BallBearingResult,
        derel_cls=_ball_pp.DynamicEquivalentRollingElementLoad,
        stiffness_fn=_ball_stiffness,
    ),
    BearingType.ANGULAR_CONTACT: _PostprocAdapter(
        result_cls=BallBearingResult,
        derel_cls=_ball_pp.DynamicEquivalentRollingElementLoad,
        stiffness_fn=_ball_stiffness,
    ),
    BearingType.CYLINDRICAL_ROLLER: _PostprocAdapter(
        result_cls=RollerBearingResult,
        derel_cls=_roller_pp.LaminaDynamicEquivalentLoad,
        stiffness_fn=_roller_stiffness,
    ),
    # NOTE: BearingType.THRUST_BALL still has no entry here. A multi-row
    # THRUST_BALL bearing reaches solve() successfully (see _is_multirow()),
    # but postprocess_and_record() will still raise NotImplementedError at
    # the `adapter = _POSTPROC_MAP.get(...)` lookup below for that same
    # bearing -- this dict was intentionally left untouched; closing this
    # gap is a separate decision.
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
                        full BallBearingResult (row f_r/f_a split, n_iter,
                        residual, ok) stashed under
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
                               solver registered in _SOLVER_MAP yet.
        RuntimeError         : if a multi-row bearing's `.rows` don't carry
                               point-contact geometry (raised by
                               check_bearing_ready() inside
                               ISO16281MultiRowBallSolver.solve_bearing(),
                               naming the missing attributes) -- there is
                               still no multi-row roller/other solver.
        """
        single_row: dict[str, Bearing] = {}
        multi_row: dict[str, Bearing] = {}
        for label, b in bearings.items():
            (multi_row if _is_multirow(b) else single_row)[label] = b

        merged: dict[str, list["LoadDistributionResult"]] = {}

        # ------------------------------------------------------------
        # Single-row: existing per-BearingType dispatch.
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

            # FIXED, this turn: local_lib.get(label) now returns a
            # BallBearingResult/RollerBearingResult CONTAINER (see
            # ball_bearing_solver.py's/roller_bearing_solver.py's own
            # solve(), which wraps every solve_contact() output via
            # .single() before recording it) -- not the bare per-row
            # result the old `[local_lib.get(label)]` line assumed.
            # `.rows` unwraps it back to the actual
            # list[LoadDistributionResult] the Option A contract promises
            # (length 1 here, always, since these are all single-row
            # groups) -- see module docstring.
            for label in group:
                merged[label] = local_lib.get(label).rows

        # ------------------------------------------------------------
        # Multi-row: route to ISO16281MultiRowBallSolver. Routing is
        # purely structural (_is_multirow(), already applied above to
        # build `multi_row`) -- there is no BearingType gate here.
        # ISO16281MultiRowBallSolver is the only multi-row solver that
        # exists; if a bearing's `.rows` don't carry point-contact
        # geometry, check_bearing_ready() inside solve_bearing() raises
        # RuntimeError naming the missing attributes -- see module
        # docstring, "Multi-row dispatch" section.
        # ------------------------------------------------------------
        if multi_row:
            shaft_results  = library.get(shaft_system.name)
            node_by_label  = {n.label: n for n in shaft_results.bearing_nodes}
            mr_solver      = ISO16281MultiRowBallSolver(tol=self.tol)

            for label, b in multi_row.items():
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
                # FIXED, this turn: solve_bearing() now returns a
                # BallBearingResult (via .multirow()), not the retired
                # MultiRowBallLoadDistributionResult -- its per-row list is
                # named `.rows`, not `.row_results`. See module docstring.
                merged[label] = mr_result.rows

                if results is not None:
                    results.set_load_distribution(
                        label, mr_result.rows, bearing_type=b.bearing_type,
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
                          and reads its own row count off bearing.i (no
                          "method" or "i" key here anymore -- see module
                          docstring). Always a single (Q_ci, Q_ce) pair,
                          even for a multi-row bearing -- capacity is not
                          row-indexed here (see library.py's own module
                          docstring).
                        catalog[label]["dynamic_equivalent_load"] -> forwarded
                          to adapter.derel_cls.from_bearing_result(bearing,
                          wrapped_result, **kwargs) -- typically
                          {"inner_rotating": ..., "outer_rotating": ...}.
                          ONE call handles single-row and multi-row alike
                          now (see module docstring, "BallBearingResult /
                          RollerBearingResult unification") -- always
                          recorded as a list, length 1 or i.
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

        Note
        ----
        A multi-row THRUST_BALL bearing still raises NotImplementedError
        here, at the `adapter = _POSTPROC_MAP.get(b.bearing_type)` lookup
        below -- _POSTPROC_MAP has no THRUST_BALL entry. This is a separate
        gap from the one removed in solve()'s multi-row branch; see the
        module docstring's "Multi-row dispatch" section.
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

            # Both derel_cls.from_bearing_result() and stiffness_fn() now
            # expect the WRAPPER (BallBearingResult/RollerBearingResult),
            # not a bare row -- both read .rows[0] internally for
            # stiffness. Wrap once, reuse for both.
            wrapped = adapter.result_cls(rows=row_results)

            derel_kwargs = entry.get("dynamic_equivalent_load")
            if derel_kwargs is not None:
                derel_list = adapter.derel_cls.from_bearing_result(b, wrapped, **derel_kwargs)
                results.set_dynamic_equivalent_load(label, derel_list)

            node = node_by_label[label]
            stiff = adapter.stiffness_fn(b, wrapped, node.Fr_xz, node.Fr_xy, node.Fa)
            results.set_stiffness(label, stiff)

        return results