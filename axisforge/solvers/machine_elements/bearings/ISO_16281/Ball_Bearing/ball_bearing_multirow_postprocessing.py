"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_multirow_postprocessing.py

Multi-row post-processing -- ORCHESTRATION ONLY, same discipline as
ball_bearing_multirow_solver.py: this file does not reimplement ISO/TS
16281 eq.(25)-(28) (DynamicEquivalentRollingElementLoad, already in
ball_bearing_postprocessing.py) a second time. It calls that existing,
single-row-shaped function once PER ROW.

Why per row, not merged into one array
----------------------------------------
DynamicEquivalentRollingElementLoad.from_distribution(bearing, result, ...)
takes ONE bearing-like object (needs .cp, via Q_j()) and ONE
BallLoadDistributionResult -- it has no notion of multiple rows, and it
does not need one: Q_e = (mean(Q_j**p))**(1/p) divides by len(Q_j) via
np.mean, not a hardcoded Z, so it is automatically correct for whatever
array you feed it -- there is no "x2" anywhere in that formula, for any
row count. But merging two rows' Q_j arrays into one before calling it
would average together two DIFFERENT raceway rotation histories (each
row's own loaded zone, potentially its own phi_j/cp/Z) into one number
with no ISO/TS 16281 meaning -- eq.(25)-(28) is defined per raceway PER
ROW. So "multi-row support" here means calling the existing function i
times, once per row, never modifying it and never merging its inputs.

Generalized to i rows -- not hardcoded to 2
-----------------------------------------------
i is read from len(bearing.rows) (checked against len(mr_result.row_results)
below, they must match) -- same as
ISO16281MultiRowBallSolver.solve_bearing() itself, which already handles
any i >= 2 rows, not just the double-row case this repo currently
exercises. This file makes the same "any i" guarantee for the Q_ei/Q_ee
side, so it does not need touching again if a 3+ row bearing shows up
later.

Life/capacity note -- NOT implemented here, deliberately out of scope
--------------------------------------------------------------------------
Per-row Q_ei/Q_ee (this file) feed a RACEWAY-level fatigue analysis, one
per row. They are NOT used to build a whole-bearing equivalent load P for
an L10 life estimate -- that comparison uses the standard external-force
-based P = X*Fr + Y*Fa against the multi-row dynamic capacity C (already
i**0.7-scaled by BearingCapacity.dynamic_multirow(), Formula 29).
Row-count effects on the CAPACITY side (i**0.7) and on the raceway
EQUIVALENT LOAD side (per row, no combination at all, no i-dependent
factor) are two separate, non-interacting things.
"""
from __future__ import annotations

from types import SimpleNamespace

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_multirow_solver import (
    MultiRowBallLoadDistributionResult,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_postprocessing import (
    DynamicEquivalentRollingElementLoad,
)


def multirow_dynamic_equivalent_load(
    bearing: Bearing,
    mr_result: MultiRowBallLoadDistributionResult,
    inner_rotating: bool = True,
    outer_rotating: bool = False,
    label: str = "",
) -> list[DynamicEquivalentRollingElementLoad]:
    """
    Per-row Q_ei/Q_ee for a multi-row point-contact bearing -- i rows in,
    i DynamicEquivalentRollingElementLoad out, index-aligned with
    bearing.rows / mr_result.row_results. Works for any i >= 2 (the
    double-row case is just i=2 here, nothing special-cased for it).

    bearing must expose `.rows` (list[dict], the same _REQUIRED_ATTRS shape
    ISO16281MultiRowBallSolver.solve_bearing() itself consumes) -- pass the
    SAME bearing object/view you gave to solve_bearing() to produce
    mr_result, so row j here lines up with mr_result.row_results[j].

    inner_rotating/outer_rotating are forwarded unchanged to every row's
    DynamicEquivalentRollingElementLoad.from_distribution() call -- the
    whole bearing (all rows) shares one physical rotating/stationary
    arrangement, it is not a per-row property.
    """
    rows  = bearing.rows
    i     = len(rows)
    label = label or getattr(bearing, "label", "multirow")

    if len(mr_result.row_results) != i:
        raise ValueError(
            f"{label}: bearing.rows has {i} rows but mr_result.row_results "
            f"has {len(mr_result.row_results)} -- mismatched inputs, this "
            f"mr_result was not produced by solving THIS bearing"
        )

    return [
        DynamicEquivalentRollingElementLoad.from_distribution(
            SimpleNamespace(**row, label=f"{label}-row{j}"), row_res,
            inner_rotating=inner_rotating, outer_rotating=outer_rotating,
            label=f"{label}-row{j}",
        )
        for j, (row, row_res) in enumerate(zip(rows, mr_result.row_results))
    ]