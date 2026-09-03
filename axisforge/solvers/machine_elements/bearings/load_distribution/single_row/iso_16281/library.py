"""
axisforge/solvers/machine_elements/bearings/ISO_16281/library.py

BearingResultsLibrary mirrors SimpleFEMResultsLibrary's pattern
(ShaftResultsReader.read(library) -> library.get(name) -> result) but for
ISO/TS 16281 bearing results.

Only two classes live here: BearingResultBundle and BearingResultsLibrary.
There's no shared per-solve result shape or Protocol scaffolding -- the
secant-stiffness projection is defined self-contained where it's computed
(BallBearingStiffness in Ball_Bearing/ball_bearing_postprocessing.py,
RollerBearingStiffness in Roller_Bearing/roller_bearing_postprocessing.py),
same as BallLoadDistributionResult / RollerLoadDistributionResult. Ball_Bearing
and Roller_Bearing stay independent by duck typing, not by satisfying a
declared shape or shared base class. This file is only the final cross-type
aggregation step.

The registry is deliberately type-aware: it's the single place every
computed result for a bearing gets gathered under one label, so a future
module (lubrication, fatigue) can find everything about a bearing without
knowing which per-type solver produced each piece. It computes nothing
itself.

Multi-row storage (Option A)
-----------------------------
bundle.load_distribution and bundle.dynamic_equivalent_load are always
LISTS -- length 1 for an ordinary single-row bearing, length i for a
multi-row bearing's i rows, index-aligned with bearing.rows. Every
consumer indexes [0] explicitly rather than the shape silently differing
per bearing.

bundle.capacity is a single (Q_ci, Q_ce) pair, never a list, even for a
multi-row bearing -- it is a geometry-level property computed once per
bearing, not once per row. Row-count effects on capacity (the i**0.7
scaling, Formula 29) live entirely inside
bearing.family.per_element_dynamic_capacity() / dynamic_multirow(), not here.

Load distribution grouping
----------------------------
add_load_distribution_library() is keyed by the SOLVER CLASS that produced
the local library (e.g. ISO16281BallSolver), not by BearingType -- a group
can mix bearing types that share the same solver (e.g. deep groove +
angular contact, both point-contact). Per-label bundle.bearing_type is
backfilled from each individual bearing's own .bearing_type, passed in via
`bearings`.

Capacity type
--------------
BearingResultBundle.capacity is whatever
bearing.family.per_element_dynamic_capacity(bearing, ...) returns -- a
plain (Q_ci, Q_ce) tuple[float, float], the same shape for every family,
ball or roller alike. No per-contact-type import needed for it here.

No circular import: BearingType is a leaf enum, safe to import for real.
The concrete per-type classes are only imported under TYPE_CHECKING -- real
objects reach the registry because callers hand them to set_*() methods,
not because the registry goes and imports anything itself.


Per-bearing results registry (BearingResultBundle, BearingResultsLibrary)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import root

from axisforge.core.machine_elements.bearings.bearing_types import BearingType

if TYPE_CHECKING:
    from axisforge.core.machine_elements.bearings.bearing import Bearing
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import SimpleFEMResultsLibrary

    from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.ball_bearing.results import (
        BallLoadDistributionResult,
        BallBearingResult,
    )
    from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.ball_bearing.postprocessing import (
        BallBearingStiffness,
        DynamicEquivalentRollingElementLoad,
    )
    from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.roller_bearing.results import (
        RollerLoadDistributionResult,
        RollerBearingResult,
    )
    from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.roller_bearing.postprocessing import (
        RollerBearingStiffness,
        LaminaDynamicEquivalentLoad,
    )

    LoadDistributionResult = BallLoadDistributionResult | RollerLoadDistributionResult
    BearingStiffness = BallBearingStiffness | RollerBearingStiffness
    CapacityResult = tuple[float, float]
    MultirowResult = BallBearingResult | RollerBearingResult


# ===========================================================================
# Per-bearing results registry
# ===========================================================================

@dataclass
class BearingResultBundle:
    """
    Every computed ISO/TS 16281 result for a single bearing, gathered under
    its label -- the "folder" for that bearing inside BearingResultsLibrary.

    Fields are optional (None until set). extra is an open slot for results
    from analyses this package doesn't define (lubrication, fatigue, ...).

    Attributes
    ----------
    label                    str
    bearing_type             BearingType | None   -- informational label only,
                             not used for dispatch (see dispatch.py)
    load_distribution        list[...] | None   -- length 1 (single-row) or i
                             (multi-row, index-aligned with bearing.rows)
    stiffness                BearingStiffness | None   -- always single, even
                             for a multi-row bearing (row 0's shared ring
                             displacement)
    capacity                 CapacityResult | None   -- (Q_ci, Q_ce), always
                             single, never row-indexed
    dynamic_equivalent_load  list[...] | None   -- same indexing as
                             load_distribution
    extra                    dict[str, object]   e.g. {"multirow_result": ...}
    """
    label: str
    bearing_type: "BearingType | None" = None
    load_distribution: "list[BallLoadDistributionResult | RollerLoadDistributionResult] | None" = None
    stiffness: "BearingStiffness | None" = None
    capacity: "CapacityResult | None" = None
    dynamic_equivalent_load: "list[DynamicEquivalentRollingElementLoad | LaminaDynamicEquivalentLoad] | None" = None
    extra: dict[str, object] = field(default_factory=dict)


class BearingResultsLibrary:
    """
    Registry of BearingResultBundle, keyed by bearing label.

        lib = BearingResultsLibrary()
        lib.add_load_distribution_library(ISO16281RollerSolver, roller_local_lib, bearings)
        lib.set_capacity("brg1a", cap)
        lib.set_dynamic_equivalent_load("brg1a", derel)
        lib.set_stiffness("brg1a", stiff)
        lib.set_extra("brg1a", "lubrication", lube_result)
        ...
        lib.get("brg1a").load_distribution[0].delta_r   # [0]: row 0 / the only row
        lib.get("brg1a").capacity   # (Q_ci, Q_ce) -- always single
        lib.load_distribution_library(ISO16281RollerSolver)   # whole sub-library back

    Computes nothing -- only organizes and retrieves results computed
    elsewhere. Each result type has exactly one slot per label; set_*()
    overwrites rather than accumulating history.

    Load distribution is handed over WHOLESALE per solver, via
    add_load_distribution_library() -- not copied label-by-label.
    """

    def __init__(self):
        self._bundles: dict[str, BearingResultBundle] = {}
        self._load_distribution_libraries: dict[type, object] = {}

    def _bundle(self, label: str) -> BearingResultBundle:
        if label not in self._bundles:
            self._bundles[label] = BearingResultBundle(label=label)
        return self._bundles[label]

    def add_load_distribution_library(self, solver_cls: type, local_library,
                                       bearings: dict) -> None:
        """
        Store an entire per-solver local library as a sub-container, keyed
        by the solver class. `local_library` must duck-type labels() and
        get(label); `bearings` is the {label: Bearing} group that produced
        it, used only to backfill bundle.bearing_type per label (a group
        can mix bearing types that share one solver, e.g. deep groove +
        angular contact).

        local_library.get(label) returns a BallBearingResult/
        RollerBearingResult container -- `.rows` unwraps it to the actual
        list[LoadDistributionResult] bundle.load_distribution holds, length
        1 here always (this is only ever called with a single-row local
        library).
        """
        self._load_distribution_libraries[solver_cls] = local_library
        for label in local_library.labels():
            bundle = self._bundle(label)
            bundle.bearing_type = bearings[label].bearing_type
            bundle.load_distribution = local_library.get(label).rows

    def load_distribution_library(self, solver_cls: type):
        """The whole local sub-library recorded for one solver class, or None."""
        return self._load_distribution_libraries.get(solver_cls)

    def set_load_distribution(self, label: str, result: "list[LoadDistributionResult]",
                              bearing_type: BearingType | None = None) -> None:
        """
        Set a single label's load distribution directly, bypassing a local
        library. Prefer add_load_distribution_library() when a whole group
        is available.

        `result` is a LIST -- length 1 for a single-row bearing, length i
        for a multi-row bearing's i rows, already unwrapped by the caller
        (e.g. mr_result.rows). This method does not wrap or unwrap for you.
        """
        bundle = self._bundle(label)
        bundle.load_distribution = result
        if bearing_type is not None:
            bundle.bearing_type = bearing_type

    def set_bearing_type(self, label: str, bearing_type: BearingType) -> None:
        self._bundle(label).bearing_type = bearing_type

    def set_stiffness(self, label: str, stiffness: "BearingStiffness") -> None:
        self._bundle(label).stiffness = stiffness

    def set_capacity(self, label: str, capacity) -> None:
        self._bundle(label).capacity = capacity

    def set_dynamic_equivalent_load(self, label: str, derel) -> None:
        """`derel` is a LIST -- length 1 (single-row) or i (multi-row),
        matching load_distribution's indexing. Not wrapped here."""
        self._bundle(label).dynamic_equivalent_load = derel

    def set_extra(self, label: str, key: str, value) -> None:
        self._bundle(label).extra[key] = value

    def get(self, label: str) -> BearingResultBundle:
        """Raises KeyError if nothing has been recorded for this label yet."""
        if label not in self._bundles:
            raise KeyError(
                f"No results recorded for bearing '{label}' -- "
                f"nothing has been set on it yet."
            )
        return self._bundles[label]

    def labels(self) -> list[str]:
        return list(self._bundles)

    def __contains__(self, label: str) -> bool:
        return label in self._bundles

    def __iter__(self):
        return iter(self._bundles.values())

    def __len__(self) -> int:
        return len(self._bundles)