"""
axisforge/solvers/machine_elements/bearings/ISO_16281/library.py

Generic solve utilities and the per-bearing results registry shared by the
per-bearing-type ISO/TS 16281 solvers (Ball_Bearing/, Roller_Bearing/).

BearingResultsLibrary mirrors SimpleFEMResultsLibrary's
pattern (ShaftResultsReader.read(library) -> library.get(name) -> result)
but for ISO/TS 16281 bearing results.

Only two classes live here: BearingResultBundle and BearingResultsLibrary --
both real data containers. There's no shared per-solve result shape or
Protocol scaffolding: BearingStiffness used to live here as a class both
contact types imported for real; the secant-stiffness projection is now
defined self-contained where it's computed instead -- BallBearingStiffness
in Ball_Bearing/ball_bearing_postprocessing.py, RollerBearingStiffness in
Roller_Bearing/roller_bearing_postprocessing.py -- same as
BallLoadDistributionResult / RollerLoadDistributionResult. Everything before
the final cross-type aggregation step is local; this file is only that
aggregation step. Ball_Bearing and Roller_Bearing stay independent by duck
typing, not by satisfying a declared shape or shared base class.

The utilities (check_bearing_ready, warn_if_floating_loaded, run_root)
contain no contact physics -- that lives in each type's own module.

The registry (BearingResultBundle, BearingResultsLibrary) is deliberately
type-aware: it's the single place every computed result for a bearing gets
gathered under one label, so a future module (lubrication, fatigue) can
find everything about a bearing without knowing which per-type solver
produced each piece. It computes nothing itself.

Multi-row storage (Option A)
-------------------------------
bundle.load_distribution and bundle.dynamic_equivalent_load are always
LISTS -- length 1 for an ordinary single-row bearing, length i for a
multi-row bearing's i rows, index-aligned with bearing.rows. This is a
deliberate design choice (Option A, chosen over a separate parallel
"row_results" slot left None for ordinary bearings): every consumer of
bundle.load_distribution / bundle.dynamic_equivalent_load indexes [0]
explicitly rather than the shape silently differing per bearing. See
rolling_bearing_solver.py's module docstring for the caller-side half of
this (solve()/postprocess_and_record() populate these as lists).

bundle.capacity is UNCHANGED -- still a single (Q_ci, Q_ce) pair, never a
list, even for a multi-row bearing. Capacity (per_element_dynamic_capacity)
is a geometry-level property computed once per bearing, not once per row --
this was already established and confirmed correct earlier this session
(the 6204 Q_ci/Q_ce value is identical across every shaft and scenario,
single-row or double-row, because it depends on contact geometry, not on
which row's load distribution was solved). Row-count effects on capacity
(the i**0.7 scaling, Formula 29) live entirely inside
bearing.family.per_element_dynamic_capacity() / dynamic_multirow(), not here.

UPDATE, this turn -- unwrap .rows, not the bare container
-----------------------------------------------------------------------
Ball_Bearing/ and Roller_Bearing/ were reconstructed so every per-type
local library (BallLoadDistributionLibrary, RollerLoadDistributionLibrary)
now stores a BallBearingResult/RollerBearingResult CONTAINER per label
(ISO16281BallSolver.solve()/ISO16281RollerSolver.solve() each wrap their
per-label solve_contact() output via .single() before recording it) --
`local_library.get(label)` returns that container, not the bare per-row
result the way it used to. add_load_distribution_library() below used to
read `[local_library.get(label)]`, wrapping the CONTAINER itself in a
length-1 list -- that produced `[BallBearingResult]` instead of the
`list[BallLoadDistributionResult]` bundle.load_distribution is documented
to hold. Fixed to `local_library.get(label).rows`, which unwraps the
container back to the actual per-row list (length 1 here, always, since
every label reaching this method comes from a single-row per-type local
library). Same fix applied on the caller side in
rolling_bearing_solver.py's solve() -- see that module's own docstring.

Similarly, ISO16281MultiRowBallSolver.solve_bearing() used to return its
own MultiRowBallLoadDistributionResult (retired), whose per-row list was
named `.row_results`; it now returns a BallBearingResult via `.multirow()`,
named `.rows` like every other result this package produces. Every
docstring reference below that used to say `.row_results` /
MultiRowBallLoadDistributionResult is updated to `.rows` / BallBearingResult
accordingly -- this file's own code never touched that attribute directly
(callers pass the already-unwrapped list to set_load_distribution()), so
only documentation changes were needed here, not logic changes.

Capacity type -- core cleanup
------------------------------
BearingResultBundle.capacity used to be typed as
BallRollingElementCapacity | RollerElementCapacity, two dataclasses that
lived in ball_bearing_solver.py / roller_bearing_solver.py and duplicated
the SAME eq.(19)-(24)/(47)-(57) formulas now owned by core/.../families/
ball/{radial,thrust}/ and roller/{radial,thrust}/. Those dataclasses are
gone (see both solver modules' docstrings); capacity is now whatever
bearing.family.per_element_dynamic_capacity(bearing, ...) returns, which is
a plain (Q_ci, Q_ce) tuple[float, float] -- no bearing-type-specific import
needed here anymore, so the TYPE_CHECKING block below is shorter than
before.

No circular import: BearingType is a leaf enum, safe to import for real.
The concrete per-type classes are only imported under TYPE_CHECKING -- real
objects reach the registry because callers hand them to set_*() methods,
not because the registry goes and imports anything itself.

Blocks
------
1. Generic, contact-agnostic utilities
2. Per-bearing results registry (BearingResultBundle, BearingResultsLibrary)

LoadDistributionResult and BearingStiffness are TYPE_CHECKING-only type
aliases here (Ball... | Roller...), not classes.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import root

from axisforge.core.machine_elements.Bearings.bearing_types import BearingType

if TYPE_CHECKING:
    from axisforge.core.machine_elements.Bearings.bearing import Bearing
    from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import SimpleFEMResultsLibrary

    from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
        BallLoadDistributionResult,
        BallBearingResult,
    )
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_postprocessing import (
        BallBearingStiffness,
        DynamicEquivalentRollingElementLoad,
    )
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import (
        RollerLoadDistributionResult,
        RollerBearingResult,
    )
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_postprocessing import (
        RollerBearingStiffness,
        LaminaDynamicEquivalentLoad,
    )

    LoadDistributionResult = BallLoadDistributionResult | RollerLoadDistributionResult
    BearingStiffness = BallBearingStiffness | RollerBearingStiffness
    # capacity is now bearing.family.per_element_dynamic_capacity(bearing, ...)'s
    # own return type -- a plain (Q_ci, Q_ce) pair -- for every family, ball or
    # roller alike. No per-contact-type import needed for it anymore.
    CapacityResult = tuple[float, float]
    # The full multi-row solve result -- stashed under
    # bundle.extra["multirow_result"] (see set_extra() usage in
    # rolling_bearing_solver.py). Not a formally-typed field on
    # BearingResultBundle (extra is a plain dict[str, object]) -- named
    # here only so this module's own docstrings have something concrete to
    # point at.
    MultirowResult = BallBearingResult | RollerBearingResult


# ===========================================================================
# Block 1 -- Generic, contact-agnostic utilities
# ===========================================================================

def check_bearing_ready(bearing, label: str, required_attrs: tuple[str, ...]) -> None:
    """Raise if the bearing hasn't had its type-specific setup called yet."""
    missing = [a for a in required_attrs if getattr(bearing, a, None) is None]
    if missing:
        raise RuntimeError(
            f"Bearing '{label}': missing {missing} -- call the type-specific "
            f"geometry/contact setup before solving."
        )


FA_FLOATING_EPS = 1e-6   # [N] Fa below this on a floating bearing is treated as zero


def warn_if_floating_loaded(bearing, label: str, Fa: float,
                            eps: float = FA_FLOATING_EPS) -> None:
    """Warn if a bearing marked arrangement='floating' has Fa != 0."""
    if getattr(bearing, "arrangement", None) == "floating" and abs(Fa) > eps:
        warnings.warn(
            f"Bearing '{label}' is floating but Fa = {Fa:.3f} N != 0 -- "
            f"check the shaft's axial load path."
        )


def run_root(fun, x0: list, tol: float) -> tuple:
    """
    scipy.optimize.root, hybr with lm fallback.
    Returns (x: np.ndarray, nfev: int, residual_norm: float, success: bool).
    """
    sol = root(fun, x0, method="hybr", tol=tol)
    if not sol.success:
        sol_lm = root(fun, x0, method="lm", tol=tol)
        if (np.linalg.norm(np.atleast_1d(sol_lm.fun)) <
                np.linalg.norm(np.atleast_1d(sol.fun))):
            sol = sol_lm

    return (np.atleast_1d(sol.x),
            int(getattr(sol, "nfev", 0)),
            float(np.linalg.norm(np.atleast_1d(sol.fun))),
            bool(sol.success))


# ===========================================================================
# Block 2 -- Per-bearing results registry
# ===========================================================================

@dataclass
class BearingResultBundle:
    """
    Every computed ISO/TS 16281 result for a single bearing, gathered under
    its label -- the "folder" for that bearing inside BearingResultsLibrary.

    Fields are optional (None until set). bearing_type is set alongside
    load_distribution so a consumer can dispatch on it without the original
    Bearing instance in scope. extra is an open slot for results from
    analyses this package doesn't define (lubrication, fatigue, ...).

    Attributes
    ----------
    label                    str
    bearing_type             BearingType | None
    load_distribution        list[BallLoadDistributionResult | RollerLoadDistributionResult] | None
                             length 1 (single-row) or i (multi-row, index-
                             aligned with bearing.rows) -- see module
                             docstring, "Multi-row storage" section.
    stiffness                BearingStiffness | None
                             ALWAYS single, even for a multi-row bearing --
                             computed from row 0's shared ring displacement
                             (see rolling_bearing_solver.py's
                             postprocess_and_record()).
    capacity                 CapacityResult | None    (Q_ci, Q_ce) from
                             bearing.family.per_element_dynamic_capacity() --
                             ALWAYS single, never row-indexed, even for a
                             multi-row bearing -- see module docstring.
    dynamic_equivalent_load  list[DynamicEquivalentRollingElementLoad | LaminaDynamicEquivalentLoad] | None
                             length 1 (single-row) or i (multi-row) -- same
                             indexing as load_distribution.
    extra                    dict[str, object]   e.g. {"lubrication": ...,
                             "multirow_result": BallBearingResult}   -- the
                             full multi-row solve (row f_r/f_a split,
                             n_iter, residual, ok -- see
                             ball_bearing_results.py's BallBearingResult.
                             multirow()) is stashed here for a multi-row
                             bearing, since none of that fits
                             load_distribution's per-row list.
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
        lib.add_load_distribution_library(BearingType.CYLINDRICAL_ROLLER, roller_local_lib)
        lib.set_capacity("brg1a", cap)
        lib.set_dynamic_equivalent_load("brg1a", derel)
        lib.set_stiffness("brg1a", stiff)
        lib.set_extra("brg1a", "lubrication", lube_result)
        ...
        lib.get("brg1a").load_distribution[0].delta_r   # [0]: row 0 / the only row
        lib.get("brg1a").capacity   # (Q_ci, Q_ce) -- always single, see below
        lib.load_distribution_library(BearingType.CYLINDRICAL_ROLLER)   # whole sub-library back

    Computes nothing -- only organizes and retrieves results computed
    elsewhere. Each result type has exactly one slot per label; set_*()
    overwrites rather than accumulating history.

    Load distribution is handed over WHOLESALE per type, via
    add_load_distribution_library() -- not copied label-by-label. The
    orchestrator calls each per-type solver once per BearingType group and
    hands the whole local library (RollerLoadDistributionLibrary,
    BallLoadDistributionLibrary, ...) here as a sub-container.
    bundle.load_distribution is populated FROM that same object's entries
    (via `.rows`, see add_load_distribution_library()'s own docstring), so
    `results.get(label).load_distribution` works without needing the type.
    """

    def __init__(self):
        self._bundles: dict[str, BearingResultBundle] = {}
        self._load_distribution_libraries: dict[BearingType, object] = {}

    def _bundle(self, label: str) -> BearingResultBundle:
        if label not in self._bundles:
            self._bundles[label] = BearingResultBundle(label=label)
        return self._bundles[label]

    def add_load_distribution_library(self, bearing_type: BearingType, local_library) -> None:
        """
        Store an entire per-type local library as a sub-container, keyed by
        BearingType. `local_library` must duck-type labels() and get(label).

        Back-fills bundle.bearing_type and bundle.load_distribution for
        every label already in `local_library`, by reference.

        local_library.get(label) returns a BallBearingResult/
        RollerBearingResult CONTAINER (see module docstring, "unwrap .rows,
        not the bare container") -- `.rows` is the actual
        list[LoadDistributionResult] bundle.load_distribution is documented
        to hold. Every label coming through this path is single-row by
        construction (this is only ever called with a per-type LOCAL
        library, e.g. BallLoadDistributionLibrary, which never holds a
        multi-row result -- see rolling_bearing_solver.py's solve()), so
        `.rows` is length 1 here, always -- but reading it off the
        container rather than re-wrapping a bare object keeps this uniform
        with how a multi-row bundle's load_distribution is populated
        elsewhere (set_load_distribution(), called directly with
        mr_result.rows -- also already a list, also never re-wrapped).
        """
        self._load_distribution_libraries[bearing_type] = local_library
        for label in local_library.labels():
            bundle = self._bundle(label)
            bundle.bearing_type = bearing_type
            bundle.load_distribution = local_library.get(label).rows

    def load_distribution_library(self, bearing_type: BearingType):
        """The whole local sub-library recorded for one BearingType, or None."""
        return self._load_distribution_libraries.get(bearing_type)

    def set_load_distribution(self, label: str, result: "list[LoadDistributionResult]",
                              bearing_type: BearingType | None = None) -> None:
        """
        Set a single label's load distribution directly, bypassing a local
        library. Prefer add_load_distribution_library() when a whole
        per-type group is available.

        `result` is a LIST -- length 1 for a single-row bearing, length i
        for a multi-row bearing's i rows. The caller passes an
        already-unwrapped list, e.g. rolling_bearing_solver.py passes
        mr_result.rows directly for a multi-row bearing (mr_result being a
        BallBearingResult, from ISO16281MultiRowBallSolver.solve_bearing()'s
        .multirow() classmethod), or local_library.get(label).rows for a
        single-row one. This method does not wrap or unwrap for you -- see
        module docstring, "Multi-row storage" section.
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
        """
        `derel` is a LIST -- length 1 (single-row) or i (multi-row),
        matching load_distribution's indexing. Not wrapped here -- see
        set_load_distribution()'s docstring, same convention. Typically the
        return value of adapter.derel_cls.from_bearing_result(...) in
        rolling_bearing_solver.py's postprocess_and_record(), which already
        returns a list of exactly this shape.
        """
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