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
    )
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_postprocessing import (
        BallBearingStiffness,
        DynamicEquivalentRollingElementLoad,
    )
    from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing_results import (
        RollerLoadDistributionResult,
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
    load_distribution        BallLoadDistributionResult | RollerLoadDistributionResult | None
    stiffness                BearingStiffness | None
    capacity                 CapacityResult | None    (Q_ci, Q_ce) from
                             bearing.family.per_element_dynamic_capacity()
    dynamic_equivalent_load  DynamicEquivalentRollingElementLoad | LaminaDynamicEquivalentLoad | None
    extra                    dict[str, object]   e.g. {"lubrication": ...}
    """
    label: str
    bearing_type: "BearingType | None" = None
    load_distribution: "BallLoadDistributionResult | RollerLoadDistributionResult | None" = None
    stiffness: "BearingStiffness | None" = None
    capacity: "CapacityResult | None" = None
    dynamic_equivalent_load: "DynamicEquivalentRollingElementLoad | LaminaDynamicEquivalentLoad | None" = None
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
        lib.get("brg1a").load_distribution.delta_r
        lib.get("brg1a").capacity   # (Q_ci, Q_ce)
        lib.load_distribution_library(BearingType.CYLINDRICAL_ROLLER)   # whole sub-library back

    Computes nothing -- only organizes and retrieves results computed
    elsewhere. Each result type has exactly one slot per label; set_*()
    overwrites rather than accumulating history.

    Load distribution is handed over WHOLESALE per type, via
    add_load_distribution_library() -- not copied label-by-label. The
    orchestrator calls each per-type solver once per BearingType group and
    hands the whole local library (RollerLoadDistributionLibrary,
    BallLoadDistributionLibrary, ...) here as a sub-container.
    bundle.load_distribution is populated from the SAME object reference,
    so `results.get(label).load_distribution` works without needing the type.
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
        """
        self._load_distribution_libraries[bearing_type] = local_library
        for label in local_library.labels():
            bundle = self._bundle(label)
            bundle.bearing_type = bearing_type
            bundle.load_distribution = local_library.get(label)

    def load_distribution_library(self, bearing_type: BearingType):
        """The whole local sub-library recorded for one BearingType, or None."""
        return self._load_distribution_libraries.get(bearing_type)

    def set_load_distribution(self, label: str, result: "LoadDistributionResult",
                              bearing_type: BearingType | None = None) -> None:
        """
        Set a single label's load distribution directly, bypassing a local
        library. Prefer add_load_distribution_library() when a whole
        per-type group is available.
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