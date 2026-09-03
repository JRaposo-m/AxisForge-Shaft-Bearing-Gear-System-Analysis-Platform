"""axisforge/fixtures/studies/bearing/load_distribution/no_lubrication/results_library.py

Registry of every computed ISO/TS 16281 result for a bearing, keyed by
label -- the cross-type aggregation step (point-contact and line-contact
bearings alike land here). Computes nothing; only organizes and retrieves
results computed by the solvers in axisforge/solvers/.

    lib = BearingResultsLibrary()
    lib.add_load_distribution_results(ISO16281RollerSolver, roller_results, bearings)
    lib.set_capacity("brg1a", cap)
    lib.set_dynamic_equivalent_load("brg1a", derel)
    lib.set_stiffness("brg1a", stiff)
    lib.set_extra("brg1a", "lubrication", lube_result)
    lib.get("brg1a").load_distribution[0].delta_r
    lib.get("brg1a").capacity   # (Q_ci, Q_ce)

load_distribution and dynamic_equivalent_load are always length-1 lists
(single-row only, see contracts/bearing/). capacity is a single (Q_ci, Q_ce)
pair, never a list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from axisforge.core.machine_elements.bearings.bearing_types import BearingType

if TYPE_CHECKING:
    from axisforge.core.machine_elements.bearings.bearing import Bearing

    from axisforge.results.bearings.load_distribution.single_row.ball_bearing_results import (
        BallLoadDistributionResult,
        BallBearingResult,
    )
    from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.ball_bearing.postprocessing import (
        BallBearingStiffness,
        DynamicEquivalentRollingElementLoad,
    )
    from axisforge.results.bearings.load_distribution.single_row.roller_bearing_results import (
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
    SingleRowResult = BallBearingResult | RollerBearingResult


@dataclass
class BearingResultBundle:
    """Every computed ISO/TS 16281 result for one bearing, under its label.

    Fields are optional (None until set). extra is an open slot for results
    from analyses this package doesn't define (lubrication, fatigue, ...).
    """
    label: str
    bearing_type: "BearingType | None" = None
    load_distribution: "list[BallLoadDistributionResult | RollerLoadDistributionResult] | None" = None
    stiffness: "BearingStiffness | None" = None
    capacity: "CapacityResult | None" = None
    dynamic_equivalent_load: "list[DynamicEquivalentRollingElementLoad | LaminaDynamicEquivalentLoad] | None" = None
    extra: dict[str, object] = field(default_factory=dict)


class BearingResultsLibrary:
    """Registry of BearingResultBundle, keyed by bearing label.

    Computes nothing -- only organizes and retrieves results computed
    elsewhere. Each result type has exactly one slot per label; set_*()
    overwrites rather than accumulating history.
    """

    def __init__(self):
        self._bundles: dict[str, BearingResultBundle] = {}
        self._load_distribution_by_solver: dict[type, dict[str, "SingleRowResult"]] = {}

    def _bundle(self, label: str) -> BearingResultBundle:
        if label not in self._bundles:
            self._bundles[label] = BearingResultBundle(label=label)
        return self._bundles[label]

    def add_load_distribution_results(self, solver_cls: type,
                                       results: dict[str, "SingleRowResult"],
                                       bearings: dict) -> None:
        """Store a whole solver's {label: result} batch, keyed by solver class.

        `results` is what ISO16281BallSolver.solve() / ISO16281RollerSolver.solve()
        return directly -- a plain dict, not a registry object. `bearings` is
        the {label: Bearing} group that produced it, used to backfill
        bundle.bearing_type per label.
        """
        self._load_distribution_by_solver[solver_cls] = results
        for label, result in results.items():
            bundle = self._bundle(label)
            bundle.bearing_type = bearings[label].bearing_type
            bundle.load_distribution = result.rows

    def load_distribution_results(self, solver_cls: type):
        """The whole {label: result} dict recorded for one solver class, or None."""
        return self._load_distribution_by_solver.get(solver_cls)

    def set_load_distribution(self, label: str, result: "list[LoadDistributionResult]",
                              bearing_type: BearingType | None = None) -> None:
        """Set a single label's load distribution directly, bypassing a batch.

        `result` is a list -- length 1, already unwrapped by the caller
        (e.g. some_result.rows).
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