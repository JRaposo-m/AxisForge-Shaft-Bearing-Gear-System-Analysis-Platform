"""
axisforge/solvers/machine_elements/bearings/ISO_16281/rolling_bearing_solver.py

Orchestrator for per-bearing-type ISO/TS 16281 solves.

A shaft's bearing set can legitimately mix contact types — e.g. a locating
deep-groove ball bearing plus a floating cylindrical roller bearing on the
same shaft, a very common real gearbox arrangement. RollingBearingSolver
groups the bearings passed to it by BearingType, dispatches each group to
the registered per-type solver, and merges the results back into a single
{label: LoadDistributionResult} dict — in the same order the caller supplied
`bearings`, regardless of how the groups were split internally.

This is the ONLY module in the bearings/solvers package that knows about
more than one contact type. Each per-type solver (ISO16281BallSolver,
ISO16281RollerSolver) is self-contained and does not import, or get
imported by, any other per-type solver — see library.py for what they do
share (the neutral output contract, generic utilities, and the
BearingResultsLibrary registry — no contact physics in any of it).

Dispatch mirrors axisforge/core/machine_elements/Bearings/bearing_factory.py:
a _SOLVER_MAP keyed by BearingType, extended as new types are implemented,
raising NotImplementedError (not a silent skip) for anything not yet
registered — so an unimplemented bearing type fails loudly at solve time
instead of quietly vanishing from the results.

Post-processing utilities that need the contact-type physics baked in
(Q_j, contact_distribution, bearing_stiffness, minimum_axial_load,
DynamicEquivalentRollingElementLoad, *ElementCapacity) are NOT re-exposed
here — call them on the concrete per-type solver/module for a bearing whose
type you already know (e.g. Ball_Bearing.ball_bearing.ISO16281BallSolver.
minimum_axial_load(...), or Roller_Bearing.roller_bearing_postprocessing.
bearing_stiffness(...)). The orchestrator only unifies solve(), which is
the one operation that must span mixed-type bearing sets on a shaft. A
caller iterating over a mixed result dict typically dispatches on
`bearing.bearing_type` to pick the matching per-type module — see
fixtures/integration/design_load_distribution_mixed_bearings.py for the
pattern (a small BEARING_KIT registry keyed by BearingType).
"""
from __future__ import annotations

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    LoadDistributionResult,
    RollingBearingTypeSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing import (
    ISO16281BallSolver,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Roller_Bearing.roller_bearing import (
    ISO16281RollerSolver,
)
from axisforge.config import SOLVER_TOLERANCE


# ---------------------------------------------------------------------------
# Type dispatch table — extend as new per-type solvers are implemented.
# Point-contact families (deep groove, angular contact) share the same
# kinematics, so both map to ISO16281BallSolver. CYLINDRICAL_ROLLER maps to
# ISO16281RollerSolver — line contact, §5.2 lamina model, no axial capacity.
# TAPERED_ROLLER / SPHERICAL_ROLLER share the lamina mechanics but need an
# additional coordinate transform (cone half-angle / crown-osculation) that
# ISO16281RollerSolver does not implement — not wired in yet.
# ---------------------------------------------------------------------------

_SOLVER_MAP: dict[BearingType, type] = {
    BearingType.DEEP_GROOVE_BALL:  ISO16281BallSolver,
    BearingType.ANGULAR_CONTACT:   ISO16281BallSolver,
    BearingType.CYLINDRICAL_ROLLER: ISO16281RollerSolver,
    # BearingType.TAPERED_ROLLER:     ISO16281RollerSolver-derived,  # Phase 2
    # BearingType.SPHERICAL_ROLLER:   ISO16281RollerSolver-derived,  # Phase 2
}


class RollingBearingSolver:
    """
    Orchestrates ISO/TS 16281 solves across a shaft's full bearing set,
    dispatching each bearing to the solver registered for its BearingType.

    Parameters
    ----------
    tol       float   residual tolerance, forwarded to every per-type solver
    psi_input bool    forwarded to every per-type solver — see
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
              ) -> dict[str, LoadDistributionResult]:
        """
        Solve every bearing in `bearings`, regardless of contact type.

        Parameters
        ----------
        shaft_system  : ShaftSystem — fully resolved
        bearings      : {label: Bearing} — may mix BearingType values
        library       : SimpleFEMResultsLibrary — must contain results for
                        shaft_system.name
        psi_override  : {label: psi [rad]} — forwarded as-is to whichever
                        per-type solver owns that label; see each per-type
                        solver's solve() for the convention.

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

            solver: RollingBearingTypeSolver = solver_cls(
                tol=self.tol, psi_input=self.psi_input)
            group_override = (
                {lbl: psi_override[lbl] for lbl in group if lbl in psi_override}
                if psi_override else None
            )
            merged.update(solver.solve(shaft_system, group, library,
                                       psi_override=group_override))

        # Preserve the caller's original ordering rather than the grouping order.
        return {label: merged[label] for label in bearings}