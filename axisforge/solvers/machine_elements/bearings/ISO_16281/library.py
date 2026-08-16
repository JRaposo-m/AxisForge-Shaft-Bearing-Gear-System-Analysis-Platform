"""
axisforge/solvers/machine_elements/bearings/ISO_16281/library.py

Shared data contract, generic utilities, and per-bearing results registry
for the per-bearing-type ISO/TS 16281 solvers (ISO_16281_ball_bearing.py,
and a future ISO_16281_roller_bearing.py).

Renamed from common.py — this module now also owns BearingResultsLibrary,
which organizes computed results per bearing label. It mirrors the existing
SimpleFEMResultsLibrary pattern (ShaftResultsReader.read(library) ->
library.get(name) -> result with .x, .v_xz, .bearing_nodes, ...) but for
ISO/TS 16281 bearing results instead of FEM shaft results.

This module deliberately contains NO contact physics — no kinematics, no
capacity formulas, no load-deflection exponents. Those differ between point
contact (ball) and line contact (roller) and belong entirely inside each
type's own module (e.g. RollingElementCapacity.radial() lives in
ISO_16281_ball_bearing.py, not here — nothing here computes anything).
Ball and roller solvers do not import each other; they both import from
here and both hand their computed results to whoever wants to record them
in a BearingResultsLibrary, which is the only place a bearing's various
result types get gathered under one label.

Blocks
------
1. Per-solve result shapes (shared across contact types)
     LoadDistributionResult, BearingStiffness
2. Orchestrator contract
     RollingBearingTypeSolver (Protocol)
3. Generic, contact-agnostic utilities
     check_bearing_ready(), warn_if_floating_loaded(), run_root()
4. Per-bearing results registry
     BearingResultBundle, BearingResultsLibrary
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import numpy as np
from scipy.optimize import root

# Imported only for static type-checking (Pylance/mypy) — never executed at
# runtime, so this cannot create a circular import with bearing.py,
# shaft_system.py, or static_analysis.py regardless of what they import.
if TYPE_CHECKING:
    from axisforge.core.machine_elements.Bearings.bearing import Bearing
    from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import SimpleFEMResultsLibrary


# ===========================================================================
# Block 1 — Per-solve result shapes (shared across contact types)
# ===========================================================================

class LoadDistributionResult:
    """
    Result of a single bearing internal load distribution solve.

    Shape is identical regardless of contact type (point/ball or line/roller)
    — only how delta_j/alpha_j were computed differs, inside each type solver.

    Attributes
    ----------
    delta_r   float [mm]        radial ring displacement in the resultant-force plane
    delta_a   float [mm]        axial ring displacement
    psi       float [rad]       prescribed ring misalignment in the resultant-force plane
    phi_Fr    float [rad]       angle of resultant Fr in the global frame (output/plots only)
    delta_j   ndarray(Z,) [mm]  elastic deflection per rolling element
    alpha_j   ndarray(Z,) [rad] effective contact angle per element
    Mz        float [N·mm]      moment reaction
    n_iter    int               solver function evaluations
    residual  float [N]         final ||R||
    ok        bool              solver convergence flag

    Contact force per element is type-specific (Q_j = cp · delta_j^n, n depends
    on contact type) and is computed by each type solver's own Q_j().
    """

    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
    )

    def __init__(self, delta_r, delta_a, psi, phi_Fr,
                 delta_j, alpha_j, Mz, n_iter, residual, ok):
        self.delta_r  = delta_r
        self.delta_a  = delta_a
        self.psi      = psi
        self.phi_Fr   = phi_Fr
        self.delta_j  = delta_j
        self.alpha_j  = alpha_j
        self.Mz       = Mz
        self.n_iter   = n_iter
        self.residual = residual
        self.ok       = ok


class BearingStiffness:
    """
    Secant stiffness of a bearing, decomposed onto the XZ / XY / axial axes.

    Built from a converged LoadDistributionResult by projecting delta_r back
    onto the global axes:

        delta_r_xz = delta_r · cos(phi_Fr)
        delta_r_xy = delta_r · sin(phi_Fr)
        Kr_xz = Fr_xz / delta_r_xz
        Kr_xy = Fr_xy / delta_r_xy
        Ka    = Fa    / delta_a

    Any stiffness component is set to float('inf') when the corresponding
    displacement is negligible (rigid in that direction).

    Fr_xz, Fr_xy, Fa and delta_a are NOT stored on this object — Fr_xz/Fr_xy/Fa
    already live in BearingNodeData (SimpleFEMResultsLibrary) and delta_a
    already lives on the LoadDistributionResult that from_result() was built
    from; the caller has all three in scope by construction, since it had to
    pass them in to call from_result() in the first place. Keeping them here
    too would just be a second copy of data that already exists elsewhere.
    Only what from_result() actually computes — the projected displacements
    and the stiffnesses/regime derived from them — is stored.

    This is the single implementation of the stiffness projection — every
    type solver's bearing_stiffness() convenience method is a thin wrapper
    around from_result(), not a second implementation.

    Ka_regime
    ---------
    "no_load"           Fa = 0
    "engaged"           delta_a >= 0 — axial contact active
    "closing_clearance" delta_a < 0  — axial clearance not yet closed

    Attributes
    ----------
    label      str
    delta_r_xz float [mm]      delta_r projected onto the XZ plane
    Kr_xz      float [N/mm]
    delta_r_xy float [mm]      delta_r projected onto the XY plane
    Kr_xy      float [N/mm]
    Ka         float | None   [N/mm]
    Ka_regime  str   | None
    """

    __slots__ = (
        "label",
        "delta_r_xz", "Kr_xz",
        "delta_r_xy", "Kr_xy",
        "Ka", "Ka_regime",
    )

    def __init__(self, label,
                 delta_r_xz, Kr_xz,
                 delta_r_xy, Kr_xy,
                 Ka=None, Ka_regime=None):
        self.label      = label
        self.delta_r_xz = delta_r_xz
        self.Kr_xz      = Kr_xz
        self.delta_r_xy = delta_r_xy
        self.Kr_xy      = Kr_xy
        self.Ka         = Ka
        self.Ka_regime  = Ka_regime

    @classmethod
    def from_result(cls,
                    label: str,
                    result: LoadDistributionResult,
                    Fr_xz: float,
                    Fr_xy: float,
                    Fa: float,
                    eps: float = 1e-9) -> "BearingStiffness":
        """
        Build from a converged LoadDistributionResult (any contact type).

        Fr_xz, Fr_xy, Fa are used only to compute Kr_xz/Kr_xy/Ka here — they
        are not retained on the returned object (see class docstring).
        """
        delta_r_xz = result.delta_r * np.cos(result.phi_Fr)
        delta_r_xy = result.delta_r * np.sin(result.phi_Fr)
        delta_a    = result.delta_a

        Kr_xz = (Fr_xz / delta_r_xz) if abs(delta_r_xz) > eps else float("inf")
        Kr_xy = (Fr_xy / delta_r_xy) if abs(delta_r_xy) > eps else float("inf")

        if Fa == 0.0:
            Ka, regime = float("inf"), "no_load"
        elif abs(delta_a) > eps:
            Ka     = Fa / delta_a
            regime = "engaged" if delta_a >= 0.0 else "closing_clearance"
        else:
            Ka, regime = float("inf"), "engaged"

        return cls(label=label,
                   delta_r_xz=delta_r_xz, Kr_xz=Kr_xz,
                   delta_r_xy=delta_r_xy, Kr_xy=Kr_xy,
                   Ka=Ka, Ka_regime=regime)


# ===========================================================================
# Block 2 — Orchestrator contract
# ===========================================================================

class RollingBearingTypeSolver(Protocol):
    """
    What rolling_bearing_solver.RollingBearingSolver expects from any
    per-type solver (ISO16281BallSolver, and later a roller equivalent).

    This is a typing.Protocol, not a base class: a type solver satisfies it
    by matching the shape, without importing this module or subclassing
    anything. That keeps ball and roller solvers independent of each other
    and of the orchestrator — only the orchestrator needs to know this shape.
    """

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              psi_override: dict[str, float] | None = None,
              ) -> dict[str, LoadDistributionResult]:
        ...


# ===========================================================================
# Block 3 — Generic, contact-agnostic utilities
# ===========================================================================

def check_bearing_ready(bearing, label: str, required_attrs: tuple[str, ...]) -> None:
    """
    Raise if the bearing hasn't had its type-specific setup called yet
    (e.g. setup_internal_geometry() / compute_hertz_point_contact() for
    ball bearings — each type solver defines its own required_attrs).
    """
    missing = [a for a in required_attrs if getattr(bearing, a, None) is None]
    if missing:
        raise RuntimeError(
            f"Bearing '{label}': missing {missing} — call the type-specific "
            f"geometry/contact setup before solving."
        )


# Axial reaction threshold [N] below which a "floating" bearing is treated as
# carrying no axial load. Anything above this on a floating bearing is almost
# certainly a modelling error upstream (axial load path not statically
# determinate, or the wrong bearing marked as locating).
FA_FLOATING_EPS = 1e-6


def warn_if_floating_loaded(bearing, label: str, Fa: float,
                            eps: float = FA_FLOATING_EPS) -> None:
    """Warn if a bearing marked arrangement='floating' has Fa != 0."""
    if getattr(bearing, "arrangement", None) == "floating" and abs(Fa) > eps:
        warnings.warn(
            f"Bearing '{label}' is floating but Fa = {Fa:.3f} N != 0 — "
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
# Block 4 — Per-bearing results registry
# ===========================================================================

@dataclass
class BearingResultBundle:
    """
    Every computed ISO/TS 16281 result for a single bearing, gathered under
    its label — the "folder" for that bearing inside BearingResultsLibrary.

    Fields are optional (None until set): a caller may run only the load
    distribution solve for a given case without ever computing capacity or
    dynamic equivalent load, and vice versa.

    capacity and dynamic_equivalent_load are typed as plain `object` rather
    than importing RollingElementCapacity / DynamicEquivalentRollingElementLoad
    directly — those classes AND the formulas that compute them live in each
    contact type's own module (ISO_16281_ball_bearing.py, and later a roller
    equivalent), which this module must not depend on. Any object with the
    matching fields (Q_ci/Q_ce, Q_ei/Q_ee, ...) fits here regardless of
    contact type.

    Attributes
    ----------
    label                    str
    load_distribution        LoadDistributionResult | None
    stiffness                BearingStiffness | None
    capacity                 object | None   — e.g. RollingElementCapacity
    dynamic_equivalent_load  object | None   — e.g. DynamicEquivalentRollingElementLoad
    """
    label: str
    load_distribution: LoadDistributionResult | None = None
    stiffness: BearingStiffness | None = None
    capacity: object | None = None
    dynamic_equivalent_load: object | None = None


class BearingResultsLibrary:
    """
    Registry of BearingResultBundle, keyed by bearing label.

    Mirrors SimpleFEMResultsLibrary's usage pattern — something populates it
    incrementally (a solver, or the calling script right after each
    computation), and later code reads back a bearing's full result set from
    one place instead of threading four separate loose dicts around:

        lib = BearingResultsLibrary()
        lib.set_load_distribution("brg1a", result)
        lib.set_capacity("brg1a", cap)
        lib.set_dynamic_equivalent_load("brg1a", derel)
        lib.set_stiffness("brg1a", stiff)
        ...
        lib.get("brg1a").load_distribution.delta_r
        lib.get("brg1a").capacity.Q_ci
        lib.get("brg1a").stiffness.Kr_xz

    This class computes nothing — it only organizes and retrieves results
    computed elsewhere (by a type solver's own classes). No duplication:
    each result type has exactly one slot per bearing label; set_*()
    overwrites that slot rather than accumulating a history.
    """

    def __init__(self):
        self._bundles: dict[str, BearingResultBundle] = {}

    def _bundle(self, label: str) -> BearingResultBundle:
        if label not in self._bundles:
            self._bundles[label] = BearingResultBundle(label=label)
        return self._bundles[label]

    def set_load_distribution(self, label: str, result: LoadDistributionResult) -> None:
        self._bundle(label).load_distribution = result

    def set_stiffness(self, label: str, stiffness: BearingStiffness) -> None:
        self._bundle(label).stiffness = stiffness

    def set_capacity(self, label: str, capacity) -> None:
        self._bundle(label).capacity = capacity

    def set_dynamic_equivalent_load(self, label: str, derel) -> None:
        self._bundle(label).dynamic_equivalent_load = derel

    def get(self, label: str) -> BearingResultBundle:
        """Raises KeyError if nothing has been recorded for this label yet."""
        if label not in self._bundles:
            raise KeyError(
                f"No results recorded for bearing '{label}' — "
                f"nothing has been set on it yet."
            )
        return self._bundles[label]

    def labels(self) -> list[str]:
        """All bearing labels with at least one recorded result."""
        return list(self._bundles)

    def __contains__(self, label: str) -> bool:
        return label in self._bundles

    def __iter__(self):
        return iter(self._bundles.values())

    def __len__(self) -> int:
        return len(self._bundles)