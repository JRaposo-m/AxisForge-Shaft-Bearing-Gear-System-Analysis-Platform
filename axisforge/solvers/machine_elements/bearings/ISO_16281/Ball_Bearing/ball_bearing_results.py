"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_results.py

The LOCAL library for point-contact (ball) bearings: the result *shape* a
solve produces (BallLoadDistributionResult) together with the registry that
holds one per bearing label (BallLoadDistributionLibrary). "results" IS the
local library — not a separate concept from it, mirroring Roller_Bearing/
roller_bearing_results.py on the line-contact side.

BallLoadDistributionResult has NO import from the global library.py, real or
TYPE_CHECKING — it defines the full field set directly (delta_r, delta_a,
psi, phi_Fr, delta_j, alpha_j, Mz, n_iter, residual, ok). Point contact adds
no per-lamina data, so unlike RollerLoadDistributionResult there is nothing
extra to carry — but the class is still defined here, fully self-contained,
rather than borrowed from the global library, so this module has ZERO
runtime dependency on library.py, not even a one-way one. It only happens
to structurally match library.LoadDistributionResult (a Protocol); it does
not depend on it. BallLoadDistributionLibrary itself never referenced that
import directly either — it only stores/hands back BallLoadDistributionResult
instances someone else already built.

Populated by ISO16281BallSolver.solve() (ball_bearing.py) and read back by
rolling_bearing_solver.RollingBearingSolver.solve(), which hands the whole
BallLoadDistributionLibrary object over to the FINAL, cross-type
BearingResultsLibrary in the global library.py via
add_load_distribution_library() — as one sub-container per BearingType, not
copied label-by-label. This module itself never touches
BearingResultsLibrary.

ball_bearing_postprocessing.py imports BallLoadDistributionResult from here
(not from ball_bearing.py); ball_bearing.py never imports
ball_bearing_postprocessing.py, and this module never imports either of them.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# BallLoadDistributionResult — self-contained, no import from library.py
# ---------------------------------------------------------------------------

class BallLoadDistributionResult:
    """
    Result of a single point-contact (ball) bearing internal load
    distribution solve — every field defined directly here. Fully
    self-contained: no import, subclassing, or other runtime dependency on
    library.py or on Roller_Bearing — it only happens to structurally match
    library.LoadDistributionResult (a Protocol), it does not depend on it.

    Attributes
    ----------
    delta_r   float [mm]        radial ring displacement in the resultant-force plane
    delta_a   float [mm]        axial ring displacement
    psi       float [rad]       prescribed ring misalignment in the resultant-force plane
    phi_Fr    float [rad]       angle of resultant Fr in the global frame (output/plots only)
    delta_j   ndarray(Z,) [mm]  elastic deflection per rolling element
    alpha_j   ndarray(Z,) [rad] effective contact angle per element
    Mz        float [N*mm]      moment reaction
    n_iter    int               solver function evaluations
    residual  float [N]         final ||R||
    ok        bool              solver convergence flag

    Contact force per element (Q_j = cp * delta_j^n) is computed by
    ball_bearing_postprocessing.Q_j(), not stored here.
    """
    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
    )

    def __init__(self, *, delta_r, delta_a, psi, phi_Fr,
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


# ---------------------------------------------------------------------------
# BallLoadDistributionLibrary — LOCAL registry, point-contact bearings only
# ---------------------------------------------------------------------------

class BallLoadDistributionLibrary:
    """
    Local registry of BallLoadDistributionResult, keyed by bearing label —
    scoped to point-contact bearings (DEEP_GROOVE_BALL, ANGULAR_CONTACT).

    Populated by ISO16281BallSolver.solve() (in ball_bearing.py) and read
    back by rolling_bearing_solver.RollingBearingSolver.solve(), which
    merges this together with whatever other per-type local libraries
    (e.g. RollerLoadDistributionLibrary) were produced for the rest of a
    shaft's bearing set. This class is NOT the final, cross-type registry —
    that is BearingResultsLibrary in the global library.py one level up,
    which also holds capacity/dynamic_equivalent_load/stiffness/extra for
    every bearing regardless of type. This one only ever holds the load
    distribution result for point-contact bearings.

    Same shape as SimpleFEMResultsLibrary / BearingResultsLibrary /
    RollerLoadDistributionLibrary (set/get/labels/__contains__/__iter__/
    __len__) — satisfies library.LoadDistributionLibrary structurally,
    without importing it.
    """

    def __init__(self):
        self._results: dict[str, BallLoadDistributionResult] = {}

    def set(self, label: str, result: BallLoadDistributionResult) -> None:
        self._results[label] = result

    def get(self, label: str) -> BallLoadDistributionResult:
        """Raises KeyError if nothing has been recorded for this label yet."""
        if label not in self._results:
            raise KeyError(
                f"No load distribution recorded for bearing '{label}' — "
                f"nothing has been set on it yet."
            )
        return self._results[label]

    def labels(self) -> list[str]:
        """All bearing labels with a recorded result."""
        return list(self._results)

    def __contains__(self, label: str) -> bool:
        return label in self._results

    def __iter__(self):
        return iter(self._results.values())

    def __len__(self) -> int:
        return len(self._results)