"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing_results.py

The LOCAL library for CYLINDRICAL_ROLLER bearings: the result *shape* a
solve produces (RollerLoadDistributionResult) together with the registry
that holds one per bearing label (RollerLoadDistributionLibrary). "results"
IS the local library — not a separate concept from it.

RollerLoadDistributionResult has NO import from the global library.py, real
or TYPE_CHECKING — it does not subclass or otherwise reference anything
there. It defines the full field set directly (the base load-distribution
fields plus the per-lamina extras), so this module is fully self-contained:
it only structurally matches library.LoadDistributionResult (a Protocol),
it does not depend on it. RollerLoadDistributionLibrary likewise never
imports the global library; it only stores/hands back
RollerLoadDistributionResult instances someone else already built.

Populated by ISO16281RollerSolver.solve() (roller_bearing.py) and read back
by rolling_bearing_solver.RollingBearingSolver.solve(), which hands the
whole RollerLoadDistributionLibrary object over to the FINAL, cross-type
BearingResultsLibrary in the global library.py via
add_load_distribution_library() — as one sub-container per BearingType,
not copied label-by-label. This module itself never touches
BearingResultsLibrary.

roller_bearing_postprocessing.py imports RollerLoadDistributionResult from
here (not from roller_bearing.py); roller_bearing.py never imports
roller_bearing_postprocessing.py, and this module never imports either of
them.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# RollerLoadDistributionResult — self-contained, no import from library.py
# ---------------------------------------------------------------------------

class RollerLoadDistributionResult:
    """
    Result of a single CYLINDRICAL_ROLLER bearing internal load distribution
    solve — every field defined directly here, with the per-lamina data the
    §5.2 lamina model produces added alongside. Fully self-contained: no
    import, subclassing, or other runtime dependency on library.py or on
    Ball_Bearing — it only happens to structurally match
    library.LoadDistributionResult (a Protocol), it does not depend on it.

    Base fields keep the same meaning documented in library.py, adapted for
    line contact:
      delta_r  : radial ring displacement [mm] — the one unknown solved
      delta_a  : always 0.0 — radial roller bearings (NU/N-type) carry no
                 axial load
      psi      : prescribed ring misalignment [rad]
      phi_Fr   : angle of resultant Fr in the global frame (output/plots only)
      delta_j  : roller-centreline deflection per roller (Z,) [mm],
                 eq.(38), BEFORE the lamina/profile correction — the
                 per-lamina deflection is delta_jk, not this
      alpha_j  : bearing.alpha_0 broadcast to (Z,) — for a cylindrical
                 roller the contact normal stays radial regardless of tilt,
                 unlike a ball's alpha_j, which genuinely varies with load
      Mz       : diagnostic reaction moment, eq.(46), evaluated at the
                 converged (delta_r, psi) — NOT a solve constraint here,
                 see roller_bearing.py's module docstring
      n_iter   : solver function evaluations
      residual : final ||R|| [N]
      ok       : solver convergence flag

    Extra fields (line contact only)
    ----------------------------------
    x_k       ndarray(n_s,) [mm]     lamina positions, eq.(38)-figure 3
    psi_j     ndarray(Z,)   [rad]    per-roller local misalignment, eq.(39)
    delta_jk  ndarray(Z,n_s)[mm]     per-lamina elastic deflection, eq.(41)
    q_jk      ndarray(Z,n_s)[N]      per-lamina contact force, eq.(36)
    """
    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
        "x_k", "psi_j", "delta_jk", "q_jk",
    )

    def __init__(self, *, delta_r, delta_a, psi, phi_Fr,
                 delta_j, alpha_j, Mz, n_iter, residual, ok,
                 x_k, psi_j, delta_jk, q_jk):
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
        self.x_k      = x_k
        self.psi_j    = psi_j
        self.delta_jk = delta_jk
        self.q_jk     = q_jk


# ---------------------------------------------------------------------------
# RollerLoadDistributionLibrary — LOCAL registry, CYLINDRICAL_ROLLER only
# ---------------------------------------------------------------------------

class RollerLoadDistributionLibrary:
    """
    Local registry of RollerLoadDistributionResult, keyed by bearing label —
    scoped to CYLINDRICAL_ROLLER bearings only. This class does not itself
    use LoadDistributionResult/the global library.py import above — it only
    stores and hands back RollerLoadDistributionResult instances someone
    else already built.

    This is NOT the final, cross-type registry — that is BearingResultsLibrary
    in the global library.py one level up, which also holds capacity/
    dynamic_equivalent_load/stiffness/extra for every bearing regardless of
    type. This one only ever holds the load distribution result for
    CYLINDRICAL_ROLLER bearings.

    Same shape as SimpleFEMResultsLibrary / BearingResultsLibrary
    (set/get/labels/__contains__/__iter__/__len__) — satisfies
    library.LoadDistributionLibrary structurally, without importing it.
    """

    def __init__(self):
        self._results: dict[str, RollerLoadDistributionResult] = {}

    def set(self, label: str, result: RollerLoadDistributionResult) -> None:
        self._results[label] = result

    def get(self, label: str) -> RollerLoadDistributionResult:
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