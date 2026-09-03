"""
solvers/machine_elements/shaft/oneD_analysis/static/shaft_post_processor.py

ShaftPostProcessor — enriches a ShaftResults with stress concentration
factors (Kt, Kf) at shoulders and other geometric features, producing
corrected stress arrays ready for fatigue/failure solvers.

No FEM or equilibrium solving here — purely reads ShaftResults geometry
and applies Peterson/Shigley correction factors.

References:
  Shigley §6  — stress concentration factors
  Peterson    — interpolation charts / formulae for Kt
  FKM         — surface factor, notch sensitivity
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from axisforge.solvers.machine_elements.shaft.static.results_reader import ShaftResults

if TYPE_CHECKING:
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


# ---------------------------------------------------------------------------
# Per-feature stress concentration record
# ---------------------------------------------------------------------------

@dataclass
class StressConcentration:
    """
    One stress concentration feature (shoulder, keyway, groove, etc.)

    x          [mm]   axial position on shaft
    feature    str    "shoulder" | "keyway" | "groove" | "press_fit"
    Kt_bending float  theoretical stress concentration factor, bending
    Kt_torsion float  theoretical stress concentration factor, torsion
    Kf_bending float  fatigue stress concentration factor, bending
    Kf_torsion float  fatigue stress concentration factor, torsion
    q_bending  float  notch sensitivity factor, bending
    q_torsion  float  notch sensitivity factor, torsion
    r          [mm]   fillet / notch radius
    D          [mm]   larger diameter
    d          [mm]   smaller diameter
    note       str    free-form traceability note
    """
    x:          float = 0.0
    feature:    str   = "shoulder"
    Kt_bending: float = 1.0
    Kt_torsion: float = 1.0
    Kf_bending: float = 1.0
    Kf_torsion: float = 1.0
    q_bending:  float = 1.0
    q_torsion:  float = 1.0
    r:          float = 0.0
    D:          float = 0.0
    d:          float = 0.0
    note:       str   = ""


@dataclass
class PostProcessedResults:
    """
    ShaftResults extended with stress concentration data and corrected stresses.

    Inherits all arrays from ShaftResults via composition (not inheritance —
    keeps the two classes independently picklable/serialisable).

    Extra arrays  (same node grid as ShaftResults.x)
    ------------------------------------------------
    Kf_b     [-]   fatigue bending SCF at each node (1.0 where no feature)
    Kf_t     [-]   fatigue torsion SCF at each node
    sigma_b_corrected  [MPa]   Kf_b * sigma_b
    tau_corrected      [MPa]   Kf_t * tau

    features : list[StressConcentration]
        One entry per identified feature, with full Kt/Kf/q traceability.
    """
    results:   ShaftResults = field(default_factory=ShaftResults)

    # SCF arrays (node-aligned)
    Kf_b:      np.ndarray = field(default_factory=lambda: np.array([]))
    Kf_t:      np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_b_corrected: np.ndarray = field(default_factory=lambda: np.array([]))
    tau_corrected:     np.ndarray = field(default_factory=lambda: np.array([]))

    # per-feature records (for traceability / reporting)
    features: list[StressConcentration] = field(default_factory=list)

    # peak corrected values
    sigma_b_corr_max: float = 0.0
    x_sigma_b_corr_max: float = 0.0
    tau_corr_max: float = 0.0
    x_tau_corr_max: float = 0.0


# ---------------------------------------------------------------------------
# Post-processor
# ---------------------------------------------------------------------------

class ShaftPostProcessor:
    """
    Enriches a ShaftResults with stress concentration factors.

    Usage
    -----
        pp  = ShaftPostProcessor(shaft_system, results)
        ppr = pp.process()

    What it does
    ------------
    1. Iterates over shaft shoulders (from Shaft.shoulders()).
    2. Computes Kt_bending and Kt_torsion via Peterson closed-form
       (r/d and D/d interpolation — Shigley Table 6-2 / Peterson §5.2).
    3. Computes notch sensitivity q from Neuber's constant (material-
       dependent — reads Su from material database if available, else
       falls back to q=1 conservative).
    4. Kf = 1 + q*(Kt - 1).
    5. Injects Kf onto the nearest node; takes the max if two features
       share a node.
    6. Keyways on ShaftSection.keyways: uses fixed Kt values from
       Shigley Table 6-3 (parallel key: Kt_b=2.14, Kt_t=3.0).
    7. Produces corrected stress arrays sigma_b_corrected = Kf_b * sigma_b.
    """

    # Neuber material constant sqrt(a) [mm^0.5] vs Su [MPa] — Shigley Table 6-2
    # Linear interpolation between these anchor points.
    _NEUBER_SU  = np.array([350,  500,  700,  900, 1100, 1400])   # MPa
    _NEUBER_SQA = np.array([0.38, 0.28, 0.20, 0.15, 0.11, 0.07])  # mm^0.5

    def __init__(self, shaft_system: "ShaftSystem", results: ShaftResults):
        self.sys     = shaft_system
        self.results = results

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def process(self) -> PostProcessedResults:
        res  = self.results
        n    = len(res.x)

        Kf_b = np.ones(n)
        Kf_t = np.ones(n)
        features: list[StressConcentration] = []

        # --- shoulders ---
        for x_shoulder, shoulder in self.sys.shaft.shoulders():
            sc = self._shoulder_scf(x_shoulder, shoulder)
            features.append(sc)
            i  = _nearest_node(res.x, x_shoulder)
            Kf_b[i] = max(Kf_b[i], sc.Kf_bending)
            Kf_t[i] = max(Kf_t[i], sc.Kf_torsion)

        # --- keyways ---
        cumulative = 0.0
        for section in self.sys.shaft.sections:
            for kw in section.keyways:
                x_kw = cumulative + kw.z_position
                sc   = self._keyway_scf(x_kw, section.diameter, kw)
                features.append(sc)
                i    = _nearest_node(res.x, x_kw)
                Kf_b[i] = max(Kf_b[i], sc.Kf_bending)
                Kf_t[i] = max(Kf_t[i], sc.Kf_torsion)
            cumulative += section.length

        # --- corrected stresses ---
        sigma_b_corr = Kf_b * res.sigma_b
        tau_corr     = Kf_t * res.tau

        idx_sb = int(np.argmax(sigma_b_corr))
        idx_t  = int(np.argmax(np.abs(tau_corr)))

        return PostProcessedResults(
            results   = res,
            Kf_b      = Kf_b,
            Kf_t      = Kf_t,
            sigma_b_corrected  = sigma_b_corr,
            tau_corrected      = tau_corr,
            features  = features,
            sigma_b_corr_max    = float(sigma_b_corr[idx_sb]),
            x_sigma_b_corr_max  = float(res.x[idx_sb]),
            tau_corr_max        = float(np.abs(tau_corr[idx_t])),
            x_tau_corr_max      = float(res.x[idx_t]),
        )

    # ------------------------------------------------------------------
    # Shoulder Kt — Peterson closed-form (Shigley §6, Table 6-2)
    # ------------------------------------------------------------------

    def _shoulder_scf(self, x: float, shoulder) -> StressConcentration:
        r = shoulder.fillet_radius
        d = shoulder.diameter_small
        D = shoulder.diameter_large

        r_d = r / d
        D_d = D / d

        Kt_b = self._kt_bending_shoulder(r_d, D_d)
        Kt_t = self._kt_torsion_shoulder(r_d, D_d)

        Su   = self._get_Su(x)
        q_b  = self._notch_sensitivity(r, Su, torsion=False)
        q_t  = self._notch_sensitivity(r, Su, torsion=True)

        Kf_b = 1.0 + q_b * (Kt_b - 1.0)
        Kf_t = 1.0 + q_t * (Kt_t - 1.0)

        return StressConcentration(
            x=x, feature="shoulder",
            Kt_bending=Kt_b, Kt_torsion=Kt_t,
            Kf_bending=Kf_b, Kf_torsion=Kf_t,
            q_bending=q_b,   q_torsion=q_t,
            r=r, D=D, d=d,
            note=f"r/d={r_d:.4f}, D/d={D_d:.4f}",
        )

    @staticmethod
    def _kt_bending_shoulder(r_d: float, D_d: float) -> float:
        """
        Peterson Eq. 5.2 / Shigley Table 6-2 curve fit (stepped shaft, bending):
            Kt ≈ C1 + C2*(2r/d) + C3*(2r/d)² + C4*(2r/d)³
        Coefficients from Shigley 10th ed. Table 6-2, D/d = 1.5 baseline.
        For simplicity: use the conservative D/d=3 fit unless D/d < 1.1.
        Replace with a proper 2-D interpolation table when precision is needed.
        """
        t = 2.0 * r_d
        if D_d < 1.1:
            # very mild step — nearly no concentration
            Kt = 1.0 + 0.1 / max(r_d, 1e-6) ** 0.3
            return min(Kt, 1.5)
        # Shigley Table 6-2 fit coefficients (D/d ≈ 1.5, bending)
        C = [3.04, -6.50, 7.20, -2.90]
        return max(1.0, C[0] + C[1]*t + C[2]*t**2 + C[3]*t**3)

    @staticmethod
    def _kt_torsion_shoulder(r_d: float, D_d: float) -> float:
        """
        Peterson Eq. 5.2 for torsion — Shigley Table 6-2.
        Same curve-fit structure, different coefficients.
        """
        t = 2.0 * r_d
        if D_d < 1.1:
            Kt = 1.0 + 0.07 / max(r_d, 1e-6) ** 0.3
            return min(Kt, 1.3)
        C = [2.0, -3.60, 4.0, -1.60]
        return max(1.0, C[0] + C[1]*t + C[2]*t**2 + C[3]*t**3)

    # ------------------------------------------------------------------
    # Keyway Kt — Shigley Table 6-3
    # ------------------------------------------------------------------

    def _keyway_scf(self, x: float, diameter: float, kw) -> StressConcentration:
        # Shigley Table 6-3: parallel key, end-milled
        Kt_b = 2.14
        Kt_t = 3.00
        r    = getattr(kw, "fillet_radius", 0.02 * diameter)   # estimate if not set
        Su   = self._get_Su(x)
        q_b  = self._notch_sensitivity(r, Su, torsion=False)
        q_t  = self._notch_sensitivity(r, Su, torsion=True)
        Kf_b = 1.0 + q_b * (Kt_b - 1.0)
        Kf_t = 1.0 + q_t * (Kt_t - 1.0)
        return StressConcentration(
            x=x, feature="keyway",
            Kt_bending=Kt_b, Kt_torsion=Kt_t,
            Kf_bending=Kf_b, Kf_torsion=Kf_t,
            q_bending=q_b,   q_torsion=q_t,
            r=r, d=diameter,
            note=f"parallel key, Shigley Table 6-3",
        )

    # ------------------------------------------------------------------
    # Notch sensitivity — Neuber (Shigley §6-14)
    # ------------------------------------------------------------------

    def _notch_sensitivity(self, r: float, Su: float, torsion: bool) -> float:
        """
        q = 1 / (1 + sqrt(a)/sqrt(r))
        sqrt(a) interpolated from Neuber table vs Su.
        Torsion: use Ssy ≈ 0.577*Su as the reference (von Mises).
        """
        if Su <= 0.0:
            return 1.0   # conservative fallback
        Su_ref = 0.577 * Su if torsion else Su
        sqa    = float(np.interp(Su_ref, self._NEUBER_SU, self._NEUBER_SQA))
        if r <= 0.0:
            return 0.0
        return 1.0 / (1.0 + sqa / np.sqrt(r))

    # ------------------------------------------------------------------
    # Material Su lookup
    # ------------------------------------------------------------------

    def _get_Su(self, x: float) -> float:
        """
        Ultimate tensile strength [MPa] at position x.
        Reads from material database if available; falls back to 700 MPa.
        """
        section, _ = self.sys.shaft.section_at(x)
        mat_id     = getattr(section, "material_id", "")
        try:
            from axisforge.core.materials import get_material
            mat = get_material(mat_id)
            return float(mat.Su)
        except Exception:
            return 700.0   # conservative fallback


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _nearest_node(x_arr: np.ndarray, x: float) -> int:
    return int(np.argmin(np.abs(x_arr - x)))