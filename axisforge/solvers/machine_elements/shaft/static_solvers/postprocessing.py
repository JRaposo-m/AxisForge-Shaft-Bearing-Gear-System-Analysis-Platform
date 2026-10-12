"""
Stress concentration factors (Kt, q, Kf) at shaft shoulders and keyways, and the
corrected stress arrays built from them.

``ShaftPostProcessor`` reads a ``ShaftResults`` and the shaft geometry and
multiplies the nominal stresses by Kf at the node nearest to each feature. It
solves no FEM or equilibrium problem and implements no failure criterion. It has
no caller outside this file. The sources named in the code (Shigley, Peterson,
Neuber) are not in the library; see the References of each member.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.global_postprocessing import ShaftResults

if TYPE_CHECKING:
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


# ---------------------------------------------------------------------------
# Per-feature stress concentration record
# ---------------------------------------------------------------------------

@dataclass
class StressConcentration:
    """
    Record of one stress concentration feature (shoulder or keyway).

    Traceability record of the factors computed for one feature. The code
    produces only ``"shoulder"`` and ``"keyway"``.

    Parameters
    ----------
    x : float
        Axial position on the shaft [mm].
    feature : str
        ``"shoulder"`` or ``"keyway"`` (the only values the code produces; the
        field is a free string).
    Kt_bending, Kt_torsion : float
        Theoretical stress concentration factor, bending and torsion [-].
    Kf_bending, Kf_torsion : float
        Fatigue stress concentration factor, bending and torsion [-].
    q_bending, q_torsion : float
        Notch sensitivity, bending and torsion [-].
    r : float
        Fillet / notch radius [mm].
    D, d : float
        Larger and smaller diameter [mm] (``D`` stays 0 for keyways).
    note : str
        Free-form traceability note.
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
    Shaft results together with the stress concentration factors and corrected stresses.

    Holds the original ``ShaftResults`` in the field ``results`` (it is not a
    subclass) and node-aligned arrays on the same grid as ``results.x``.

    Parameters
    ----------
    results : ShaftResults
        The nominal results the factors were applied to.
    Kf_b, Kf_t : numpy.ndarray
        Fatigue stress concentration factor, bending and torsion, at each node
        [-]; 1.0 where there is no feature.
    sigma_b_corrected : numpy.ndarray
        ``Kf_b * results.sigma_b`` [MPa].
    tau_corrected : numpy.ndarray
        ``Kf_t * results.tau`` [MPa].
    features : list of StressConcentration
        One entry per feature, with its Kt, q and Kf.
    sigma_b_corr_max : float
        Maximum of ``sigma_b_corrected`` (signed maximum) [MPa].
    x_sigma_b_corr_max : float
        Axial position of that maximum [mm].
    tau_corr_max : float
        Maximum of ``abs(tau_corrected)`` [MPa].
    x_tau_corr_max : float
        Axial position of that maximum [mm].
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
    Applies stress concentration factors at shoulders and keyways to the nominal stresses.

    For each shoulder (``Shaft.shoulders()``) and each keyway of each section it
    computes Kt, the notch sensitivity q and Kf, assigns Kf to the node nearest to
    the feature (the larger value if two features share a node) and multiplies
    the nominal stresses by it. No failure criterion, equivalent stress or safety
    factor is computed here (``static_failure.py`` is empty).

    Parameters
    ----------
    shaft_system : ShaftSystem
        Provides ``shaft.shoulders()``, ``shaft.sections`` (``diameter``,
        ``length``, ``keyways``, ``material_id``) and ``shaft.section_at(x)``.
        Lengths and diameters [mm].
    results : ShaftResults
        Nominal results. Used: ``x`` [mm], ``sigma_b`` [MPa] (bending stress) and
        ``tau`` [MPa] (torsional shear stress). ``sigma_b`` [MPa] = resultant
        bending moment over W, always >= 0 (``build_shaft_result``,
        global_postprocessing.py); ``tau`` [MPa] = T [N.m] * 1000 / Wt [mm^3],
        signed (``TorsionSolver``, torsion.py). No unit is converted here.

    Warning
    -------
    The material strength lookup, the keyway inputs and the shoulder Kt formula
    differ from what their names suggest; see ``_get_Su``, ``_keyway_scf`` and
    ``_kt_bending_shoulder``.

    Notes
    -----
    Per feature (repeated at the method that computes each term)::

        Kf = 1 + q * (Kt - 1)
        sigma_b_corrected = Kf_b * sigma_b
        tau_corrected     = Kf_t * tau

    Neuber table: ``_NEUBER_SU`` [MPa] and ``_NEUBER_SQA`` [mm^0.5], interpolated
    linearly by ``numpy.interp`` (clamped to the end values outside 350-1400 MPa).

    References
    ----------
    Shigley Table 6-2, for the Neuber constants, named in the comments and
    docstrings of the code; not in the library; not checked against the source
    ((source to find)).
    """

    # Neuber material constant sqrt(a) [mm^0.5] vs Su [MPa]; the code names
    # "Shigley Table 6-2" as the origin (not checked).
    _NEUBER_SU  = np.array([350,  500,  700,  900, 1100, 1400])   # MPa
    _NEUBER_SQA = np.array([0.38, 0.28, 0.20, 0.15, 0.11, 0.07])  # mm^0.5

    def __init__(self, shaft_system: "ShaftSystem", results: ShaftResults):
        """Store the arguments; see the class docstring."""
        self.sys     = shaft_system
        self.results = results

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def process(self) -> PostProcessedResults:
        """
        Compute the factors of all features and the corrected stress arrays.

        Returns
        -------
        PostProcessedResults
            Kf arrays [-], corrected stresses [MPa], feature records and the
            peak corrected stresses with their positions [mm].

        Warning
        -------
        The reference of ``Keyway.z_position`` is not established (the ``Keyway``
        docstring leaves it open). As coded here it is read as measured from the
        left face of the owning section. If it is meant from the left face of
        the first section, ``x_kw`` counts the start of the section twice for
        every keyway not in the first section. ``_get_Su`` calls ``section_at(x_kw)``
        outside its ``try`` block; it raises ``ValueError`` when ``x_kw`` is
        outside [0, L] by more than ``TOL_GEOMETRY_mm``.

        Notes
        -----
        At the node ``i`` nearest to a feature, ``Kf_b[i] = max(Kf_b[i],
        Kf_bending)`` (same for torsion); Kf = 1 elsewhere. Then::

            sigma_b_corrected = Kf_b * sigma_b          [MPa]
            tau_corrected     = Kf_t * tau              [MPa]
            peak sigma_b: argmax(sigma_b_corrected)     (signed)
            peak tau:     argmax(abs(tau_corrected))

        Keyway axial position: ``x_kw = cumulative + kw.z_position``, where
        ``cumulative`` is the sum of the lengths of the previous sections [mm].

        The nominal stress at a shoulder node is the one in ``results``: computed
        with W and Wt of the section to the right of the boundary
        (``Shaft.section_at``). For a step up (larger diameter on the right) Kf
        therefore multiplies the stress of the larger section, although Kt is
        computed from r/d of the smaller one. Kf is applied at one node only
        (the nearest); a keyway is not spread over its length.
        """
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
    # Shoulder Kt (curve fits named after Peterson / Shigley Table 6-2)
    # ------------------------------------------------------------------

    def _shoulder_scf(self, x: float, shoulder) -> StressConcentration:
        """
        Kt, q and Kf of one shoulder fillet, bending and torsion.

        Parameters
        ----------
        x : float
            Axial position of the shoulder [mm].
        shoulder : Shoulder
            Provides ``fillet_radius`` r, ``diameter_small`` d and
            ``diameter_large`` D [mm].

        Returns
        -------
        StressConcentration
            Kt, q, Kf [-]; ``note`` holds r/d and D/d.

        Notes
        -----
        ``r/d = r / d``, ``D/d = D / d`` [-]; Kt from ``_kt_bending_shoulder`` and
        ``_kt_torsion_shoulder``; q from ``_notch_sensitivity(r, Su)`` with Su from
        ``_get_Su(x)`` [MPa]; ``Kf = 1 + q (Kt - 1)``, bending and torsion.
        """
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
        Theoretical stress concentration factor of a shoulder fillet in bending.

        Parameters
        ----------
        r_d : float
            Fillet radius over small diameter, r/d [-].
        D_d : float
            Large over small diameter, D/d [-].

        Returns
        -------
        Kt : float
            Theoretical stress concentration factor, bending [-].

        Warning
        -------
        The coefficients and the power-law branch are not verified against any
        source.

        Notes
        -----
        With ``t = 2 r/d``:

        - if ``D/d < 1.1``: ``Kt = min(1 + 0.1 / max(r/d, 1e-6)**0.3, 1.5)``;
        - otherwise ``Kt = max(1, 3.04 - 6.50 t + 7.20 t**2 - 2.90 t**3)``.

        Above ``D/d = 1.1`` the result does not depend on D/d. The origin of the
        coefficients is not established. The comment at the cubic names "Shigley
        Table 6-2 fit coefficients (D/d ~ 1.5, bending)".

        Kt is discontinuous at ``D/d = 1.1``. From the coded formula, the cubic
        gives a finite Kt as r -> 0 (3.04) and is clamped to 1.0 for
        r/d >= 0.368. No validity range of r/d is stated in the code.

        References
        ----------
        Named by the original author, not checked (not in the library; source to
        find): the comment at the cubic, "Shigley Table 6-2 fit coefficients
        (D/d ~ 1.5, bending)"; the docstring at commit b3f624a, "Peterson Eq. 5.2 /
        Shigley Table 6-2 curve fit", "Shigley 10th ed. Table 6-2, D/d = 1.5
        baseline" and "the conservative D/d=3 fit" (the last two contradict each
        other).
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
        Theoretical stress concentration factor of a shoulder fillet in torsion.

        Parameters
        ----------
        r_d : float
            Fillet radius over small diameter, r/d [-].
        D_d : float
            Large over small diameter, D/d [-].

        Returns
        -------
        Kt : float
            Theoretical stress concentration factor, torsion [-].

        Notes
        -----
        With ``t = 2 r/d``:

        - if ``D/d < 1.1``: ``Kt = min(1 + 0.07 / max(r/d, 1e-6)**0.3, 1.3)``;
        - otherwise ``Kt = max(1, 2.0 - 3.60 t + 4.0 t**2 - 1.60 t**3)``.

        Independent of D/d above 1.1. The origin of the coefficients is not
        established.

        Kt is discontinuous at ``D/d = 1.1``. From the coded formula, the cubic
        gives a finite Kt as r -> 0 (2.0) and is clamped to 1.0 for
        r/d >= 0.25. No validity range of r/d is stated in the code.

        References
        ----------
        Named by the original author, not checked (not in the library; source to
        find): the section comment "Peterson / Shigley Table 6-2"; the docstring
        at commit b3f624a, "Peterson Eq. 5.2 for torsion - Shigley Table 6-2".
        """
        t = 2.0 * r_d
        if D_d < 1.1:
            Kt = 1.0 + 0.07 / max(r_d, 1e-6) ** 0.3
            return min(Kt, 1.3)
        C = [2.0, -3.60, 4.0, -1.60]
        return max(1.0, C[0] + C[1]*t + C[2]*t**2 + C[3]*t**3)

    # ------------------------------------------------------------------
    # Keyway Kt (fixed values named after Shigley Table 6-3)
    # ------------------------------------------------------------------

    def _keyway_scf(self, x: float, diameter: float, kw) -> StressConcentration:
        """
        Kt, q and Kf of a keyway, bending and torsion.

        Parameters
        ----------
        x : float
            Axial position of the keyway, as computed by ``process`` [mm].
        diameter : float
            Diameter of the section carrying the keyway [mm].
        kw : Keyway
            The keyway (only ``fillet_radius``, if present, is read).

        Returns
        -------
        StressConcentration
            ``feature="keyway"``, ``d = diameter``, ``D`` left at 0.

        Warning
        -------
        ``Keyway`` has no ``fillet_radius`` attribute, so the radius is always
        0.02 d [mm]. ``Keyway.Kf_bending`` and ``Keyway.Kf_torsion`` are ignored,
        and Kt is fixed whatever the keyway type (parallel or Woodruff) and
        dimensions. The record's ``note`` says "parallel key, Shigley Table 6-3"
        for every keyway type.

        Notes
        -----
        ``Kt_b = 2.14``, ``Kt_t = 3.00`` (fixed); root radius
        ``r = getattr(kw, "fillet_radius", 0.02 * diameter)`` [mm]; q from
        ``_notch_sensitivity(r, Su)``; ``Kf = 1 + q (Kt - 1)``. The code labels
        the Kt values "Shigley Table 6-3: parallel key, end-milled".

        References
        ----------
        Shigley Table 6-3, named in the comments and docstrings of the
        code; not in the library; not checked against the source ((source to
        find)).
        """
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
    # Notch sensitivity — Neuber (named in the code: Shigley §6-14)
    # ------------------------------------------------------------------

    def _notch_sensitivity(self, r: float, Su: float, torsion: bool) -> float:
        """
        Notch sensitivity from the Neuber constant.

        Parameters
        ----------
        r : float
            Fillet / notch radius [mm].
        Su : float
            Ultimate tensile strength [MPa].
        torsion : bool
            True for torsion, False for bending.

        Returns
        -------
        q : float
            Notch sensitivity [-].

        Warning
        -------
        For torsion the table, which is indexed by Su, is read at 0.577 Su.
        Whether the sources named support this is not checked (source to find).
        ``neuber_constant_sqrt_a`` (solvers/machine_elements/shaft/utils.py),
        which names the same book, gives the opposite trend: at Su = 700 MPa,
        sqrt(a) = 0.307 (bending) and 0.234 (torsion) mm^0.5, against 0.20 and
        0.344 here. The two implementations disagree; neither is checked against
        a source.

        Notes
        -----
        ``q = 1 / (1 + sqrt(a) / sqrt(r))``. ``sqrt(a)`` [mm^0.5] is interpolated
        in ``_NEUBER_SU`` / ``_NEUBER_SQA`` at ``Su_ref = Su`` (bending) or
        ``Su_ref = 0.577 Su`` (torsion; the original author's docstring called it
        "Ssy ~ 0.577*Su ... (von Mises)").
        Returns 1.0 if ``Su <= 0`` and 0.0 if ``r <= 0`` (the ``Su`` test comes
        first).

        Consequence of the 0.577 factor: sqrt(a) is larger in torsion than in
        bending, so q (torsion) < q (bending) for the same r; from the coded
        formula, for Su < 606.6 MPa the torsion lookup is clamped at
        0.38 mm^0.5. With Su = 700 MPa (the value ``_get_Su`` always returns):
        sqrt(a) = 0.20 (bending) and 0.344 (torsion) mm^0.5.

        References
        ----------
        Neuber constants "Shigley Table 6-2" and "Neuber (Shigley 6-14)", named in
        the comments and docstrings of the code; not in the library; not checked
        against the source ((source to find)).
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
        Ultimate tensile strength of the section at position x.

        Parameters
        ----------
        x : float
            Axial position [mm].

        Returns
        -------
        Su : float
            Ultimate tensile strength [MPa].

        Warning
        -------
        ``_get_Su`` always returns 700.0 MPa. AxisForge registers no material, so
        ``get_material(material_id)`` raises ``KeyError``; for a material
        registered by the caller, ``mat.Su`` raises ``AttributeError`` because
        ``Material`` stores the ultimate strength as ``strength.Sut``
        (core/materials/base.py). Both are swallowed by ``except Exception``.
        ``section_at(x)`` is called outside the ``try`` and raises ``ValueError``
        for x outside [0, L] (beyond ``TOL_GEOMETRY_mm``).
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
    """Index of the node of ``x_arr`` nearest to ``x`` (same length unit, [mm])."""
    return int(np.argmin(np.abs(x_arr - x)))